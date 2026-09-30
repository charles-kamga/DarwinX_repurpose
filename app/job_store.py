"""Module de gestion et de persistance atomique de la machine à états de job.

Ce module gère le cycle de vie complet des jobs de traitement vidéo :
  queued -> processing (avec suivi d'étape) -> completed | failed

Conformément à AGENTS.md (Principe n°3) et SKILL.md :
- Chaque mise à jour de statut est persistée sur disque au format JSON.
- Écriture atomique systématique (fichier temporaire + os.replace) pour
  éviter toute corruption de fichier en cas de crash intempestif du process.
- Verrouillage scindé par job_id (via filelock) protégeant l'intégralité du
  cycle lecture-modification-écriture contre les accès concurrents.
- Timeout explicite sur l'acquisition du verrou, avec levée de JobStoreError.
- Traçabilité complète des étapes accomplies, des avertissements et des erreurs.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import uuid

from filelock import FileLock, Timeout

from app import config

logger = logging.getLogger(__name__)


class JobStoreError(Exception):
    """Exception de base pour les erreurs du magasin de jobs."""


def _get_job_path(job_id: str) -> Path:
    """Retourne le chemin absolu du fichier JSON associé au job."""
    jobs_dir = Path(config.JOBS_DIR)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / f"{job_id}.json"


def _get_lock_path(job_id: str) -> Path:
    """Retourne le chemin du fichier verrou dédié au job."""
    jobs_dir = Path(config.JOBS_DIR)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir / f"{job_id}.lock"


@contextmanager
def _job_lock(job_id: str):
    """Context manager acquérant un verrou exclusif inter-processus pour un job donné."""
    lock_path = _get_lock_path(job_id)
    timeout = config.JOB_LOCK_TIMEOUT_SECONDS
    lock = FileLock(str(lock_path), timeout=timeout)
    try:
        with lock:
            yield
    except Timeout as exc:
        raise JobStoreError(
            f"Timeout ({timeout}s) lors de l'acquisition du verrou pour le job {job_id}"
        ) from exc


def _atomic_write_job(job_path: Path, data: Dict[str, Any]) -> None:
    """Écrit de manière atomique un dictionnaire de job sur disque.

    Écrit d'abord dans un fichier temporaire unique dans le même répertoire,
    puis effectue un remplacement atomique (os.replace).
    """
    directory = job_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    tmp_path = directory / f".tmp_{job_path.stem}_{uuid.uuid4().hex[:8]}.json"
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, job_path)
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise JobStoreError(f"Échec d'écriture atomique du job {job_path.name} : {exc}") from exc


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Charge et retourne les données d'un job depuis le disque.

    Returns:
        Dictionnaire des données du job ou None si le job n'existe pas.
    """
    job_path = _get_job_path(job_id)
    if not job_path.is_file():
        return None

    try:
        content = job_path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as exc:
        logger.error("Lecture impossible ou JSON corrompu pour le job %s : %s", job_id, exc)
        return None


def _atomic_modify_job(job_id: str, modifier_fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    """Exécute une mutation atomique d'un job sous verrou exclusif par job_id.

    Le verrou est acquis AVANT la lecture de l'état frais et relâché APRÈS
    l'écriture atomique sur disque, éliminant toute course lecture-modification-écriture.
    """
    with _job_lock(job_id):
        data = get_job(job_id)
        if data is None:
            raise JobStoreError(f"Job inexistant : {job_id}")

        modifier_fn(data)
        data["updated_at"] = time.time()
        _atomic_write_job(_get_job_path(job_id), data)
        return data


def create_job(source_path: str, job_id: Optional[str] = None) -> str:
    """Initialise et persiste un nouveau job en statut 'queued'.

    Args:
        source_path: Chemin du fichier média source à traiter.
        job_id: Identifiant unique optionnel (généré par défaut si omis).

    Returns:
        L'identifiant unique du job créé.
    """
    actual_job_id = job_id or str(uuid.uuid4())
    job_path = _get_job_path(actual_job_id)

    now = time.time()
    job_data: Dict[str, Any] = {
        "job_id": actual_job_id,
        "source_path": str(source_path),
        "status": "queued",
        "current_step": None,
        "steps_done": [],
        "warnings": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }

    with _job_lock(actual_job_id):
        _atomic_write_job(job_path, job_data)

    logger.info("Job %s créé avec succès (statut: queued)", actual_job_id)
    return actual_job_id


def _update_job(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Applique une mise à jour partielle sur le job et le persiste atomiquement."""
    def _modifier(data: Dict[str, Any]) -> None:
        data.update(updates)

    return _atomic_modify_job(job_id, _modifier)


def set_step(job_id: str, step_name: str) -> None:
    """Bascule le statut en 'processing' et enregistre l'étape active."""
    def _modifier(data: Dict[str, Any]) -> None:
        data["status"] = "processing"
        data["current_step"] = step_name

    _atomic_modify_job(job_id, _modifier)
    logger.info("Job %s -> étape en cours : %s", job_id, step_name)


def mark_step_done(job_id: str, step_name: str) -> None:
    """Marque une étape comme achevée et l'ajoute à l'historique steps_done."""
    def _modifier(data: Dict[str, Any]) -> None:
        steps_done: List[str] = list(data.get("steps_done", []))
        if step_name not in steps_done:
            steps_done.append(step_name)
        data["steps_done"] = steps_done
        data["current_step"] = None

    _atomic_modify_job(job_id, _modifier)
    logger.info("Job %s -> étape achevée : %s", job_id, step_name)


def add_warning(job_id: str, message: str) -> None:
    """Consigne un avertissement de dégradation gracieuse dans le job."""
    def _modifier(data: Dict[str, Any]) -> None:
        warnings: List[str] = list(data.get("warnings", []))
        warnings.append(message)
        data["warnings"] = warnings

    _atomic_modify_job(job_id, _modifier)
    logger.warning("Job %s [Avertissement] %s", job_id, message)


def mark_completed(job_id: str, result: Dict[str, Any]) -> None:
    """Bascule le job en statut 'completed' avec ses résultats finaux."""
    def _modifier(data: Dict[str, Any]) -> None:
        data["status"] = "completed"
        data["current_step"] = None
        data["result"] = result

    _atomic_modify_job(job_id, _modifier)
    logger.info("Job %s terminé avec succès (statut: completed)", job_id)


def mark_failed(job_id: str, error: str) -> None:
    """Bascule le job en statut 'failed' avec le message d'erreur associé."""
    def _modifier(data: Dict[str, Any]) -> None:
        data["status"] = "failed"
        data["error"] = error

    _atomic_modify_job(job_id, _modifier)
    logger.error("Job %s marqué en échec (statut: failed) : %s", job_id, error)


def purge_old_jobs(max_age_days: Optional[int] = None) -> int:
    """Supprime les fichiers JSON et verrous de jobs plus anciens que max_age_days.

    Args:
        max_age_days: Âge maximal en jours (par défaut config.JOB_RETENTION_DAYS).

    Returns:
        Nombre de fichiers supprimés.
    """
    days = max_age_days if max_age_days is not None else config.JOB_RETENTION_DAYS
    cutoff_time = time.time() - (days * 86400)
    jobs_dir = Path(config.JOBS_DIR)
    if not jobs_dir.is_dir():
        return 0

    deleted_count = 0
    for p in jobs_dir.glob("*"):
        if p.is_file() and (p.suffix in (".json", ".lock")):
            try:
                if p.stat().st_mtime < cutoff_time:
                    p.unlink()
                    deleted_count += 1
            except OSError as err:
                logger.warning("Impossible de purger %s : %s", p, err)

    logger.info("Purge des vieux jobs : %d fichier(s) supprimé(s) (> %d jours)", deleted_count, days)
    return deleted_count

