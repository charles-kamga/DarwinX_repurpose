"""Module de transcription audio/vidéo avec horodatage mot-par-mot via faster-whisper.

Ce module extrait la transcription complète et les timestamps précis de chaque mot
d'un fichier média pour alimenter les étapes ultérieures du pipeline :
scoring LLM des moments forts, découpage précis des clips et sous-titrage animé.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import faster_whisper
except ImportError:
    faster_whisper = None  # type: ignore

from app import config

logger = logging.getLogger(__name__)

_last_run_from_cache: bool = False
_last_run_device: str = "auto"


def was_last_from_cache() -> bool:
    """Indique si la dernière transcription a été servie depuis le cache disque."""
    return _last_run_from_cache


def get_last_device_used() -> str:
    """Retourne la plateforme matérielle utilisée lors de la dernière transcription."""
    return _last_run_device


class TranscriberError(Exception):
    """Exception de base pour les erreurs du module de transcription."""


class EmptyTranscriptionError(TranscriberError):
    """Levée lorsqu'aucune parole n'est détectée dans le fichier média (échec bloquant)."""


class TranscriberMediaError(TranscriberError):
    """Levée lorsque le fichier média est illisible ou corrompu."""


def _compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Calcule le hash SHA-256 du contenu d'un fichier par blocs de 64 Ko.
    
    Pourquoi par blocs : évite de charger des vidéos entières de plusieurs Go en mémoire RAM.
    Pourquoi le hash du contenu : empêche les collisions silencieuses de cache si deux
    vidéos différentes partagent le même nom de fichier (ex: episode.mp4).
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _setup_nvidia_cuda_paths() -> None:
    """Précharge les bibliothèques CUDA/cuBLAS installées via pip (nvidia-*-cu12).
    
    Pourquoi : sous Linux, pip installe les bibliothèques CUDA dans site-packages/nvidia/*/lib/,
    qui n'est pas dans le LD_LIBRARY_PATH système. Le préchargement via ctypes avec
    mode=RTLD_GLOBAL permet à ctranslate2 de trouver libcublas.so.12 dynamiquement.
    """
    try:
        import ctypes
        import site
        base_dirs = []
        try:
            base_dirs.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            base_dirs.append(site.getusersitepackages())
        except Exception:
            pass
        proj_root = Path(__file__).resolve().parent.parent
        for venv_site in proj_root.glob(".venv/lib/python*/site-packages"):
            if venv_site.is_dir():
                base_dirs.append(str(venv_site))

        for base in base_dirs:
            nvidia_dir = Path(base) / "nvidia"
            if nvidia_dir.is_dir():
                for so_file in sorted(nvidia_dir.glob("*/lib/*.so*")):
                    try:
                        ctypes.CDLL(str(so_file), mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
    except Exception:
        pass


def _detect_device_and_compute_type() -> Tuple[str, str]:
    """Détecte automatiquement le matériel disponible pour faster-whisper.
    
    Supporte un override explicite via config.WHISPER_DEVICE :
    - "cpu"  : force CPU avec int8, sans tenter la détection GPU.
    - "cuda" : force GPU avec float16.
    - "auto" (défaut) : auto-détection CUDA via ctranslate2/torch, repli CPU sinon.
    
    Pourquoi cet override : la détection automatique vérifie la présence d'un GPU,
    pas si l'inférence GPU fonctionne réellement à l'exécution (ex. libcublas.so
    manquant). L'override permet un contrôle manuel sans modifier le code.
    """
    device_override = config.WHISPER_DEVICE.lower().strip()

    if device_override == "cpu":
        return "cpu", "int8"

    # Précharge les bibliothèques CUDA pip si disponibles
    _setup_nvidia_cuda_paths()

    if device_override == "cuda":
        return "cuda", "float16"

    # Mode "auto" : comportement d'auto-détection existant
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass

    return "cpu", "int8"


def _get_cache_dir() -> Path:
    """Retourne le répertoire de cache sur disque."""
    cache_dir = Path(os.environ.get("TRANSCRIPT_CACHE_DIR", ".cache/transcripts"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_cache_path(content_hash: str, model_size: str, language: Optional[str]) -> Path:
    """Génère le chemin du fichier cache intégrant hash, modèle et langue."""
    lang_tag = language if language else "auto"
    return _get_cache_dir() / f"{content_hash}_{model_size}_{lang_tag}.json"


def _load_valid_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Tente de charger et valider un fichier cache.
    
    Distingue explicitement les erreurs I/O / JSON corrompu des vérifications de structure :
    - Fichier illisible ou JSON malformé -> journalisé comme corruption de cache.
    - Structure invalide (0 segments, clés manquantes) -> invalidé et retranscrit.
    """
    if not cache_path.is_file():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as io_exc:
        logger.warning("Fichier de cache corrompu ou illisible %s : %s -> retranscription", cache_path, io_exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Structure racine du cache non-dict dans %s -> retranscription", cache_path)
        return None

    duration, segments = data.get("duration"), data.get("segments")
    if duration is None or not isinstance(duration, (int, float)) or duration <= 0:
        logger.warning("Durée invalide ou manquante dans le cache %s -> retranscription", cache_path)
        return None

    if not isinstance(segments, list) or len(segments) == 0:
        logger.warning("Cache sans segments (%s) -> retranscription", cache_path)
        return None

    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict) or not all(k in seg for k in ("start", "end", "text", "words")):
            logger.warning("Segment #%d invalide ou incomplet dans le cache %s -> retranscription", idx, cache_path)
            return None
        if not isinstance(seg["words"], list):
            logger.warning("Format de words invalide dans le segment #%d (%s) -> retranscription", idx, cache_path)
            return None

    return data


def _save_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    """Sauvegarde le résultat de transcription dans le fichier de cache."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Impossible d'écrire le cache %s : %s", cache_path, exc)


def transcribe(
    media_path: str,
    language: Optional[str] = None,
    force: bool = False,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcrit un fichier audio ou vidéo avec timestamps mot-par-mot via faster-whisper.

    Args:
        media_path: Chemin vers le fichier média source (audio/vidéo).
        language: Code langue optionnel (ex: 'fr', 'en'). Si None, auto-détecté par Whisper.
        force: Si True, ignore le cache existant et force une nouvelle transcription.
        model_size: Taille du modèle ('tiny', 'base', 'small', 'medium', 'large-v3').
                    Par défaut, utilise 'medium' (ou WHISPER_MODEL_SIZE si défini)
                    afin de garantir une qualité optimale pour le scoring LLM en aval.

    Returns:
        Dictionnaire conforme au contrat du projet :
        {
            "duration": float,
            "segments": [
                {
                    "start": float,
                    "end": float,
                    "text": str,
                    "words": [{"w": str, "s": float, "e": float}, ...]
                }
            ]
        }

    Raises:
        FileNotFoundError: Si media_path n'existe pas.
        EmptyTranscriptionError: Si aucune parole n'est détectée dans le fichier (échec bloquant).
        TranscriberMediaError: Si le fichier est illisible ou corrompu.
        TranscriberError: En cas d'autre erreur inattendue lors de la transcription.
    """
    path = Path(media_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier média introuvable : {media_path}")

    selected_model_size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "medium")

    # Calcul du hash SHA-256 pour indexer le cache de façon unique et sans collision
    try:
        content_hash = _compute_file_sha256(path)
    except Exception as exc:
        raise TranscriberMediaError(f"Impossible de lire le fichier média '{media_path}': {exc}") from exc

    global _last_run_from_cache, _last_run_device

    cache_file = _get_cache_path(content_hash, selected_model_size, language)

    # Récupération du cache existant si non forcé
    if not force:
        cached_result = _load_valid_cache(cache_file)
        if cached_result is not None:
            logger.info("Transcription récupérée depuis le cache : %s", cache_file)
            _last_run_from_cache = True
            _last_run_device = "Cache disque"
            return cached_result

    # Détection du matériel : GPU (float16) si disponible, sinon CPU (int8)
    device, compute_type = _detect_device_and_compute_type()
    logger.info(
        "Initialisation de WhisperModel(model='%s', device='%s', compute_type='%s')",
        selected_model_size,
        device,
        compute_type,
    )

    if faster_whisper is None:
        raise TranscriberError("Le package 'faster-whisper' n'est pas installé dans l'environnement.")

    try:
        model = faster_whisper.WhisperModel(
            selected_model_size,
            device=device,
            compute_type=compute_type,
        )
    except Exception as exc:
        raise TranscriberError(f"Erreur d'initialisation du modèle Whisper : {exc}") from exc

    try:
        # vad_filter=True : filtre les zones silencieuses/parasites, évite les hallucinations
        # word_timestamps=True : indispensable pour le sous-titrage et le calage fin des cuts
        segments_generator, info = model.transcribe(
            str(path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        segments_list: List[Dict[str, Any]] = []
        for segment in segments_generator:
            words_list: List[Dict[str, Any]] = []
            if segment.words:
                for w in segment.words:
                    # Dégradation gracieuse (AGENTS.md) : si un mot a un timestamp None
                    # (ex: bord de segment / VAD agressif), on l'ignore de la liste des mots
                    # plutôt que de faire échouer tout le clip sur TypeError: float(None).
                    if w.start is None or w.end is None:
                        logger.warning(
                            "Timestamp None sur le mot '%s' (start=%s, end=%s) du segment [%s, %s] "
                            "-> mot exclu du sous-titrage fin (repli sur l'intervalle du segment)",
                            getattr(w, "word", ""),
                            getattr(w, "start", None),
                            getattr(w, "end", None),
                            getattr(segment, "start", None),
                            getattr(segment, "end", None),
                        )
                        continue

                    try:
                        words_list.append({
                            "w": str(w.word),
                            "s": round(float(w.start), 3),
                            "e": round(float(w.end), 3),
                        })
                    except (ValueError, TypeError) as conv_err:
                        logger.warning(
                            "Timestamp invalide pour le mot '%s' (%s) -> ignoré",
                            getattr(w, "word", ""),
                            conv_err,
                        )
                        continue

            cleaned_text = segment.text.strip()
            seg_start = round(float(segment.start), 3) if segment.start is not None else 0.0
            seg_end = (
                round(float(segment.end), 3)
                if segment.end is not None
                else (words_list[-1]["e"] if words_list else seg_start)
            )

            segments_list.append({
                "start": seg_start,
                "end": seg_end,
                "text": cleaned_text,
                "words": words_list,
            })
    except Exception as exc:
        raise TranscriberMediaError(f"Échec de transcription pour '{media_path}': {exc}") from exc

    # Cas d'échec bloquant : aucune parole détectée (0 segments)
    if not segments_list:
        raise EmptyTranscriptionError(
            f"Aucune parole détectée dans le fichier média '{media_path}'. "
            "Le pipeline ne peut pas continuer sans transcription."
        )

    # Durée totale issue des métadonnées Whisper ou calculée sur le dernier segment
    audio_duration = getattr(info, "duration", None)
    if audio_duration is None or audio_duration <= 0:
        audio_duration = segments_list[-1]["end"]

    result: Dict[str, Any] = {
        "duration": round(float(audio_duration), 3),
        "segments": segments_list,
    }

    # Mise en cache du résultat validé
    _save_cache(cache_file, result)

    _last_run_from_cache = False
    _last_run_device = f"{device.upper()} ({compute_type})"
    return result
