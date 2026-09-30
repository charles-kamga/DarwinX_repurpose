"""Point d'entrée CLI d'intégration du pipeline de repurposing vidéo (Darwinx_Repurpose).

Permet de lancer le pipeline de bout en bout sur une vidéo locale ou une URL YouTube :
  python main.py <url_ou_chemin_local> [--out-dir DIR] [--max-height HEIGHT]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Optional

from app import __version__, config, downloader, job_store, pipeline
from app.downloader import DownloaderError
from app.job_store import JobStoreError
from app.pipeline import PipelineFatalError, PipelineUnexpectedError
from app.ui import ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("repurpose_cli")


def _is_url(source: str) -> bool:
    """Détermine si la source fournie ressemble à une URL plutôt qu'un chemin de fichier."""
    source_lower = source.lower()
    return (
        source_lower.startswith("http://")
        or source_lower.startswith("https://")
        or "youtube.com" in source_lower
        or "youtu.be" in source_lower
    )


def resolve_project_dir(raw_target: Optional[str]) -> str:
    """Résout le répertoire d'un projet sous le dossier racine 'projects/'.

    Règles :
    1. Si raw_target est None ou vide : repli sur 'projects/default'.
    2. Si raw_target est absolu (ex: tests automatisés) : respecté tel quel.
    3. Si raw_target commence déjà par 'projects/' : conservé sans duplication.
    4. Si raw_target est un nom relatif (ex: 'num_4') : résolu sous 'projects/<nom>'.
    """
    base_name = os.environ.get("PROJECTS_DIR", config.PROJECTS_DIR)
    base_dir = Path(base_name)
    if not raw_target:
        return str(base_dir / "default")

    p = Path(raw_target)
    if p.is_absolute():
        return str(p)

    parts = p.parts
    if parts and parts[0] in ("projects", base_name, Path(base_name).name):
        return str(p)

    return str(base_dir / p)


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Point d'entrée CLI pour l'exécution du pipeline de repurposing.

    Args:
        argv: Arguments optionnels de ligne de commande (utilise sys.argv[1:] par défaut).

    Returns:
        Code de retour système (0 en succès, 1 en échec).
    """
    effective_argv = list(argv if argv is not None else sys.argv[1:])

    # Tolérance ergonomique : si l'utilisateur a littéralement tapé le mot 'source' avant la valeur
    if len(effective_argv) >= 2 and effective_argv[0].lower() == "source" and not Path(effective_argv[0]).is_file():
        effective_argv = effective_argv[1:]

    parser = argparse.ArgumentParser(
        description="Générateur de clips courts verticaux (9:16) depuis une vidéo longue ou une URL YouTube.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Darwinx_Repurpose v{__version__}",
        help="Affiche la version du programme et quitte.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        type=str,
        help="URL YouTube de la vidéo ou chemin d'un fichier vidéo local (ex: 'https://youtu.be/...').",
    )
    parser.add_argument(
        "-s",
        "--source",
        dest="opt_source",
        type=str,
        default=None,
        help="Alternative pour spécifier la source (ex: -s 'https://youtu.be/...').",
    )
    parser.add_argument(
        "-p",
        "--project",
        "-o",
        "--out-dir",
        dest="project",
        type=str,
        default=None,
        help="Nom du projet sous 'projects/' ou répertoire de destination (défaut : default).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Résolution maximale pour le téléchargement YouTube (défaut : 1080).",
    )
    parser.add_argument(
        "--clean-source",
        action="store_true",
        help="Supprime la vidéo source téléchargée une fois le job terminé avec succès.",
    )
    parser.add_argument(
        "--purge-jobs",
        nargs="?",
        const=7,
        type=int,
        default=None,
        help="Purger les vieux fichiers de jobs et verrous datant de plus de N jours (défaut : 7 jours).",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Force le recalcul de la transcription Whisper sans utiliser le cache disque.",
    )
    parser.add_argument(
        "--non-verbose",
        "-q",
        "--quiet",
        action="store_true",
        help="Désactive l'affichage verbeux du terminal (mode silencieux/minimal).",
    )

    args = parser.parse_args(effective_argv)

    # Configuration de la verbosité de l'interface
    is_verbose = not args.non_verbose
    ui.set_verbose(is_verbose)
    if not is_verbose:
        logging.getLogger().setLevel(logging.WARNING)

    if args.purge_jobs is not None:
        deleted = job_store.purge_old_jobs(args.purge_jobs)
        if is_verbose:
            ui.print_success(f"Purge terminée : {deleted} fichier(s) de vieux jobs supprimé(s).")
        else:
            print(f"[Maintenance] Purge terminée : {deleted} fichier(s) de vieux jobs supprimé(s).")
        if not (args.source or args.opt_source):
            return 0

    raw_source = (args.source or args.opt_source or "").strip()
    if not raw_source:
        parser.print_help(sys.stderr)
        return 1

    project_dir = resolve_project_dir(args.project)
    max_height = args.max_height

    # Résolution dynamique du matériel réel pour affichage dans le bandeau
    from app import subtitles, transcriber
    try:
        detected_dev, compute_type = transcriber._detect_device_and_compute_type()
        whisper_label = f"{detected_dev.upper()} ({compute_type})"
    except Exception:
        whisper_label = config.WHISPER_DEVICE.upper()

    encoder_label = "NVENC GPU" if subtitles._check_nvenc_support() else "libx264 CPU"

    # Affichage du bandeau de présentation
    ui.print_banner(
        source=raw_source,
        out_dir=project_dir,
        whisper_device=whisper_label,
        llm_provider=config.LLM_PROVIDER,
        encoder=encoder_label,
    )

    t_start = time.time()
    source_path: str

    try:
        # 1. Étape d'ingestion : Aiguillage URL vs Fichier local
        if Path(raw_source).is_file():
            source_path = str(Path(raw_source).resolve())
            print(f"\n[Ingestion] Fichier local détecté : {source_path}")
        elif _is_url(raw_source):
            print(f"\n[Ingestion] URL YouTube détectée : {raw_source}")
            # Sonde de qualité (fail-open)
            probe = downloader.probe_max_quality(raw_source)
            if probe.get("max_height", 0) > 0:
                print(
                    f"  -> Résolution source dispo : {probe['max_height']}p | Durée : {probe.get('duration', 0)}s"
                )

            ui.print_info("Téléchargement de la vidéo source via yt-dlp...")
            source_path = downloader.download_youtube(
                url=raw_source,
                out_dir=os.path.join(project_dir, "source"),
                max_height=max_height,
            )
            print(f"  -> Vidéo téléchargée : {source_path}")
        else:
            print(
                f"\n[ERREUR] La source spécifiée n'est ni un fichier local existant, ni une URL valide : '{raw_source}'",
                file=sys.stderr,
            )
            return 1

        # 2. Création et initialisation du job
        job_id = job_store.create_job(source_path)
        ui.print_info(f"Initialisation du job [cyan]{job_id}[/cyan]...")

        # 3. Exécution orchestrée du pipeline
        result = pipeline.run_pipeline(
            job_id=job_id,
            source_path=source_path,
            out_dir=project_dir,
            force_transcribe=args.force_transcribe,
        )

        elapsed = time.time() - t_start

        # 4. Affichage du bilan terminal soigné
        job_data = job_store.get_job(job_id) or {}
        warnings = job_data.get("warnings", [])
        clips = result.get("result", {}).get("clips", [])

        ui.print_final_report(
            job_id=job_id,
            duration_sec=elapsed,
            clips=clips,
            warnings=warnings,
            out_dir=project_dir,
        )

        # 5. Purge de la vidéo source si demandée et téléchargée
        if args.clean_source and _is_url(raw_source):
            try:
                src_p = Path(source_path)
                if src_p.is_file():
                    src_p.unlink()
                    ui.print_success(f"Vidéo source téléchargée supprimée : {source_path}")
            except OSError as del_err:
                logger.warning("Impossible de supprimer la vidéo source %s : %s", source_path, del_err)

        return 0

    except (PipelineFatalError, DownloaderError, JobStoreError) as exc:
        elapsed = time.time() - t_start
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ÉCHEC BLOQUANT DU PIPELINE]", file=sys.stderr)
        print(f"  Raison : {exc}", file=sys.stderr)
        print(f"  Temps écoulé : {elapsed:.2f}s", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        return 1
    except PipelineUnexpectedError as exc:
        elapsed = time.time() - t_start
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERREUR IMPRÉVUE DU PIPELINE]", file=sys.stderr)
        print(f"  Raison : {exc}", file=sys.stderr)
        print(f"  Temps écoulé : {elapsed:.2f}s", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        return 1
    except Exception as exc:
        elapsed = time.time() - t_start
        print("\n" + "!" * 60, file=sys.stderr)
        print(f"  [ERREUR NON GÉRÉE] : {exc}", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    _proj = Path(__file__).resolve().parent
    _venv_py = _proj / ".venv" / "bin" / "python"
    if _venv_py.is_file() and sys.prefix != str(_proj / ".venv"):
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    sys.exit(cli_main())
