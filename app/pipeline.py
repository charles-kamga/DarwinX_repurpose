"""Module orchestrateur du pipeline de repurposing vidéo (long -> shorts).

Enchaîne : transcriber -> highlights -> clipper -> subtitles -> metadata_studio.
Respecte les principes d'AGENTS.md : dégradation gracieuse, suivi d'état traçable,
et gestion fine des erreurs bloquantes vs avertissements tolérés.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional

from app import (
    clipper,
    config,
    highlights,
    job_store,
    metadata_studio,
    subtitles,
    telegram_notifier,
    transcriber,
)
from app.highlights import NoHighlightsFoundError
from app.transcriber import EmptyTranscriptionError, TranscriberMediaError
from app.ui import ui

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Exception de base pour les erreurs de l'orchestrateur de pipeline."""


class PipelineFatalError(PipelineError):
    """Levée lors d'un échec bloquant identifié (transcription, scoring, ou tous clips)."""


class PipelineUnexpectedError(PipelineError):
    """Levée lors d'une exception imprévue capturée par le filet de dernier recours."""


def _extract_clip_script_text(transcript: Dict[str, Any], start: float, end: float) -> str:
    """Extrait le texte des segments compris dans l'intervalle [start, end]."""
    words = [
        seg.get("text", "").strip()
        for seg in transcript.get("segments", [])
        if float(seg.get("end", 0.0)) > start and float(seg.get("start", 0.0)) < end and seg.get("text")
    ]
    return " ".join(words).strip()


def run_pipeline(
    job_id: str,
    source_path: str,
    out_dir: str,
    force_transcribe: bool = False,
) -> Dict[str, Any]:
    """Orchestre le pipeline complet de repurposing vidéo pour un job donné.

    Args:
        job_id: Identifiant unique du job (préalablement créé via job_store.create_job).
        source_path: Chemin du fichier média source sur le disque.
        out_dir: Répertoire racine de destination pour les artefacts générés.
        force_transcribe: Force le recalcul Whisper sans utiliser le cache disque.

    Returns:
        En cas de succès : {"job_id": str, "status": "completed", "result": Dict[str, Any]}

    Raises:
        PipelineFatalError: Si l'une des 3 étapes critiques bloquantes échoue.
        PipelineUnexpectedError: Si une erreur non anticipée est capturée en dernier recours.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()
    telegram_notifier.notify_job_start(
        job_id=job_id,
        source=source_path,
        whisper_device=config.WHISPER_DEVICE,
        llm_provider=config.LLM_PROVIDER,
    )

    try:
        # ---------------------------------------------------------------------
        # Étape 1 : Transcription audio (faster-whisper)
        # ---------------------------------------------------------------------
        job_store.set_step(job_id, "transcription")
        ui.print_step_header(1, 4, "Transcription Audio", f"source : {Path(source_path).name}")
        ui.print_info("Extraction audio et horodatage mot-par-mot via Whisper...")
        try:
            transcript = (
                transcriber.transcribe(source_path, force=True)
                if force_transcribe
                else transcriber.transcribe(source_path)
            )
            seg_count = len(transcript.get("segments", []))
            dur_s = float(transcript.get("duration", 0.0))
            from_cache = transcriber.was_last_from_cache()
            dev_used = transcriber.get_last_device_used()
            if from_cache:
                ui.print_success(f"⚡ Chargé depuis le cache disque : {seg_count} segments ({dur_s:.1f}s)")
                ui.print_info("Note : utilisez --force-transcribe pour forcer le recalcul GPU sans cache.")
            else:
                ui.print_success(f"⚡ Inférence Whisper exécutée sur {dev_used} : {seg_count} segments ({dur_s:.1f}s)")
            telegram_notifier.notify_step_done(
                job_id=job_id,
                step_name="Transcription Audio",
                metrics={"Segments": seg_count, "Durée": f"{dur_s:.1f}s", "Mode": "Cache" if from_cache else dev_used},
            )
        except (EmptyTranscriptionError, TranscriberMediaError, FileNotFoundError) as exc:
            err_msg = f"Échec bloquant de transcription : {exc}"
            logger.error("Job %s - %s", job_id, err_msg)
            job_store.mark_failed(job_id, err_msg)
            ui.print_error(err_msg)
            telegram_notifier.notify_job_error(job_id, "Transcription Audio", err_msg)
            raise PipelineFatalError(err_msg) from exc

        job_store.mark_step_done(job_id, "transcription")

        # ---------------------------------------------------------------------
        # Étape 2 : Scoring des moments viraux via LLM
        # ---------------------------------------------------------------------
        job_store.set_step(job_id, "scoring")
        ui.print_step_header(2, 4, "Scoring des Moments Viraux", f"fournisseur : {config.LLM_PROVIDER.upper()}")
        ui.print_info("Analyse de viralité, hooks et dédoublonnage temporel...")
        try:
            highlights_list = highlights.get_highlights(transcript)
            ui.print_success(f"{len(highlights_list)} moments viraux retenus")
            ui.print_highlights_table(highlights_list)
            telegram_notifier.notify_step_done(
                job_id=job_id,
                step_name="Scoring Moments Viraux",
                metrics={"Moments retenus": len(highlights_list), "Fournisseur": config.LLM_PROVIDER.upper()},
            )
        except NoHighlightsFoundError as exc:
            err_msg = f"Échec bloquant de scoring : {exc}"
            logger.error("Job %s - %s", job_id, err_msg)
            job_store.mark_failed(job_id, err_msg)
            ui.print_error(err_msg)
            telegram_notifier.notify_job_error(job_id, "Scoring Moments Viraux", err_msg)
            raise PipelineFatalError(err_msg) from exc

        job_store.mark_step_done(job_id, "scoring")

        # ---------------------------------------------------------------------
        # Étape 3 : Découpe et cadrage vertical 9:16 (clipper)
        # ---------------------------------------------------------------------
        job_store.set_step(job_id, "clipper")
        ui.print_step_header(3, 4, "Découpe & Cadrage Vertical 9:16", f"{len(highlights_list)} clips à produire")
        raw_clips_dir = out_path / "raw_clips"
        raw_clips = clipper.make_clips(source_path, highlights_list, str(raw_clips_dir))

        # Échec bloquant SSI tous les clips ont échoué au découpage
        if not raw_clips or all(c.get("path") is None for c in raw_clips):
            err_msg = "Échec bloquant : tous les clips ont échoué au montage vidéo"
            logger.error("Job %s - %s", job_id, err_msg)
            job_store.mark_failed(job_id, err_msg)
            ui.print_error(err_msg)
            telegram_notifier.notify_job_error(job_id, "Découpe & Cadrage 9:16", err_msg)
            raise PipelineFatalError(err_msg)

        # Dégradation gracieuse : isolation des clips valides et enregistrement des warnings
        valid_clips: List[Dict[str, Any]] = []
        for idx, clip in enumerate(raw_clips, 1):
            clip_name = Path(clip.get("path", "") or f"clip_{idx:02d}").stem
            c_dur = float(clip.get("end_time", 0)) - float(clip.get("start_time", 0))
            if clip.get("path") is None:
                warn = f"Clip {idx} échoué au découpage : {clip.get('warning')}"
                job_store.add_warning(job_id, warn)
                ui.print_clip_action(idx, len(raw_clips), "Cadrage 9:16", clip_name, warning=clip.get("warning"))
            else:
                if clip.get("warning"):
                    job_store.add_warning(job_id, f"Clip {idx} : {clip.get('warning')}")
                valid_clips.append(clip)
                ui.print_clip_action(idx, len(raw_clips), "Cadrage 9:16", clip_name, detail=f"{c_dur:.1f}s", warning=clip.get("warning"))

        job_store.mark_step_done(job_id, "clipper")

        # ---------------------------------------------------------------------
        # Étape 4 : Génération et incrustation des sous-titres animés ASS
        # ---------------------------------------------------------------------
        job_store.set_step(job_id, "subtitles")
        ui.print_step_header(4, 4, "Génération & Incrustation Sous-Titres", f"{len(valid_clips)} clips valides")
        subtitles_dir = out_path / "subtitles"
        final_clips_dir = out_path / "final_clips"
        subtitles_dir.mkdir(parents=True, exist_ok=True)
        final_clips_dir.mkdir(parents=True, exist_ok=True)

        final_clips: List[Dict[str, Any]] = []
        for idx, clip in enumerate(valid_clips, 1):
            clip_name = Path(clip["path"]).stem
            ass_path = subtitles_dir / f"{clip_name}.ass"
            final_clip_path = final_clips_dir / f"{clip_name}.mp4"
            sub_warning: Optional[str] = None
            try:
                subtitles.generate_ass(transcript, float(clip["start_time"]), float(clip["end_time"]), str(ass_path))
                sub_result = subtitles.burn_subtitles(clip["path"], str(ass_path), str(final_clip_path))
                if sub_result.get("warning"):
                    sub_warning = f"Sous-titres clip {idx} : {sub_result.get('warning')}"
            except Exception as sub_err:
                logger.warning("Sous-titres clip %d ignorés (%s) -> repli", idx, sub_err)
                sub_warning = f"Sous-titres clip {idx} ignorés ({sub_err})"
                try:
                    shutil.copyfile(clip["path"], str(final_clip_path))
                    sub_result = {"path": str(final_clip_path), "warning": sub_warning}
                except OSError:
                    sub_result = {"path": clip["path"], "warning": sub_warning}

            if sub_warning:
                job_store.add_warning(job_id, sub_warning)
            ui.print_clip_action(idx, len(valid_clips), "Sous-titres ASS", clip_name, warning=sub_warning)

            # 3. Kit de diffusion (miniature 9:16, métadonnées SEO, fiche .txt)
            pub_info: Dict[str, Any] = {}
            try:
                clip_script = _extract_clip_script_text(transcript, float(clip["start_time"]), float(clip["end_time"]))
                pub_info = metadata_studio.create_clip_publishing_kit(clip, Path(sub_result["path"]), final_clips_dir, script_text=clip_script)
                ui.print_clip_action(idx, len(valid_clips), "Pack Diffusion", clip_name, detail="Miniature + SEO")
            except Exception as pub_err:
                logger.warning("Kit publication clip %d partiel : %s", idx, pub_err)
                job_store.add_warning(job_id, f"Kit publication clip {idx} : {pub_err}")

            final_clips.append({
                **clip, "final_path": sub_result["path"],
                "thumbnail_path": pub_info.get("thumbnail_path"),
                "publish_txt_path": pub_info.get("publish_txt_path"),
                "metadata_json_path": pub_info.get("metadata_json_path"),
                "recommended_title": pub_info.get("recommended_title"),
                "description": pub_info.get("description"),
                "hashtags": pub_info.get("hashtags", []),
            })

        job_store.mark_step_done(job_id, "subtitles")
        telegram_notifier.notify_step_done(
            job_id=job_id,
            step_name="Cadrage & Sous-Titres",
            metrics={"Clips produits": len(final_clips)},
        )

        # ---------------------------------------------------------------------
        # Étape 5 : Purge des fichiers intermédiaires & Finalisation du résultat
        # ---------------------------------------------------------------------
        if config.CLEAN_INTERMEDIATE_CLIPS:
            purged = 0
            for clip in valid_clips:
                raw_clip_p = Path(clip["path"])
                try:
                    if raw_clip_p.is_file() and str(raw_clip_p) != clip.get("final_path"):
                        raw_clip_p.unlink()
                        purged += 1
                except OSError as rm_err:
                    logger.warning("Purge impossible %s : %s", raw_clip_p, rm_err)
            if purged > 0:
                ui.print_success(f"Purge automatique : {purged} clip(s) brut(s) intermédiaire(s) supprimé(s)")

        result_payload = {"clips": final_clips, "total_clips": len(final_clips)}
        job_store.mark_completed(job_id, result_payload)

        # Manifeste de projet autonome
        try:
            manifest_data = {
                "job_id": job_id, "source": source_path,
                "duration_sec": round(time.time() - t_pipeline_start, 2),
                "total_clips": len(final_clips), "clips": final_clips,
                "warnings": (job_store.get_job(job_id) or {}).get("warnings", []),
            }
            (out_path / "project_manifest.json").write_text(
                json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as man_err:
            logger.warning("Manifeste non écrit pour %s : %s", out_path, man_err)

        job_data = job_store.get_job(job_id) or {}
        telegram_notifier.notify_job_completed(
            job_id=job_id, duration_sec=time.time() - t_pipeline_start,
            clips=final_clips, warnings=job_data.get("warnings", []),
        )

        return {"job_id": job_id, "status": "completed", "result": result_payload}

    except PipelineFatalError:
        raise
    except Exception as unexpected_exc:
        # Filet de sécurité absolu : empêche un job d'être abandonné en statut 'processing'
        err_msg = f"Erreur critique imprévue dans le pipeline : {type(unexpected_exc).__name__} - {unexpected_exc}"
        logger.exception("Échec critique imprévu sur le job %s : %s", job_id, unexpected_exc)
        try:
            job_store.mark_failed(job_id, err_msg)
        except Exception as store_err:
            logger.critical("Double défaillance sur job %s : %s", job_id, store_err, exc_info=True)
        telegram_notifier.notify_job_error(job_id, "Erreur Critique Imprévue", err_msg)
        raise PipelineUnexpectedError(err_msg) from unexpected_exc
