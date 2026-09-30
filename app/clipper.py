"""Module de découpe de clips et cadrage vertical (9:16) avec suivi de visage.

Ce module extrait les segments vidéo correspondant aux moments forts identifiés
par le scoring, les recadre en 9:16 avec un suivi de visage dynamique
frame-par-frame (Haar cascade + lissage exponentiel), et fournit un repli
gracieux sur crop centré si aucun visage n'est détecté ou si OpenCV échoue.

Comportements de repli (AGENTS.md — dégradation gracieuse) :
- Aucun visage détecté → crop centré 9:16, warning loggué, clip livré.
- Échec technique OpenCV (absent, erreur lecture) → crop centré ffmpeg, clip livré.
- Échec total ffmpeg sur un clip → path=None + warning, les autres clips continuent.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config

logger = logging.getLogger(__name__)


class ClipperError(Exception):
    """Exception de base pour les erreurs du module clipper."""

class FFmpegProcessError(ClipperError):
    """ffmpeg a retourné un code de sortie non-nul (commande refusée)."""

class FFmpegTimeoutError(ClipperError):
    """ffmpeg n'a pas terminé dans le délai imparti (bloqué sur fichier corrompu)."""


def _run_ffmpeg(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """Exécute une commande ffmpeg/ffprobe avec les garanties de sécurité du SKILL.md.

    Règles appliquées :
    - Jamais shell=True : cmd est toujours une liste d'arguments.
    - Timeout explicite sur chaque appel (config.FFMPEG_TIMEOUT_SECONDS par défaut).
    - Distinction entre CalledProcessError (ffmpeg a refusé la commande) et
      TimeoutExpired (ffmpeg est resté bloqué, typiquement sur un fichier corrompu).

    Raises:
        FFmpegProcessError: ffmpeg a retourné un code de sortie non-nul.
        FFmpegTimeoutError: ffmpeg n'a pas terminé dans le délai imparti.
    """
    effective_timeout = timeout if timeout is not None else config.FFMPEG_TIMEOUT_SECONDS
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=effective_timeout,
        )
    except subprocess.CalledProcessError as exc:
        stderr_msg = (exc.stderr or b"").decode("utf-8", errors="replace")[:500]
        logger.error(
            "ffmpeg a refusé (code %d) : %s — stderr: %s",
            exc.returncode, cmd[0], stderr_msg,
        )
        raise FFmpegProcessError(
            f"ffmpeg a échoué (code {exc.returncode}) sur {cmd[:4]}... — {stderr_msg}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        logger.error("ffmpeg bloqué (timeout %ds dépassé) : %s", effective_timeout, cmd[0])
        raise FFmpegTimeoutError(
            f"ffmpeg est resté bloqué plus de {effective_timeout}s sur {cmd[:4]}..."
        ) from exc


def _dynamic_face_crop(src: Path, dst: Path) -> Dict[str, Any]:
    """Recadre dynamiquement une vidéo en 9:16 avec suivi de visage frame-par-frame.

    Ouvre src via cv2.VideoCapture, détecte les visages via Haar cascade sur chaque
    frame, lisse le centre X avec un filtre exponentiel (α = config.FACE_SMOOTHING_FACTOR),
    et écrit les frames recadrées dans dst via cv2.VideoWriter (sans audio).

    Les handles VideoCapture et VideoWriter sont libérés inconditionnellement dans un
    bloc finally, même si une frame corrompue ou une erreur mémoire interrompt la boucle.

    Returns:
        {"face_detected": bool} — True si au moins un visage détecté sur au moins
        une frame, False sinon (crop centré appliqué par inertie du lissage).
    """
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("Le package OpenCV installé ne supporte pas CascadeClassifier (nécessite opencv-python<5)")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Impossible de charger le modèle Haar cascade depuis {cascade_path}")

    cap = cv2.VideoCapture(str(src))
    writer = None
    face_seen = False

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la vidéo : {src}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        crop_w = min(w, int(h * 9 / 16))
        crop_h = h
        if crop_w >= w:
            crop_w = w
            crop_h = int(w * 16 / 9)

        smoothed_x = (w - crop_w) // 2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst), fourcc, fps, (crop_w, crop_h))
        if not writer.isOpened():
            raise RuntimeError(f"Impossible de créer le VideoWriter : {dst}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Échantillonnage 1 frame sur 4 : réduit la charge CPU de 75% tout en maintenant
            # la fluidité du cadrage grâce à l'inertie du lissage exponentiel (smoothed_x)
            if frame_idx % 4 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    face_seen = True
                    biggest = max(faces, key=lambda f: f[2] * f[3])
                    face_center_x = biggest[0] + biggest[2] // 2
                    target_x = max(0, min(face_center_x - crop_w // 2, w - crop_w))
                    smoothed_x = int(
                        config.FACE_SMOOTHING_FACTOR * target_x
                        + (1 - config.FACE_SMOOTHING_FACTOR) * smoothed_x
                    )

            frame_idx += 1
            smoothed_x = max(0, min(smoothed_x, w - crop_w))
            cropped = frame[0:crop_h, smoothed_x:smoothed_x + crop_w]
            writer.write(cropped)

        return {"face_detected": face_seen}

    finally:
        if writer is not None:
            writer.release()
        cap.release()


def _ffmpeg_crop_center(src: Path, dst: Path) -> None:
    """Repli : crop centré statique 9:16 via filtre ffmpeg seul.

    Utilisé quand OpenCV est indisponible ou échoue techniquement.
    Récupère les dimensions via ffprobe puis applique un filtre crop centré.
    """
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(src),
    ]
    result = _run_ffmpeg(probe_cmd)
    probe_data = json.loads(result.stdout)
    stream = probe_data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])

    crop_w = min(w, int(h * 9 / 16))
    crop_h = h
    if crop_w >= w:
        crop_w = w
        crop_h = int(w * 16 / 9)

    x_offset = (w - crop_w) // 2
    y_offset = (h - crop_h) // 2

    crop_filter = f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset}"
    crop_cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", crop_filter,
        "-c:a", "copy",
        str(dst),
    ]
    _run_ffmpeg(crop_cmd)


def make_clip(
    source_path: str, start: float, end: float, out_path: str,
) -> Dict[str, Any]:
    """Découpe et recadre un clip en 9:16 avec suivi de visage dynamique.

    Pipeline : trim ffmpeg → cadrage dynamique OpenCV → remux audio ffmpeg.
    Fichiers temporaires nettoyés inconditionnellement dans le bloc finally.

    Returns:
        {"path": Optional[str], "warning": Optional[str]}
    """
    temp_files: List[Path] = []
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        temp_trimmed = out_p.parent / f"_tmp_trim_{uuid.uuid4().hex[:8]}.mp4"
        temp_files.append(temp_trimmed)
        _run_ffmpeg([
            "ffmpeg", "-y", "-ss", str(start), "-i", source_path,
            "-t", str(end - start), "-c", "copy", str(temp_trimmed),
        ])

        warning_msg: Optional[str] = None
        try:
            temp_silent_crop = out_p.parent / f"_tmp_crop_{uuid.uuid4().hex[:8]}.mp4"
            temp_files.append(temp_silent_crop)

            crop_result = _dynamic_face_crop(temp_trimmed, temp_silent_crop)
            if not crop_result["face_detected"]:
                warning_msg = "Aucun visage détecté : repli sur crop centré 9:16"

            _run_ffmpeg([
                "ffmpeg", "-y",
                "-i", str(temp_silent_crop), "-i", str(temp_trimmed),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", str(out_p),
            ])

        except Exception as crop_err:
            logger.warning(
                "Échec technique du cadrage (%s) -> repli crop centré ffmpeg", crop_err,
            )
            warning_msg = (
                f"Échec technique du cadrage ({crop_err}) : repli sur crop centré 9:16"
            )
            _ffmpeg_crop_center(temp_trimmed, out_p)

        return {"path": str(out_p), "warning": warning_msg}

    except Exception as exc:
        logger.error("Échec critique du clip [%.1f - %.1f]: %s", start, end, exc)
        return {"path": None, "warning": f"Échec critique du clip: {exc}"}

    finally:
        for tmp in temp_files:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError as err:
                logger.warning("Impossible de supprimer %s : %s", tmp, err)


def make_clips(
    source_path: str,
    highlights: List[Dict[str, Any]],
    out_dir: str,
) -> List[Dict[str, Any]]:
    """Découpe un lot de clips depuis la vidéo source avec dégradation gracieuse.

    Un clip en échec total (path=None) n'interrompt pas le traitement des autres.
    Les noms de fichiers sont générés par le code (clip_01.mp4, clip_02.mp4, ...)
    pour éviter toute injection de chemin.

    Returns:
        Liste de dicts highlight enrichis de "path" et "warning".
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for i, hl in enumerate(highlights):
        clip_name = f"clip_{i + 1:02d}.mp4"
        clip_path = out_dir_p / clip_name

        result = make_clip(
            source_path=source_path,
            start=float(hl["start_time"]),
            end=float(hl["end_time"]),
            out_path=str(clip_path),
        )

        enriched = {**hl, "path": result["path"], "warning": result["warning"]}
        results.append(enriched)

    return results
