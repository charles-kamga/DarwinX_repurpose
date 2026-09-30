"""Module de génération et d'incrustation des sous-titres animés (format ASS).

Ce module convertit les transcriptions horodatées mot-par-mot en sous-titres ASS
stylisés pour vidéo verticale (9:16). Il effectue le recalage temporel relatif
au début de chaque clip, regroupe les mots en blocs de lecture lisibles,
et incruste les sous-titres via ffmpeg avec dégradation gracieuse en cas d'échec.

Principes appliqués (AGENTS.md & SKILL.md) :
- Recalage temporel relatif : rel_start = word.start - clip_start.
- Mots avec timestamp manquant (None) : ignorés gracieusement sans crash.
- Échappement ASS : neutralisation des balises style { } et anti-slashs,
  préservation des virgules dans le champ Text.
- Dégradation gracieuse : en cas d'échec d'incrustation, livraison du clip
  sans sous-titres plutôt que rejet du clip.
- Réutilisation : _run_ffmpeg est importé depuis app.clipper.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from app import config
from app.clipper import _run_ffmpeg

logger = logging.getLogger(__name__)

_NVENC_SUPPORTED: Optional[bool] = None


def _check_nvenc_support() -> bool:
    """Vérifie si l'encodeur matériel h264_nvenc est utilisable sur le système."""
    global _NVENC_SUPPORTED
    if _NVENC_SUPPORTED is not None:
        return _NVENC_SUPPORTED
    try:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.05", "-c:v", "h264_nvenc", "-f", "null", "-"]
        res = subprocess.run(cmd, capture_output=True, timeout=4)
        _NVENC_SUPPORTED = (res.returncode == 0)
    except Exception:
        _NVENC_SUPPORTED = False
    return _NVENC_SUPPORTED


def _get_encoder_args() -> List[str]:
    """Sélectionne l'encodeur optimal : GPU h264_nvenc si disponible, sinon libx264."""
    if _check_nvenc_support():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    return ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]


class SubtitlesError(Exception):
    """Exception de base pour les erreurs du module de sous-titres."""


def _format_ass_time(seconds: float) -> str:
    """Convertit une durée en secondes au format horodaté ASS 'H:MM:SS.cs'."""
    clamped_sec = max(0.0, float(seconds))
    centiseconds = int(round(clamped_sec * 100))
    h = centiseconds // 360000
    centiseconds %= 360000
    m = centiseconds // 6000
    centiseconds %= 6000
    s = centiseconds // 100
    cs = centiseconds % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Neutralise les caractères de contrôle ASS tout en préservant les virgules.

    - '{' et '}' sont remplacés par '(' et ')' pour empêcher l'interprétation
      comme balises de style ASS.
    - '\\' est remplacé par '/' pour éviter des séquences d'échappement invalides.
    - Les virgules ',' sont conservées telles quelles dans le texte.
    """
    cleaned = text.replace("\\", "/")
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned.strip()


def _extract_recalibrated_words(
    transcript: Dict[str, Any],
    clip_start: float,
    clip_end: float,
) -> List[Dict[str, Any]]:
    """Extrait et recale les mots du transcript par rapport au début du clip."""
    words: List[Dict[str, Any]] = []
    segments = transcript.get("segments", [])

    for segment in segments:
        seg_words = segment.get("words", [])
        for w in seg_words:
            # Récupération tolérante des clés (format whisper "w"/"s"/"e" ou "word"/"start"/"end")
            word_text = str(w.get("w", w.get("word", "")))
            start_raw = w.get("s", w.get("start"))
            end_raw = w.get("e", w.get("end"))

            # Dégradation gracieuse sur timestamp manquant
            if start_raw is None or end_raw is None:
                logger.warning(
                    "Timestamp None sur le mot '%s' -> mot ignoré du calage fin",
                    word_text,
                )
                continue

            try:
                start_val = float(start_raw)
                end_val = float(end_raw)
            except (ValueError, TypeError):
                continue

            # Exclusion des mots entièrement hors de l'intervalle du clip
            if end_val <= clip_start or start_val >= clip_end:
                continue

            # Recalage relatif au début du clip
            rel_start = max(0.0, start_val - clip_start)
            rel_end = min(clip_end - clip_start, end_val - clip_start)

            if rel_end <= rel_start:
                continue

            escaped_text = _escape_ass_text(word_text)
            if not escaped_text:
                continue

            words.append({
                "text": escaped_text,
                "start": round(rel_start, 3),
                "end": round(rel_end, 3),
            })

    return words


def _group_words_into_blocks(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Regroupe les mots en blocs de sous-titres selon les contraintes de lisibilité.

    Contraintes appliquées (SKILL.md) :
    - Limite de caractères par bloc : config.SUBTITLE_MAX_CHARS (~20)
    - Limite de durée par bloc : config.SUBTITLE_MAX_DURATION_SECONDS (~2.0s)
    - Coupures strictes entre deux mots entiers
    """
    if not words:
        return []

    blocks: List[Dict[str, Any]] = []
    current_block: Optional[Dict[str, Any]] = None

    for w in words:
        if current_block is None:
            current_block = {
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
            }
            continue

        candid_text = f"{current_block['text']} {w['text']}"
        candid_duration = w["end"] - current_block["start"]

        # Nouveau bloc déclenché si dépassement de caractères ou de durée
        if (
            len(candid_text) > config.SUBTITLE_MAX_CHARS
            or candid_duration > config.SUBTITLE_MAX_DURATION_SECONDS
        ):
            blocks.append(current_block)
            current_block = {
                "start": w["start"],
                "end": w["end"],
                "text": w["text"],
            }
        else:
            current_block["text"] = candid_text
            current_block["end"] = w["end"]

    if current_block is not None:
        blocks.append(current_block)

    return blocks


def _build_ass_content(blocks: List[Dict[str, Any]]) -> str:
    """Génère la chaîne complète du fichier .ass stylisée pour format vertical 9:16."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        "Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,"
        "100,100,0,0,1,4,0,2,20,20,240,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines: List[str] = []
    for b in blocks:
        start_str = _format_ass_time(b["start"])
        end_str = _format_ass_time(b["end"])
        line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{b['text']}"
        dialogue_lines.append(line)

    return header + "\n".join(dialogue_lines) + ("\n" if dialogue_lines else "")


def generate_ass(
    transcript: Dict[str, Any],
    clip_start: float,
    clip_end: float,
    out_path: str,
) -> None:
    """Génère un fichier de sous-titres ASS recalé et découpé en blocs lisibles.

    Args:
        transcript: Dictionnaire contenant les segments et mots horodatés.
        clip_start: Horodatage de début du clip dans la vidéo source (secondes).
        clip_end: Horodatage de fin du clip dans la vidéo source (secondes).
        out_path: Chemin du fichier .ass à écrire.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    words = _extract_recalibrated_words(transcript, clip_start, clip_end)
    blocks = _group_words_into_blocks(words)
    ass_content = _build_ass_content(blocks)

    out_p.write_text(ass_content, encoding="utf-8")


def burn_subtitles(video_path: str, ass_path: str, out_path: str) -> Dict[str, Any]:
    """Incruste les sous-titres ASS dans la vidéo avec repli gracieux en cas d'échec.

    Si l'incrustation échoue (police manquante, ffmpeg refuse le filtre), la vidéo
    d'origine est copiée à out_path et un warning est retourné. Le clip est livré.

    Returns:
        {"path": str, "warning": Optional[str]}
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Échappement requis par la syntaxe de filtre ffmpeg (ass='chemin')
    escaped_ass = str(Path(ass_path).resolve()).replace("\\", "/").replace(":", "\\:")
    enc_args = _get_encoder_args()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass='{escaped_ass}'",
        *enc_args,
        "-c:a", "copy",
        str(out_p),
    ]

    try:
        _run_ffmpeg(cmd)
        return {"path": str(out_p), "warning": None}
    except Exception as exc:
        if "h264_nvenc" in enc_args:
            logger.warning("Échec encodage GPU nvenc (%s) -> repli immédiat sur CPU libx264...", exc)
            fallback_cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"ass='{escaped_ass}'",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy", str(out_p),
            ]
            try:
                _run_ffmpeg(fallback_cmd)
                return {"path": str(out_p), "warning": f"Incrustation sous-titres basculée sur CPU ({exc})"}
            except Exception as cpu_err:
                exc = cpu_err

        logger.warning("Échec de l'incrustation des sous-titres (%s) -> repli : clip sans sous-titres", exc)
        try:
            shutil.copyfile(video_path, out_path)
        except OSError as copy_err:
            logger.error("Impossible de copier la vidéo de repli vers %s : %s", out_path, copy_err)

        return {
            "path": str(out_p),
            "warning": f"Échec de l'incrustation des sous-titres ({exc}) : clip livré sans sous-titres",
        }
