"""Module de scoring et d'extraction des moments viraux du transcript via LLM.

Ce module découpe le transcript en fenêtres temporelles, sollicite un LLM avec
un prompt spécialisé pour identifier les clips à fort potentiel de rétention,
déduplique les segments par recouvrement et applique les bornes plancher/plafond.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

from app import config, llm_cascade
from app.llm_cascade import LLMError, PermanentLLMError, TransientLLMError

logger = logging.getLogger(__name__)


class HighlightsError(Exception):
    """Exception de base pour les erreurs du module highlights."""

class NoHighlightsFoundError(HighlightsError):
    """Levée lorsqu'aucun moment fort n'a pu être extrait (échec bloquant de job)."""



def _is_transient_error(exc: Exception) -> bool:
    """Détermine si une exception d'appel LLM est transitoire ou permanente.
    
    Conformément au Principe n°2 d'AGENTS.md :
    - Transitoire : rate limit (429), timeout, erreur réseau, erreur serveur 5xx.
    - Permanent : 401/403 (auth), 400 (filtre de modération/sécurité, prompt invalide).
    - Règle par défaut : Toute erreur non reconnue est traitée par défaut comme
      permanente (fail-safe contre le gaspillage d'appels payants sans garantie de succès),
      pas comme transitoire.
    """
    if isinstance(exc, TransientLLMError):
        return True
    if isinstance(exc, PermanentLLMError):
        return False

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if status_code in (408, 429) or status_code >= 500:
            return True
        if status_code in (400, 401, 403, 404, 422):
            return False

    exc_name = exc.__class__.__name__
    if any(k in exc_name for k in ("Timeout", "RateLimit", "Connection", "InternalServerError")):
        return True
    if any(k in exc_name for k in ("Auth", "Permission", "BadRequest")):
        return False

    msg = str(exc).lower()
    if any(k in msg for k in ("rate limit", "too many requests", "timeout", "timed out", "connection reset", "503")):
        return True

    return False


def _call_llm_with_retry(prompt: str, llm_fn: Callable[[str], str]) -> Optional[str]:
    """Exécute un appel LLM avec retry exponentiel sur erreurs transitoires uniquement."""
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            return llm_fn(prompt)
        except Exception as exc:
            if not _is_transient_error(exc):
                logger.warning("Erreur LLM permanente (%s: %s) -> abandon immédiat de la fenêtre", type(exc).__name__, exc)
                return None

            if attempt < config.LLM_MAX_RETRIES - 1:
                backoff = config.LLM_BACKOFF_BASE_SECONDS * (config.LLM_BACKOFF_FACTOR ** attempt)
                logger.warning(
                    "Erreur transitoire LLM (%s) sur tentative %d/%d -> retry dans %.1fs",
                    exc, attempt + 1, config.LLM_MAX_RETRIES, backoff
                )
                time.sleep(backoff)
            else:
                logger.warning("Échec définitif de la fenêtre après %d tentatives transitoires: %s", config.LLM_MAX_RETRIES, exc)
                return None
    return None


def _create_windows(transcript: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """Découpe le transcript en fenêtres temporelles alignées sur les segments."""
    segments = transcript.get("segments", [])
    duration = float(transcript.get("duration", 0.0))
    if not segments:
        return []

    if duration <= config.LONG_VIDEO_THRESHOLD_SECONDS:
        return [segments]

    windows: List[List[Dict[str, Any]]] = []
    n = len(segments)
    i = 0
    while i < n:
        win_start = segments[i]["start"]
        window_segments: List[Dict[str, Any]] = []
        j = i
        while j < n and (segments[j]["end"] - win_start) <= config.WINDOW_SECONDS:
            window_segments.append(segments[j])
            j += 1

        if not window_segments and j < n:
            window_segments.append(segments[j])
            j += 1

        windows.append(window_segments)
        if j >= n:
            break

        target_next_start = win_start + max(1.0, float(config.WINDOW_SECONDS - config.WINDOW_OVERLAP_SECONDS))
        next_i = i + 1
        while next_i < j and segments[next_i]["start"] < target_next_start:
            next_i += 1
        i = next_i if next_i > i else i + 1

    return windows


def _build_scoring_prompt(segments: List[Dict[str, Any]]) -> str:
    """Construit le prompt de scoring des moments viraux pour le LLM."""
    formatted_transcript = "\n".join(
        f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}" for seg in segments
    )
    return (
        "Tu es un expert mondial du montage de vidéos courtes virales (TikTok, Reels, Shorts).\n"
        "Analyse la transcription horodatée ci-dessous et identifie les meilleurs moments forts "
        f"(durée entre {config.CLIP_MIN_SECONDS}s et {config.CLIP_MAX_SECONDS}s).\n\n"
        "Critères de viralité :\n"
        "1. Accroche percutante dès la première seconde (Hook sentence).\n"
        "2. Histoire ou concept complet et auto-suffisant.\n"
        "3. Climax émotionnel, révélation surprenante ou forte valeur d'apprentissage.\n\n"
        "Format de réponse JSON strict attendu :\n"
        "[\n"
        "  {\n"
        '    "start_time": float,\n'
        '    "end_time": float,\n'
        '    "score": int (0 à 100),\n'
        '    "hook_sentence": str,\n'
        '    "reason": str\n'
        "  }\n"
        "]\n\n"
        f"Transcription :\n{formatted_transcript}"
    )


def _parse_llm_response(raw_response: str) -> List[Dict[str, Any]]:
    """Extrait et valide strictement le JSON retourné par le LLM."""
    if not raw_response:
        return []

    match = re.search(r"(\[.*\]|\{.*\})", raw_response, re.DOTALL)
    if not match:
        logger.warning("Aucun bloc JSON détecté dans la réponse LLM")
        return []

    try:
        data = json.loads(match.group(1))
    except Exception as exc:
        logger.warning("Échec du parsing JSON LLM : %s", exc)
        return []

    if isinstance(data, dict):
        data = data.get("highlights", [])
    if not isinstance(data, list):
        return []

    valid_highlights: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start = round(float(item["start_time"]), 2)
            end = round(float(item["end_time"]), 2)
            score = int(round(float(item["score"])))
            hook = str(item.get("hook_sentence", "")).strip()
            reason = str(item.get("reason", "")).strip()
        except (KeyError, ValueError, TypeError):
            continue

        dur = end - start
        if end <= start or not (config.CLIP_MIN_SECONDS <= dur <= config.CLIP_MAX_SECONDS):
            continue
        if not (0 <= score <= 100) or not hook:
            continue

        valid_highlights.append({
            "start_time": start,
            "end_time": end,
            "score": score,
            "hook_sentence": hook,
            "reason": reason,
        })

    return valid_highlights


def _deduplicate_highlights(highlights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dédoublonne les highlights par recouvrement temporel (> 50%)."""
    sorted_highlights = sorted(highlights, key=lambda h: h["score"], reverse=True)
    retained: List[Dict[str, Any]] = []

    for cand in sorted_highlights:
        conflict = False
        cand_dur = cand["end_time"] - cand["start_time"]
        for r in retained:
            r_dur = r["end_time"] - r["start_time"]
            overlap = max(0.0, min(cand["end_time"], r["end_time"]) - max(cand["start_time"], r["start_time"]))
            min_dur = min(cand_dur, r_dur)
            if min_dur > 0 and (overlap / min_dur) > 0.50:
                conflict = True
                break
        if not conflict:
            retained.append(cand)

    return retained


def _default_anthropic_caller(prompt: str) -> str:
    """Appel réel par défaut à l'API Anthropic via son SDK (délégué à llm_cascade)."""
    return llm_cascade._call_anthropic(prompt)


def _default_groq_caller(prompt: str) -> str:
    """Appel réel par défaut à l'API Groq via son SDK (délégué à llm_cascade)."""
    return llm_cascade._call_groq(prompt)


def get_highlights(
    transcript: Dict[str, Any],
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[Dict[str, Any]]:
    """Extrait et score les moments viraux d'un transcript via LLM.

    Args:
        transcript: Dictionnaire contenant la durée et les segments horodatés.
        llm_fn: Fonction exécutant l'inférence LLM (injectable pour tests).

    Returns:
        Liste de moments viraux dédupliqués et triés par score.

    Raises:
        NoHighlightsFoundError: Si aucun moment fort n'a pu être extrait.
    """
    if llm_fn is not None:
        caller = llm_fn
    elif config.LLM_PROVIDER.lower().strip() == "groq":
        caller = _default_groq_caller
    elif config.LLM_PROVIDER.lower().strip() == "anthropic":
        caller = _default_anthropic_caller
    else:
        caller = llm_cascade.resolve_llm_caller(config.LLM_PROVIDER)

    windows = _create_windows(transcript)

    if not windows:
        raise NoHighlightsFoundError("Transcript vide ou sans segments.")

    all_highlights: List[Dict[str, Any]] = []
    for idx, window in enumerate(windows, 1):
        prompt = _build_scoring_prompt(window)
        raw_response = _call_llm_with_retry(prompt, caller)
        if raw_response:
            parsed = _parse_llm_response(raw_response)
            all_highlights.extend(parsed)

    retained = _deduplicate_highlights(all_highlights)

    if not retained:
        raise NoHighlightsFoundError(
            "Aucun moment fort valide extrait après analyse de toutes les fenêtres. "
            "Le pipeline ne peut pas continuer."
        )

    if len(retained) < config.CLIP_COUNT_MIN:
        logger.warning(
            "Nombre de clips extraits (%d) inférieur au plancher recommandé (CLIP_COUNT_MIN=%d).",
            len(retained),
            config.CLIP_COUNT_MIN,
        )

    return retained[: config.CLIP_COUNT_MAX]
