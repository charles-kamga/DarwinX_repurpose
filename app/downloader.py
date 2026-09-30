"""Module d'ingestion et de téléchargement de vidéos YouTube via yt-dlp.

Ce module télécharge la vidéo source à traiter par le pipeline :
- Sonde de qualité fail-open (ne lève JAMAIS d'exception, retourne max_height et duration).
- Retry ciblé avec backoff exponentiel uniquement sur erreurs transitoires (403, 429, timeouts).
- Échec immédiat sans retry inutile sur erreurs permanentes (URL invalide, vidéo privée/supprimée).
- Levée explicite de DownloaderError si toutes les tentatives échouent (bloquant pour le job).
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

try:
    import yt_dlp
except ImportError:
    yt_dlp = None  # type: ignore

from app import config

logger = logging.getLogger(__name__)


class DownloaderError(Exception):
    """Exception de base pour les erreurs du module de téléchargement."""


class PermanentDownloadError(DownloaderError):
    """Levée lors d'une erreur permanente (URL invalide, vidéo inaccessible) sans retry."""


class TransientDownloadError(DownloaderError):
    """Levée lors d'une erreur transitoire côté YouTube (403, 429, timeout réseau)."""


def _is_transient_error(exc: Exception) -> bool:
    """Détermine si une exception est transitoire (éligible au retry) ou permanente.

    Erreurs transitoires reconnues :
    - HTTP 403 Forbidden (throttling temporaire YouTube)
    - HTTP 429 Too Many Requests
    - Timeouts réseau / Connection reset / Network unreachable

    Erreurs permanentes reconnues :
    - URL invalide, vidéo inexistante, vidéo privée, compte suspendu

    Règle par défaut (fail-safe) : si l'erreur n'est pas reconnue comme transitoire,
    elle est traitée comme permanente pour éviter des retries inutiles.
    """
    msg = str(exc).lower()

    # Mots-clés d'erreurs permanentes
    permanent_keywords = [
        "is not a valid url",
        "incomplete youtube id",
        "video unavailable",
        "this video is unavailable",
        "private video",
        "this video is private",
        "has been removed",
        "terminated",
        "requires authentication",
        "sign in to confirm",
    ]
    if any(kw in msg for kw in permanent_keywords):
        return False

    # Mots-clés d'erreurs transitoires
    transient_keywords = [
        "http error 403",
        "403: forbidden",
        "forbidden",
        "http error 429",
        "429",
        "too many requests",
        "timed out",
        "timeout",
        "connection reset",
        "network is unreachable",
        "temporary failure",
    ]
    if any(kw in msg for kw in transient_keywords):
        return True

    return False


def probe_max_quality(url: str) -> Dict[str, int]:
    """Sonde la qualité maximale disponible et la durée d'une vidéo YouTube.

    GARANTIE FAIL-OPEN : cette fonction ne lève JAMAIS d'exception. En cas d'échec
    (réseau, URL douteuse, restriction), elle logue un avertissement et retourne
    des valeurs neutres par défaut (max_height=0, duration=0).

    Returns:
        {"max_height": int, "duration": int}
    """
    if yt_dlp is None:
        logger.warning("Le package 'yt-dlp' n'est pas installé -> repli fail-open (max_height=0, duration=0)")
        return {"max_height": 0, "duration": 0}

    ydl_opts: Dict[str, Any] = {
        "extract_flat": False,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"max_height": 0, "duration": 0}

            duration = int(info.get("duration") or 0)
            formats = info.get("formats", [])
            heights = [
                f.get("height")
                for f in formats
                if f.get("height") is not None and isinstance(f.get("height"), (int, float))
            ]
            if info.get("height") and isinstance(info.get("height"), (int, float)):
                heights.append(int(info["height"]))

            max_height = int(max(heights)) if heights else 0
            return {"max_height": max_height, "duration": duration}

    except Exception as exc:
        logger.warning(
            "Échec de la sonde de qualité pour '%s' (%s) -> repli fail-open (max_height=0, duration=0)",
            url, exc,
        )
        return {"max_height": 0, "duration": 0}


def download_youtube(
    url: str,
    out_dir: Optional[str] = None,
    max_height: int = 1080,
) -> str:
    """Télécharge la vidéo source via yt-dlp avec retry ciblé sur erreurs transitoires.

    Args:
        url: URL YouTube de la vidéo à télécharger.
        out_dir: Répertoire de destination optionnel (par défaut config.DOWNLOADS_DIR).
        max_height: Résolution verticale maximale souhaitée (par défaut 1080p).

    Returns:
        Chemin absolu du fichier vidéo téléchargé.

    Raises:
        PermanentDownloadError: Si l'URL est invalide ou la vidéo inaccessible (sans retry).
        DownloaderError: Si le téléchargement échoue après épuisement des retries.
    """
    if yt_dlp is None:
        raise DownloaderError("Le package 'yt-dlp' n'est pas installé dans l'environnement.")

    destination_dir = Path(out_dir or config.DOWNLOADS_DIR)
    destination_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(destination_dir / "%(id)s.%(ext)s")
    format_spec = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]/best"
    )

    ydl_opts: Dict[str, Any] = {
        "format": format_spec,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    max_retries = config.DOWNLOAD_MAX_RETRIES
    backoff = config.DOWNLOAD_BACKOFF_BASE_SECONDS
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise DownloaderError(f"Aucune métadonnée retournée pour '{url}'")

                filepath = ydl.prepare_filename(info)
                p = Path(filepath)
                # Adaptation si fusionné en .mp4
                if not p.exists():
                    mp4_candidate = p.with_suffix(".mp4")
                    if mp4_candidate.exists():
                        filepath = str(mp4_candidate)

                logger.info("Téléchargement réussi : %s", filepath)
                return str(Path(filepath).resolve())

        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc):
                logger.error("Erreur permanente de téléchargement pour '%s' : %s", url, exc)
                raise PermanentDownloadError(
                    f"Téléchargement impossible (erreur permanente) pour '{url}' : {exc}"
                ) from exc

            if attempt < max_retries:
                logger.warning(
                    "Erreur transitoire sur tentative %d/%d pour '%s' (%s) -> retry dans %.1fs",
                    attempt, max_retries, url, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= config.DOWNLOAD_BACKOFF_FACTOR
            else:
                logger.error(
                    "Épuisement des tentatives (%d/%d) pour '%s' : %s",
                    attempt, max_retries, url, exc,
                )

    raise DownloaderError(
        f"Échec de téléchargement pour '{url}' après {max_retries} tentatives : {last_exc}"
    ) from last_exc
