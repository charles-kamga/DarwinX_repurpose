"""Tests unitaires pour le module d'ingestion YouTube (app.downloader).

100% mocké : zéro dépendance réseau ou package externe, déterministe et rapide.
Couvre tous les cas d'acceptation du SKILL.md et d'AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import config
from app.downloader import (
    DownloaderError,
    PermanentDownloadError,
    _is_transient_error,
    download_youtube,
    probe_max_quality,
)


@pytest.fixture
def mock_ydl():
    """Fixture fournissant un mock complet du module yt_dlp dans app.downloader."""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.__enter__.return_value = mock_instance
    mock_module.YoutubeDL.return_value = mock_instance

    with patch("app.downloader.yt_dlp", mock_module):
        yield mock_module, mock_instance


def test_is_transient_error_classification():
    """Vérifie la classification fine entre erreurs transitoires et permanentes."""
    # Transitoires
    assert _is_transient_error(Exception("HTTP Error 403: Forbidden")) is True
    assert _is_transient_error(Exception("HTTP Error 429: Too Many Requests")) is True
    assert _is_transient_error(Exception("Connection reset by peer")) is True
    assert _is_transient_error(Exception("Read timed out")) is True

    # Permanentes
    assert _is_transient_error(Exception("is not a valid URL")) is False
    assert _is_transient_error(Exception("This video is private")) is False
    assert _is_transient_error(Exception("Video unavailable")) is False
    assert _is_transient_error(Exception("Sign in to confirm your age")) is False

    # Erreur inconnue -> permanente par défaut (fail-safe)
    assert _is_transient_error(Exception("Unexpected schema mismatch")) is False


def test_probe_max_quality_nominal(mock_ydl):
    """Cas 1 : Sonde de qualité nominale extrayant max_height et duration."""
    _, mock_instance = mock_ydl
    mock_instance.extract_info.return_value = {
        "duration": 300,
        "formats": [
            {"height": 360},
            {"height": 720},
            {"height": 1080},
            {"height": None},  # format audio
        ],
    }

    res = probe_max_quality("https://www.youtube.com/watch?v=valid_id")
    assert res == {"max_height": 1080, "duration": 300}


def test_probe_max_quality_fail_open_on_exception(mock_ydl):
    """Cas 2 : Sonde de qualité fail-open sur exception réseau/inconnue (ne lève JAMAIS)."""
    _, mock_instance = mock_ydl
    mock_instance.extract_info.side_effect = RuntimeError("YouTube API layout changed")

    # Ne doit lever aucune exception
    res = probe_max_quality("https://www.youtube.com/watch?v=broken_id")
    assert res == {"max_height": 0, "duration": 0}


def test_download_youtube_nominal(mock_ydl, tmp_path: Path):
    """Cas 3 : Téléchargement nominal réussi sans retry."""
    _, mock_instance = mock_ydl
    fake_file = tmp_path / "vid123.mp4"
    fake_file.write_bytes(b"video payload")

    mock_instance.extract_info.return_value = {"id": "vid123", "ext": "mp4"}
    mock_instance.prepare_filename.return_value = str(fake_file)

    result_path = download_youtube(
        url="https://www.youtube.com/watch?v=vid123",
        out_dir=str(tmp_path),
        max_height=1080,
    )

    assert result_path == str(fake_file.resolve())
    assert mock_instance.extract_info.call_count == 1


def test_download_youtube_transient_retry_success(mock_ydl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cas 4 : Erreur transitoire 403 à la tentative 1, succès à la tentative 2."""
    _, mock_instance = mock_ydl
    monkeypatch.setattr(config, "DOWNLOAD_BACKOFF_BASE_SECONDS", 0.01)
    fake_file = tmp_path / "vid456.mp4"
    fake_file.write_bytes(b"video payload")

    attempts = 0

    def mock_extract(url, download=True):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("HTTP Error 403: Forbidden")
        return {"id": "vid456", "ext": "mp4"}

    mock_instance.extract_info.side_effect = mock_extract
    mock_instance.prepare_filename.return_value = str(fake_file)

    result_path = download_youtube(
        url="https://www.youtube.com/watch?v=vid456",
        out_dir=str(tmp_path),
    )

    assert result_path == str(fake_file.resolve())
    assert attempts == 2


def test_download_youtube_permanent_error_no_retry(mock_ydl, tmp_path: Path):
    """Cas 5 : Erreur permanente (URL invalide) -> échec immédiat, aucun retry."""
    _, mock_instance = mock_ydl
    mock_instance.extract_info.side_effect = RuntimeError("Invalid URL: not a valid URL")

    with pytest.raises(PermanentDownloadError, match="erreur permanente"):
        download_youtube(
            url="https://invalid-url",
            out_dir=str(tmp_path),
        )

    # Vérifie qu'il n'y a eu qu'une seule tentative (aucun retry)
    assert mock_instance.extract_info.call_count == 1


def test_download_youtube_transient_retries_exhausted(mock_ydl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cas 6 : Erreur transitoire répétée -> épuisement des retries et DownloaderError."""
    _, mock_instance = mock_ydl
    monkeypatch.setattr(config, "DOWNLOAD_MAX_RETRIES", 3)
    monkeypatch.setattr(config, "DOWNLOAD_BACKOFF_BASE_SECONDS", 0.01)

    mock_instance.extract_info.side_effect = RuntimeError("HTTP Error 429: Too Many Requests")

    with pytest.raises(DownloaderError, match="après 3 tentatives"):
        download_youtube(
            url="https://www.youtube.com/watch?v=rate_limited",
            out_dir=str(tmp_path),
        )

    assert mock_instance.extract_info.call_count == 3
