"""Tests unitaires pour le module app.telegram_notifier (Notifications Telegram résilientes)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app import telegram_notifier


def test_telegram_disabled_when_credentials_missing(monkeypatch: pytest.MonkeyPatch):
    """Cas 1 : En l'absence de tokens, send_telegram_message retourne False sans appel réseau."""
    monkeypatch.delenv("DARWINX_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DARWINX_TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr("app.config.TELEGRAM_BOT_TOKEN", None)
    monkeypatch.setattr("app.config.TELEGRAM_CHAT_ID", None)

    with patch("httpx.Client") as mock_client:
        success = telegram_notifier.send_telegram_message("Test message")
        assert success is False
        mock_client.assert_not_called()


def test_telegram_send_nominal_success(monkeypatch: pytest.MonkeyPatch):
    """Cas 2 : Envoi nominal en Markdown avec code HTTP 200."""
    monkeypatch.setenv("DARWINX_TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("DARWINX_TELEGRAM_CHAT_ID", "123456")

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__enter__.return_value = mock_client_instance

    with patch("httpx.Client", return_value=mock_client_instance):
        success = telegram_notifier.send_telegram_message("Message de test")
        assert success is True
        assert mock_client_instance.post.call_count == 1
        call_kwargs = mock_client_instance.post.call_args.kwargs
        assert call_kwargs["json"]["chat_id"] == "123456"
        assert call_kwargs["json"]["parse_mode"] == "Markdown"


def test_telegram_markdown_fallback_on_400(monkeypatch: pytest.MonkeyPatch):
    """Cas 3 : Si Telegram rejette le Markdown (HTTP 400), repli immédiat en texte brut."""
    monkeypatch.setenv("DARWINX_TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("DARWINX_TELEGRAM_CHAT_ID", "123456")

    mock_resp_400 = MagicMock()
    mock_resp_400.status_code = 400
    mock_resp_400.text = "Bad Request: can't parse entities"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = [mock_resp_400, mock_resp_200]
    mock_client_instance.__enter__.return_value = mock_client_instance

    with patch("httpx.Client", return_value=mock_client_instance):
        success = telegram_notifier.send_telegram_message("Message avec *markdown* invalide [")
        assert success is True
        assert mock_client_instance.post.call_count == 2
        # La seconde tentative ne doit pas avoir de parse_mode
        second_call_json = mock_client_instance.post.call_args_list[1].kwargs["json"]
        assert "parse_mode" not in second_call_json


def test_telegram_network_timeout_fail_safe(monkeypatch: pytest.MonkeyPatch):
    """Cas 4 : Confinement strict — une exception réseau ou timeout ne bloque jamais."""
    import httpx

    monkeypatch.setenv("DARWINX_TELEGRAM_BOT_TOKEN", "fake_bot_token")
    monkeypatch.setenv("DARWINX_TELEGRAM_CHAT_ID", "123456")

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.TimeoutException("Connection timed out")
    mock_client_instance.__enter__.return_value = mock_client_instance

    with patch("httpx.Client", return_value=mock_client_instance):
        # Ne doit JAMAIS lever d'exception
        success = telegram_notifier.send_telegram_message("Test timeout")
        assert success is False


def test_telegram_messages_contain_darwinx_repurpose_tag():
    """Cas 5 : Toutes les alertes spécialisées comportent le tag explicite [Darwinx_repurpose]."""
    with patch("app.telegram_notifier.send_telegram_message", return_value=True) as mock_send:
        # 1. Start
        telegram_notifier.notify_job_start("job-1", "test.mp4", "cuda", "groq")
        args_start, _ = mock_send.call_args
        assert "[Darwinx_repurpose]" in args_start[0]

        # 2. Step done
        telegram_notifier.notify_step_done("job-1", "Transcription", {"durée": "12s"})
        args_step, _ = mock_send.call_args
        assert "[Darwinx_repurpose]" in args_step[0]
        assert "Transcription" in args_step[0]

        # 3. Completed
        clips = [{"score": 90, "start_time": 0.0, "end_time": 20.0, "final_path": "/out/c1.mp4"}]
        telegram_notifier.notify_job_completed("job-1", 45.2, clips, ["Warning 1"])
        args_comp, _ = mock_send.call_args
        assert "[Darwinx_repurpose]" in args_comp[0]
        assert "Warning 1" in args_comp[0]

        # 4. Error
        telegram_notifier.notify_job_error("job-1", "Transcription", "Audio muet")
        args_err, _ = mock_send.call_args
        assert "[Darwinx_repurpose]" in args_err[0]
        assert "Audio muet" in args_err[0]


def test_telegram_credentials_precedence(monkeypatch: pytest.MonkeyPatch):
    """Cas 6 : DARWINX_TELEGRAM_BOT_TOKEN prend la priorité sur TELEGRAM_BOT_TOKEN."""
    monkeypatch.setenv("DARWINX_TELEGRAM_BOT_TOKEN", "specific_token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "generic_token")
    monkeypatch.setenv("DARWINX_TELEGRAM_CHAT_ID", "999")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "111")

    token, chat_id = telegram_notifier._get_credentials()
    assert token == "specific_token"
    assert chat_id == "999"
