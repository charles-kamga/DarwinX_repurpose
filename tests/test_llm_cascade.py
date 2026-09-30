"""Tests unitaires pour le module app.llm_cascade.

Couvre la cascade multi-fournisseurs :
- Mode auto : bascule transparente en cascade (Groq -> Gemini -> Mistral -> Anthropic).
- Mode forcé : échec strict sans repli silencieux.
- Adaptateurs individuels : Groq, Gemini, Mistral, Anthropic.
- Gestion d'erreurs (429, 401, 5xx, timeouts, clés absentes).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app import config, llm_cascade
from app.llm_cascade import (
    LLMError,
    PermanentLLMError,
    TransientLLMError,
    _call_anthropic,
    _call_gemini,
    _call_groq,
    _call_mistral,
    resolve_llm_caller,
)


def test_cascade_auto_selects_first_available_key(monkeypatch: pytest.MonkeyPatch):
    """Cas 1 : En mode auto, le premier fournisseur avec clé configurée est sollicité en priorité."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")

    with patch("app.llm_cascade._call_groq", return_value='[{"score": 90}]') as mock_groq, \
         patch("app.llm_cascade._call_gemini") as mock_gemini:

        caller = resolve_llm_caller("auto")
        res = caller("Test prompt")

    assert res == '[{"score": 90}]'
    mock_groq.assert_called_once_with("Test prompt")
    mock_gemini.assert_not_called()


def test_cascade_switches_on_rate_limit_429(monkeypatch: pytest.MonkeyPatch):
    """Cas 2 : En mode auto, si Groq renvoie un 429, bascule transparente sur Gemini."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")

    with patch("app.llm_cascade._call_groq", side_effect=TransientLLMError("Rate limit 429")) as mock_groq, \
         patch("app.llm_cascade._call_gemini", return_value='[{"score": 88}]') as mock_gemini:

        caller = resolve_llm_caller("auto")
        res = caller("Test prompt")

    assert res == '[{"score": 88}]'
    mock_groq.assert_called_once()
    mock_gemini.assert_called_once_with("Test prompt")


def test_cascade_switches_groq_gemini_mistral(monkeypatch: pytest.MonkeyPatch):
    """Cas 3 : Bascule en cascade Groq (429) -> Gemini (503) -> Mistral (succès)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral_test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")

    with patch("app.llm_cascade._call_groq", side_effect=TransientLLMError("429")), \
         patch("app.llm_cascade._call_gemini", side_effect=TransientLLMError("503")), \
         patch("app.llm_cascade._call_mistral", return_value='[{"score": 85}]') as mock_mistral:

        caller = resolve_llm_caller("auto")
        res = caller("Test prompt")

    assert res == '[{"score": 85}]'
    mock_mistral.assert_called_once_with("Test prompt")


def test_cascade_forced_provider_strict_error(monkeypatch: pytest.MonkeyPatch):
    """Cas 4 : En mode forcé, un échec du fournisseur exigé lève une exception stricte sans repli."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")

    with patch("app.llm_cascade._call_groq", side_effect=TransientLLMError("429")) as mock_groq, \
         patch("app.llm_cascade._call_gemini") as mock_gemini:

        caller = resolve_llm_caller("groq")
        with pytest.raises(TransientLLMError, match="429"):
            caller("Test prompt")

    mock_groq.assert_called_once()
    mock_gemini.assert_not_called()


def test_cascade_raises_when_no_api_keys(monkeypatch: pytest.MonkeyPatch):
    """Cas 5 : En mode auto, si aucune clé API n'est définie, lève PermanentLLMError."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    caller = resolve_llm_caller("auto")
    with pytest.raises(PermanentLLMError, match="Aucune clé API LLM configurée"):
        caller("Test prompt")


def test_cascade_raises_when_all_fail(monkeypatch: pytest.MonkeyPatch):
    """Cas 6 : En mode auto, si tous les fournisseurs échouent, lève PermanentLLMError explicite."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")

    with patch("app.llm_cascade._call_groq", side_effect=TransientLLMError("Groq 429")), \
         patch("app.llm_cascade._call_gemini", side_effect=PermanentLLMError("Gemini 401")):

        caller = resolve_llm_caller("auto")
        with pytest.raises(PermanentLLMError, match="Tous les fournisseurs LLM de la cascade ont échoué"):
            caller("Test prompt")


def test_cascade_raises_transient_when_all_fail_transient(monkeypatch: pytest.MonkeyPatch):
    """Cas 6b : En mode auto, si tous les fournisseurs échouent sur des erreurs transitoires, lève TransientLLMError."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test")

    with patch("app.llm_cascade._call_groq", side_effect=TransientLLMError("Groq 429")), \
         patch("app.llm_cascade._call_gemini", side_effect=TransientLLMError("Gemini 503")):

        caller = resolve_llm_caller("auto")
        with pytest.raises(TransientLLMError, match="erreurs transitoires"):
            caller("Test prompt")


def test_gemini_caller_nominal_and_error(monkeypatch: pytest.MonkeyPatch):
    """Cas 7 : Adaptateur Gemini - nominal, 429 rate limit, 401 auth."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    # Nominal
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"result": "gemini_ok"}'}]}}]
    }
    with patch("httpx.Client.post", return_value=mock_resp):
        res = _call_gemini("hello")
        assert res == '{"result": "gemini_ok"}'

    # 429 -> TransientLLMError
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    with patch("httpx.Client.post", return_value=mock_resp_429):
        with pytest.raises(TransientLLMError, match="Rate-limit"):
            _call_gemini("hello")

    # 401 -> PermanentLLMError
    mock_resp_401 = MagicMock()
    mock_resp_401.status_code = 401
    with patch("httpx.Client.post", return_value=mock_resp_401):
        with pytest.raises(PermanentLLMError, match="invalide"):
            _call_gemini("hello")


def test_mistral_caller_nominal_and_error(monkeypatch: pytest.MonkeyPatch):
    """Cas 8 : Adaptateur Mistral - nominal, 429 rate limit, 403 auth."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")

    # Nominal
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"result": "mistral_ok"}'}}]
    }
    with patch("httpx.Client.post", return_value=mock_resp):
        res = _call_mistral("hello")
        assert res == '{"result": "mistral_ok"}'

    # 429 -> TransientLLMError
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    with patch("httpx.Client.post", return_value=mock_resp_429):
        with pytest.raises(TransientLLMError, match="Rate-limit"):
            _call_mistral("hello")

    # 403 -> PermanentLLMError
    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403
    with patch("httpx.Client.post", return_value=mock_resp_403):
        with pytest.raises(PermanentLLMError, match="invalide"):
            _call_mistral("hello")


def test_groq_and_anthropic_missing_keys(monkeypatch: pytest.MonkeyPatch):
    """Cas 9 : Clé manquante lève PermanentLLMError immédiatement sans requête."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(PermanentLLMError, match="GROQ_API_KEY"):
        _call_groq("test")

    with pytest.raises(PermanentLLMError, match="ANTHROPIC_API_KEY"):
        _call_anthropic("test")


def test_unknown_forced_provider_raises_error():
    """Cas 10 : Fournisseur forcé inconnu lève PermanentLLMError."""
    with pytest.raises(PermanentLLMError, match="Fournisseur LLM inconnu : 'unknown_provider'"):
        resolve_llm_caller("unknown_provider")
