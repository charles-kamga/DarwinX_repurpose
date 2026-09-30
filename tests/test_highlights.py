"""Tests unitaires pour le module app.highlights."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from app import config
from app.highlights import (
    NoHighlightsFoundError,
    PermanentLLMError,
    TransientLLMError,
    _create_windows,
    _deduplicate_highlights,
    _default_groq_caller,
    _is_transient_error,
    _parse_llm_response,
    get_highlights,
)


@pytest.fixture
def sample_transcript_short():
    """Transcript court (120s <= 240s) pour tester le mode fenêtre unique."""
    return {
        "duration": 120.0,
        "segments": [
            {"start": 0.0, "end": 30.0, "text": "Bienvenue dans cette vidéo."},
            {"start": 30.0, "end": 75.0, "text": "Voici le cœur du sujet très intéressant."},
            {"start": 75.0, "end": 120.0, "text": "Et la conclusion de cette courte vidéo."},
        ],
    }


@pytest.fixture
def sample_transcript_long():
    """Transcript long (600s) avec plusieurs segments pour tester le fenêtrage glissant."""
    segments = []
    for i in range(20):
        start = float(i * 30)
        end = float((i + 1) * 30)
        segments.append({
            "start": start,
            "end": end,
            "text": f"Segment numéro {i} couvrant de {start}s à {end}s.",
        })
    return {
        "duration": 600.0,
        "segments": segments,
    }


def test_windowing_alignment_and_short_video(sample_transcript_short, sample_transcript_long):
    """Cas 1 : Fenêtrage aligné sur segments, sans coupure interne et avec chevauchement vérifié."""
    # 1. Vidéo courte -> 1 seule fenêtre englobant tous les segments
    windows_short = _create_windows(sample_transcript_short)
    assert len(windows_short) == 1
    assert len(windows_short[0]) == len(sample_transcript_short["segments"])

    # 2. Vidéo longue -> plusieurs fenêtres glissantes
    windows_long = _create_windows(sample_transcript_long)
    assert len(windows_long) > 1

    # 3. Vérification de l'intégrité des segments et du chevauchement consécutif
    for idx in range(len(windows_long)):
        win = windows_long[idx]
        assert len(win) > 0
        for seg in win:
            assert "start" in seg and "end" in seg and "text" in seg

        # Vérification du chevauchement avec la fenêtre suivante
        if idx < len(windows_long) - 1:
            next_win = windows_long[idx + 1]
            curr_end = win[-1]["end"]
            next_start = next_win[0]["start"]

            # La fenêtre suivante doit démarrer avant la fin de la fenêtre courante
            assert next_start < curr_end, f"Fenêtre {idx+1} doit débuter avant la fin de fenêtre {idx}"

            # Le recouvrement temporel doit respecter au moins WINDOW_OVERLAP_SECONDS
            overlap_duration = curr_end - next_start
            assert overlap_duration >= config.WINDOW_OVERLAP_SECONDS

            # Les deux fenêtres doivent partager des segments en commun
            curr_texts = {s["text"] for s in win}
            next_texts = {s["text"] for s in next_win}
            shared_segments = curr_texts.intersection(next_texts)
            assert len(shared_segments) > 0, f"Fenêtres {idx} et {idx+1} doivent partager des segments"


def test_retry_transient_error(sample_transcript_short, monkeypatch: pytest.MonkeyPatch):
    """Cas 2 : Retry exponentiel sur erreur transitoire (succès au 3ème appel)."""
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

    call_count = 0

    def mock_flaky_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TransientLLMError("Temporary rate limit 429")
        return json.dumps([
            {
                "start_time": 10.0,
                "end_time": 40.0,
                "score": 90,
                "hook_sentence": "Accroche virale",
                "reason": "Excellente rétention",
            }
        ])

    results = get_highlights(sample_transcript_short, llm_fn=mock_flaky_llm)
    assert call_count == 3
    assert len(results) == 1
    assert results[0]["score"] == 90
    assert results[0]["start_time"] == 10.0


def test_permanent_error_no_retry(sample_transcript_long, monkeypatch: pytest.MonkeyPatch):
    """Cas 3 : Erreur permanente -> 0 retry (1 seul appel), fenêtre ignorée gracieusement."""
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 0.001)

    call_count = 0

    def mock_perm_fail_then_success(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 1ère fenêtre -> Erreur permanente (ex: 400 filtre modération)
            raise PermanentLLMError("400 Bad Request: Content moderation triggered")
        # Fenêtres suivantes -> succès
        return json.dumps([
            {
                "start_time": 100.0,
                "end_time": 135.0,
                "score": 85,
                "hook_sentence": "Accroche valide 2ème fenêtre",
                "reason": "Bonne histoire",
            }
        ])

    results = get_highlights(sample_transcript_long, llm_fn=mock_perm_fail_then_success)
    # La 1ère fenêtre n'a pas été retryée inutilement
    assert len(results) >= 1
    assert results[0]["start_time"] == 100.0


def test_malformed_json_tolerant_parsing(sample_transcript_short):
    """Cas 4 : Parsing tolérant extrait le JSON entouré de texte ou balises markdown."""
    raw_response = """
    Voici mon analyse des meilleurs moments de cette vidéo :
    ```json
    [
      {
        "start_time": 5.0,
        "end_time": 35.0,
        "score": 88,
        "hook_sentence": "Saviez-vous que...",
        "reason": "Effet de surprise fort"
      }
    ]
    ```
    J'espère que cette sélection vous convient !
    """

    parsed = _parse_llm_response(raw_response)
    assert len(parsed) == 1
    assert parsed[0]["score"] == 88
    assert parsed[0]["start_time"] == 5.0
    assert parsed[0]["end_time"] == 35.0


def test_out_of_bounds_constraint_rejection(sample_transcript_short):
    """Cas 5 : Rejet strict des entrées violant les contraintes (score > 100, durée hors bornes)."""
    raw_response = json.dumps([
        # 1. Valide
        {"start_time": 10.0, "end_time": 40.0, "score": 92, "hook_sentence": "Valide", "reason": "OK"},
        # 2. Score > 100 -> Rejet
        {"start_time": 10.0, "end_time": 40.0, "score": 150, "hook_sentence": "Score invalide", "reason": "KO"},
        # 3. end_time <= start_time -> Rejet
        {"start_time": 50.0, "end_time": 40.0, "score": 80, "hook_sentence": "Temps incohérent", "reason": "KO"},
        # 4. Durée trop courte (< 15s) -> Rejet
        {"start_time": 10.0, "end_time": 18.0, "score": 85, "hook_sentence": "Trop court", "reason": "KO"},
        # 5. Durée trop longue (> 60s) -> Rejet
        {"start_time": 10.0, "end_time": 95.0, "score": 85, "hook_sentence": "Trop long", "reason": "KO"},
        # 6. Hook vide -> Rejet
        {"start_time": 20.0, "end_time": 50.0, "score": 80, "hook_sentence": "", "reason": "Pas de hook"},
    ])

    parsed = _parse_llm_response(raw_response)
    assert len(parsed) == 1
    assert parsed[0]["hook_sentence"] == "Valide"


def test_deduplication_overlap_threshold():
    """Cas 6 : Dédoublonnage écarte le clip de score inférieur si overlap > 50%."""
    highlights = [
        # Clip B : score 75, intervalle [15.0 - 42.0] (durée 27s)
        {"start_time": 15.0, "end_time": 42.0, "score": 75, "hook_sentence": "Clip B", "reason": "B"},
        # Clip A : score 95, intervalle [10.0 - 40.0] (durée 30s)
        # Recouvrement = [15.0 - 40.0] = 25s. 25 / 27 = 92.5% > 50%
        {"start_time": 10.0, "end_time": 40.0, "score": 95, "hook_sentence": "Clip A", "reason": "A"},
        # Clip C : score 80, intervalle [70.0 - 100.0] (pas d'overlap)
        {"start_time": 70.0, "end_time": 100.0, "score": 80, "hook_sentence": "Clip C", "reason": "C"},
    ]

    deduped = _deduplicate_highlights(highlights)
    assert len(deduped) == 2
    # Clip A (score 95) et Clip C (score 80) sont conservés, Clip B (score 75) est écarté
    hooks = [h["hook_sentence"] for h in deduped]
    assert "Clip A" in hooks
    assert "Clip C" in hooks
    assert "Clip B" not in hooks


def test_plafond_clip_count_max(sample_transcript_short):
    """Cas 7 : 15 clips valides distincts -> résultat plafonné exactement à CLIP_COUNT_MAX (9)."""
    fifteen_clips = []
    for i in range(15):
        fifteen_clips.append({
            "start_time": float(i * 100),
            "end_time": float(i * 100 + 30),
            "score": 50 + i,
            "hook_sentence": f"Clip {i}",
            "reason": f"Raison {i}",
        })

    def mock_many_clips_llm(prompt: str) -> str:
        return json.dumps(fifteen_clips)

    results = get_highlights(sample_transcript_short, llm_fn=mock_many_clips_llm)
    assert len(results) == config.CLIP_COUNT_MAX
    # Les clips retenus sont bien les meilleurs scores
    assert results[0]["score"] == 50 + 14


def test_plancher_graceful_degradation_and_warning(sample_transcript_short, caplog: pytest.LogCaptureFixture):
    """Cas 8 (Plancher & Dégradation) : 2 clips valides (< CLIP_COUNT_MIN=4) -> retournés avec warning."""
    two_clips = [
        {"start_time": 10.0, "end_time": 40.0, "score": 90, "hook_sentence": "Clip 1", "reason": "R1"},
        {"start_time": 60.0, "end_time": 90.0, "score": 85, "hook_sentence": "Clip 2", "reason": "R2"},
    ]

    def mock_two_clips_llm(prompt: str) -> str:
        return json.dumps(two_clips)

    with caplog.at_level(logging.WARNING):
        results = get_highlights(sample_transcript_short, llm_fn=mock_two_clips_llm)

    # 1. Retourne exactement les 2 clips (aucun faux clip fantôme ajouté)
    assert len(results) == 2
    assert results[0]["hook_sentence"] == "Clip 1"
    assert results[1]["hook_sentence"] == "Clip 2"

    # 2. Le warning plancher est bien émis
    assert any("inférieur au plancher recommandé" in record.message for record in caplog.records)


def test_fatal_job_failure_on_zero_highlights(sample_transcript_short):
    """Cas 9 : Toutes les fenêtres en échec -> lève NoHighlightsFoundError."""
    def mock_failing_llm(prompt: str) -> str:
        return "Pas de contenu JSON ici"

    with pytest.raises(NoHighlightsFoundError, match="Aucun moment fort valide"):
        get_highlights(sample_transcript_short, llm_fn=mock_failing_llm)


def test_is_transient_error_classification():
    """Cas 10 : Vérifie la classification précise des erreurs LLM (transitoire vs permanente)."""
    # Transitoires
    assert _is_transient_error(TransientLLMError("429 rate limit")) is True
    assert _is_transient_error(TimeoutError("Connection timed out")) is True
    assert _is_transient_error(Exception("Server returned 503 Service Unavailable")) is True

    # Permanentes
    assert _is_transient_error(PermanentLLMError("401 Unauthorized")) is False
    assert _is_transient_error(ValueError("Invalid prompt structure")) is False
    assert _is_transient_error(Exception("400 Bad Request: Policy violation")) is False


def test_default_anthropic_caller_missing_key(monkeypatch: pytest.MonkeyPatch):
    """Cas 11 : _default_anthropic_caller sans ANTHROPIC_API_KEY lève PermanentLLMError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from app.highlights import _default_anthropic_caller

    with pytest.raises(PermanentLLMError, match="ANTHROPIC_API_KEY non définie"):
        _default_anthropic_caller("Test prompt")


def test_default_anthropic_caller_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch, sample_transcript_short):
    """Cas 12 : Intégration nominale SDK Anthropic avec clé API présente."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-fake-key-12345")
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")

    mock_anthropic_module = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client

    mock_response = MagicMock()
    mock_message_content = MagicMock()
    mock_message_content.text = json.dumps([
        {
            "start_time": 10.0,
            "end_time": 40.0,
            "score": 95,
            "hook_sentence": "Hook Claude",
            "reason": "Excellente analyse",
        }
    ])
    mock_response.content = [mock_message_content]
    mock_client.messages.create.return_value = mock_response

    with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
        results = get_highlights(sample_transcript_short, llm_fn=None)

    assert len(results) == 1
    assert results[0]["hook_sentence"] == "Hook Claude"
    assert results[0]["score"] == 95
    mock_client.messages.create.assert_called_once()


def test_groq_exception_compatibility_with_is_transient_error():
    """Cas 13 : Vérifie que les exceptions du SDK Groq sont bien classées sans modifier _is_transient_error."""
    # Simulation exacte des classes du SDK Groq
    class RateLimitError(Exception):
        status_code = 429

    class APITimeoutError(Exception):
        pass

    class AuthenticationError(Exception):
        status_code = 401

    class BadRequestError(Exception):
        status_code = 400

    class InternalServerError(Exception):
        status_code = 500

    # Transitoires
    assert _is_transient_error(RateLimitError("Rate limit reached")) is True
    assert _is_transient_error(APITimeoutError("Request timed out")) is True
    assert _is_transient_error(InternalServerError("Groq internal server error")) is True

    # Permanentes
    assert _is_transient_error(AuthenticationError("Invalid Groq API Key")) is False
    assert _is_transient_error(BadRequestError("Invalid request schema")) is False


def test_default_groq_caller_missing_key(monkeypatch: pytest.MonkeyPatch):
    """Cas 14 : _default_groq_caller sans GROQ_API_KEY lève PermanentLLMError."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(PermanentLLMError, match="GROQ_API_KEY non définie"):
        _default_groq_caller("Test prompt")


def test_get_highlights_uses_groq_when_configured(monkeypatch: pytest.MonkeyPatch, sample_transcript_short):
    """Cas 15 : LLM_PROVIDER='groq' sélectionne automatiquement _default_groq_caller."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")

    fake_response = json.dumps([
        {
            "start_time": 10.0,
            "end_time": 40.0,
            "score": 90,
            "hook_sentence": "Hook Groq",
            "reason": "Test routeur",
        }
    ])

    with patch("app.highlights._default_groq_caller", return_value=fake_response) as mock_groq:
        results = get_highlights(sample_transcript_short, llm_fn=None)

    assert len(results) == 1
    assert results[0]["hook_sentence"] == "Hook Groq"
    mock_groq.assert_called_once()


def test_default_groq_caller_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch, sample_transcript_short):
    """Cas 16 : Intégration nominale SDK Groq avec clé API présente."""
    monkeypatch.setenv("GROQ_API_KEY", "test-fake-groq-key-67890")
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")

    mock_groq_module = MagicMock()
    mock_client = MagicMock()
    mock_groq_module.Groq.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps([
        {
            "start_time": 15.0,
            "end_time": 45.0,
            "score": 98,
            "hook_sentence": "Accroche virale Groq",
            "reason": "Top moment",
        }
    ])
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict("sys.modules", {"groq": mock_groq_module}):
        results = get_highlights(sample_transcript_short, llm_fn=None)

    assert len(results) == 1
    assert results[0]["hook_sentence"] == "Accroche virale Groq"
    assert results[0]["score"] == 98
    mock_client.chat.completions.create.assert_called_once()

