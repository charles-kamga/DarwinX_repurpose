"""Tests unitaires pour le module app.transcriber."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.transcriber import (
    EmptyTranscriptionError,
    TranscriberError,
    TranscriberMediaError,
    _compute_file_sha256,
    _detect_device_and_compute_type,
    _get_cache_path,
    _load_valid_cache,
    _setup_nvidia_cuda_paths,
    get_last_device_used,
    transcribe,
    was_last_from_cache,
)


class MockWord:
    """Simule un mot horodaté retourné par faster-whisper."""
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class MockSegment:
    """Simule un segment de parole retourné par faster-whisper."""
    def __init__(self, start: float, end: float, text: str, words: list[MockWord]):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class MockInfo:
    """Simule l'objet TranscriptionInfo retourné par faster-whisper."""
    def __init__(self, duration: float = 10.5, language: str = "fr"):
        self.duration = duration
        self.language = language


@pytest.fixture
def temp_media_file(tmp_path: Path) -> Path:
    """Crée un faux fichier média avec un contenu binaire déterminé."""
    file_path = tmp_path / "sample_video.mp4"
    file_path.write_bytes(b"FAKE_VIDEO_CONTENT_FOR_TESTING_12345")
    return file_path


@pytest.fixture
def mock_whisper_nominal():
    """Fournit un mock nominal de app.transcriber.faster_whisper."""
    mock_module = MagicMock()
    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_module.WhisperModel = mock_model_cls

    def make_segments():
        words = [
            MockWord("Bonjour", 0.0, 0.6),
            MockWord("le", 0.6, 0.8),
            MockWord("monde", 0.8, 1.4),
        ]
        segments = [
            MockSegment(0.0, 1.5, "Bonjour le monde", words)
        ]
        info = MockInfo(duration=10.5, language="fr")
        return iter(segments), info

    mock_instance.transcribe.side_effect = lambda *args, **kwargs: make_segments()

    with patch("app.transcriber.faster_whisper", mock_module):
        yield mock_model_cls, mock_instance


def test_transcribe_nominal(temp_media_file: Path, mock_whisper_nominal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cas 1 (Nominal) : Transcription réussie avec timestamps mot-par-mot conformes."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    _, mock_instance = mock_whisper_nominal

    result = transcribe(str(temp_media_file), language="fr")

    # Vérification du contrat de sortie
    assert "duration" in result
    assert "segments" in result
    assert result["duration"] == 10.5
    assert len(result["segments"]) == 1

    seg = result["segments"][0]
    assert seg["start"] == 0.0
    assert seg["end"] == 1.5
    assert seg["text"] == "Bonjour le monde"
    assert "words" in seg
    assert len(seg["words"]) == 3
    assert seg["words"][0] == {"w": "Bonjour", "s": 0.0, "e": 0.6}
    assert seg["words"][1] == {"w": "le", "s": 0.6, "e": 0.8}
    assert seg["words"][2] == {"w": "monde", "s": 0.8, "e": 1.4}

    # Vérification des options d'appel faster-whisper
    mock_instance.transcribe.assert_called_once_with(
        str(temp_media_file),
        language="fr",
        word_timestamps=True,
        vad_filter=True,
    )


def test_transcribe_cache_reuse(temp_media_file: Path, mock_whisper_nominal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cas 2 (Cache) : Deux appels successifs réutilisent le cache sans recalcul."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_cls, mock_instance = mock_whisper_nominal

    # Premier appel -> exécute la transcription et remplit le cache
    res1 = transcribe(str(temp_media_file), language="fr")
    assert mock_instance.transcribe.call_count == 1
    assert was_last_from_cache() is False

    # Deuxième appel -> lit le cache, n'appelle plus transcribe()
    res2 = transcribe(str(temp_media_file), language="fr")
    assert mock_instance.transcribe.call_count == 1
    assert was_last_from_cache() is True
    assert get_last_device_used() == "Cache disque"
    assert res1 == res2


def test_transcribe_corrupted_cache_triggers_retranscription(
    temp_media_file: Path, mock_whisper_nominal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 3 (Cache corrompu / vide) : Un cache avec 0 segments ou du JSON cassé est ignoré."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_cls, mock_instance = mock_whisper_nominal

    # Création manuelle d'un cache corrompu (0 segments)
    content_hash = _compute_file_sha256(temp_media_file)
    cache_path = _get_cache_path(content_hash, "medium", "fr")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"duration": 10.5, "segments": []}), encoding="utf-8")

    # Appel de transcribe -> doit ignorer le cache invalide et retranscrire
    res = transcribe(str(temp_media_file), language="fr")

    assert mock_instance.transcribe.call_count == 1
    assert len(res["segments"]) == 1

    # Vérification que le cache sur disque a été réécrit avec les données valides
    loaded = _load_valid_cache(cache_path)
    assert loaded is not None
    assert len(loaded["segments"]) == 1


def test_transcribe_silent_audio_raises_empty_transcription_error(
    temp_media_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 4 (Limite - Silence) : Aucune parole détectée lève EmptyTranscriptionError."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_module = MagicMock()
    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_module.WhisperModel = mock_model_cls
    # 0 segments retournés
    mock_instance.transcribe.return_value = (iter([]), MockInfo(duration=5.0))

    with patch("app.transcriber.faster_whisper", mock_module):
        with pytest.raises(EmptyTranscriptionError, match="Aucune parole détectée"):
            transcribe(str(temp_media_file))


def test_transcribe_non_existent_file_raises_file_not_found(tmp_path: Path):
    """Cas 5 : Fichier introuvable lève FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError, match="Fichier média introuvable"):
        transcribe(str(missing_file))


def test_transcribe_force_bypasses_cache(
    temp_media_file: Path, mock_whisper_nominal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 6 : force=True ignore le cache même s'il est valide."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    _, mock_instance = mock_whisper_nominal

    # Premier appel -> remplit le cache
    transcribe(str(temp_media_file), language="fr")
    assert mock_instance.transcribe.call_count == 1

    # Deuxième appel avec force=True -> réinvoque transcribe
    transcribe(str(temp_media_file), language="fr", force=True)
    assert mock_instance.transcribe.call_count == 2


def test_detect_device_and_compute_type(monkeypatch: pytest.MonkeyPatch):
    """Cas 7 : Vérifie la détection matérielle (override CPU, override CUDA, auto)."""
    from app import config

    # Cas 1 : WHISPER_DEVICE="cpu" -> force CPU sans toucher à l'auto-détection
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cpu")
    device, compute_type = _detect_device_and_compute_type()
    assert device == "cpu"
    assert compute_type == "int8"

    # Cas 2 : WHISPER_DEVICE="cuda" -> force CUDA directement
    monkeypatch.setattr(config, "WHISPER_DEVICE", "cuda")
    device, compute_type = _detect_device_and_compute_type()
    assert device == "cuda"
    assert compute_type == "float16"

    # Cas 3 : WHISPER_DEVICE="auto" avec CUDA détecté via ctranslate2
    monkeypatch.setattr(config, "WHISPER_DEVICE", "auto")
    mock_ct2_cuda = MagicMock()
    mock_ct2_cuda.get_cuda_device_count.return_value = 1
    with patch.dict("sys.modules", {"ctranslate2": mock_ct2_cuda}):
        device, compute_type = _detect_device_and_compute_type()
        assert device == "cuda"
        assert compute_type == "float16"

    # Cas 4 : WHISPER_DEVICE="auto" sans GPU disponible -> repli CPU
    mock_ct2_cpu = MagicMock()
    mock_ct2_cpu.get_cuda_device_count.return_value = 0
    mock_torch_cpu = MagicMock()
    mock_torch_cpu.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"ctranslate2": mock_ct2_cpu, "torch": mock_torch_cpu}):
        device, compute_type = _detect_device_and_compute_type()
        assert device == "cpu"
        assert compute_type == "int8"


def test_setup_nvidia_cuda_paths_graceful():
    """Cas spécifique : _setup_nvidia_cuda_paths s'exécute sans erreur même si répertoires absents."""
    # Test nominal d'exécution silencieuse fail-safe
    _setup_nvidia_cuda_paths()


def test_transcribe_media_error_on_transcribe_failure(
    temp_media_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 8 : Erreur ffmpeg/décodage pendant transcribe lève TranscriberMediaError."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_module = MagicMock()
    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_module.WhisperModel = mock_model_cls
    mock_instance.transcribe.side_effect = RuntimeError("ffmpeg decoding failed")

    with patch("app.transcriber.faster_whisper", mock_module):
        with pytest.raises(TranscriberMediaError, match="Échec de transcription"):
            transcribe(str(temp_media_file))


def test_transcribe_language_auto_and_custom_model_size(
    temp_media_file: Path, mock_whisper_nominal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 9 : language=None et model_size personnalisé."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_cls, mock_instance = mock_whisper_nominal

    result = transcribe(str(temp_media_file), language=None, model_size="small")

    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")
    mock_instance.transcribe.assert_called_once_with(
        str(temp_media_file),
        language=None,
        word_timestamps=True,
        vad_filter=True,
    )
    assert result["duration"] == 10.5


def test_transcribe_word_with_none_timestamps_graceful_degradation(
    temp_media_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cas 10 (Dégradation gracieuse) : Les mots avec timestamps None ne font pas échouer le clip."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TRANSCRIPT_CACHE_DIR", str(cache_dir))

    mock_module = MagicMock()
    mock_model_cls = MagicMock()
    mock_instance = MagicMock()
    mock_model_cls.return_value = mock_instance
    mock_module.WhisperModel = mock_model_cls

    # Segment contenant des mots avec différents types de None (asymétriques et symétriques)
    words = [
        MockWord("Bonjour", 0.0, 0.3),
        MockWord("le", None, 0.4),      # Cas asymétrique : start=None, end valide
        MockWord("beau", 0.4, None),    # Cas asymétrique : start valide, end=None
        MockWord("grand", None, None),  # Cas symétrique : start=None et end=None
        MockWord("monde", 0.8, 1.2),
    ]
    segments = [MockSegment(0.0, 1.2, "Bonjour le beau grand monde", words)]
    info = MockInfo(duration=10.5, language="fr")
    mock_instance.transcribe.return_value = (iter(segments), info)

    with patch("app.transcriber.faster_whisper", mock_module):
        result = transcribe(str(temp_media_file), language="fr")

    # Le clip est sauvé et livré, les 3 mots problématiques sont exclus de la liste fine
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "Bonjour le beau grand monde"
    assert seg["start"] == 0.0
    assert seg["end"] == 1.2
    assert len(seg["words"]) == 2
    assert seg["words"][0]["w"] == "Bonjour"
    assert seg["words"][1]["w"] == "monde"


def test_load_valid_cache_schema_validation(tmp_path: Path):
    """Cas 11 : Validation fine du cache (I/O, schéma, clés manquantes)."""
    # 1. Fichier non-JSON (I/O & JSONDecodeError)
    bad_json = tmp_path / "corrupted.json"
    bad_json.write_text("NOT_JSON_DATA_---@@@", encoding="utf-8")
    assert _load_valid_cache(bad_json) is None

    # 2. JSON non dictionnaire
    list_json = tmp_path / "list.json"
    list_json.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert _load_valid_cache(list_json) is None

    # 3. JSON sans durée valide
    no_dur = tmp_path / "no_dur.json"
    no_dur.write_text(json.dumps({"duration": -1.0, "segments": [{"start": 0, "end": 1, "text": "a", "words": []}]}), encoding="utf-8")
    assert _load_valid_cache(no_dur) is None

    # 4. JSON avec segment manquant de clés
    missing_keys = tmp_path / "missing_keys.json"
    missing_keys.write_text(json.dumps({"duration": 5.0, "segments": [{"start": 0}]}), encoding="utf-8")
    assert _load_valid_cache(missing_keys) is None

