"""Tests unitaires pour le module app.clipper.

100% mocké : zéro dépendance à ffmpeg ou OpenCV sur la machine de test.
Conforme au standard posé pour test_transcriber.py et test_highlights.py.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.clipper import (
    FFmpegProcessError,
    FFmpegTimeoutError,
    _run_ffmpeg,
    make_clip,
    make_clips,
)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Répertoire de sortie temporaire pour les clips."""
    out = tmp_path / "clips_out"
    out.mkdir()
    return out


@pytest.fixture
def sample_highlights() -> list:
    """Liste de highlights de test."""
    return [
        {
            "start_time": 10.0,
            "end_time": 40.0,
            "score": 95,
            "hook_sentence": "Moment fort 1",
            "reason": "Très engageant",
        },
        {
            "start_time": 120.0,
            "end_time": 150.0,
            "score": 88,
            "hook_sentence": "Moment fort 2",
            "reason": "Surprise",
        },
        {
            "start_time": 300.0,
            "end_time": 340.0,
            "score": 75,
            "hook_sentence": "Moment fort 3",
            "reason": "Pédagogique",
        },
    ]


def _mock_run_ffmpeg_creates_file(cmd, *args, **kwargs):
    """Side effect générique : crée le fichier de sortie attendu par ffmpeg."""
    for i, arg in enumerate(cmd):
        if arg == "-y" and i == 1:
            continue
    out_file = cmd[-1]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_bytes(b"fake video data")
    return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


def test_nominal_face_detected(tmp_output_dir: Path):
    """Cas 1 : Visage détecté → cadrage suivi, path renseigné, warning=None."""
    out_path = str(tmp_output_dir / "clip_01.mp4")

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._dynamic_face_crop") as mock_crop:

        mock_ffmpeg.side_effect = _mock_run_ffmpeg_creates_file
        mock_crop.return_value = {"face_detected": True}

        result = make_clip("/fake/source.mp4", 10.0, 40.0, out_path)

    assert result["path"] == out_path
    assert result["warning"] is None
    assert mock_crop.call_count == 1
    assert mock_ffmpeg.call_count == 2  # trim + remux audio


def test_no_face_detected(tmp_output_dir: Path):
    """Cas 2 : Aucun visage détecté → repli crop centré, path renseigné, warning présent."""
    out_path = str(tmp_output_dir / "clip_01.mp4")

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._dynamic_face_crop") as mock_crop:

        mock_ffmpeg.side_effect = _mock_run_ffmpeg_creates_file
        mock_crop.return_value = {"face_detected": False}

        result = make_clip("/fake/source.mp4", 10.0, 40.0, out_path)

    assert result["path"] == out_path
    assert result["warning"] is not None
    assert "Aucun visage" in result["warning"]


def test_opencv_technical_failure_fallback(tmp_output_dir: Path):
    """Cas 3 : OpenCV échoue → repli crop centré ffmpeg, path renseigné, warning présent."""
    out_path = str(tmp_output_dir / "clip_01.mp4")

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._dynamic_face_crop") as mock_crop, \
         patch("app.clipper._ffmpeg_crop_center") as mock_fallback:

        mock_ffmpeg.side_effect = _mock_run_ffmpeg_creates_file
        mock_crop.side_effect = RuntimeError("OpenCV crashed")
        mock_fallback.return_value = None
        # Créer le fichier de sortie attendu par le repli
        def create_out_file(src, dst):
            Path(dst).write_bytes(b"fallback video")
        mock_fallback.side_effect = create_out_file

        result = make_clip("/fake/source.mp4", 10.0, 40.0, out_path)

    assert result["path"] == out_path
    assert result["warning"] is not None
    assert "Échec technique" in result["warning"]
    mock_fallback.assert_called_once()


def test_ffmpeg_total_failure_batch_continues(
    tmp_output_dir: Path, sample_highlights: list,
):
    """Cas 4 : ffmpeg échoue sur 1 clip → path=None, les autres clips continuent."""
    call_idx = 0

    def selective_ffmpeg_failure(cmd, *args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        # Faire échouer le premier appel (trim du clip 1)
        if call_idx == 1:
            raise FFmpegProcessError("ffmpeg a refusé: codec invalide")
        return _mock_run_ffmpeg_creates_file(cmd, *args, **kwargs)

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._dynamic_face_crop") as mock_crop:

        mock_ffmpeg.side_effect = selective_ffmpeg_failure
        mock_crop.return_value = {"face_detected": True}

        results = make_clips("/fake/source.mp4", sample_highlights, str(tmp_output_dir))

    assert len(results) == 3

    # Le premier clip a échoué
    assert results[0]["path"] is None
    assert results[0]["warning"] is not None
    assert "Échec critique" in results[0]["warning"]

    # Les autres clips ont réussi
    assert results[1]["path"] is not None
    assert results[2]["path"] is not None


def test_temp_file_cleanup_on_failure(tmp_output_dir: Path):
    """Cas 5 : Fichier temporaire supprimé même quand le cadrage et le repli échouent."""
    out_path = str(tmp_output_dir / "clip_01.mp4")
    created_temp_files: list = []

    def trim_creates_real_file(cmd, *args, **kwargs):
        """Le trim crée un vrai fichier temporaire sur disque."""
        out_file = Path(cmd[-1])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b"trimmed video data")
        created_temp_files.append(out_file)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._dynamic_face_crop") as mock_crop, \
         patch("app.clipper._ffmpeg_crop_center") as mock_fallback:

        mock_ffmpeg.side_effect = trim_creates_real_file
        # Le cadrage dynamique échoue
        mock_crop.side_effect = RuntimeError("OpenCV crashed hard")
        # Le repli ffmpeg échoue aussi (double échec)
        mock_fallback.side_effect = FFmpegProcessError("ffmpeg fallback also failed")

        result = make_clip("/fake/source.mp4", 10.0, 40.0, out_path)

    # Le clip a échoué totalement
    assert result["path"] is None

    # VÉRIFICATION CRITIQUE : le fichier temporaire a bien été nettoyé
    assert len(created_temp_files) >= 1, "Le trim aurait dû créer un fichier temporaire"
    for tmp_file in created_temp_files:
        assert not tmp_file.exists(), (
            f"Le fichier temporaire {tmp_file} n'a pas été nettoyé par le bloc finally"
        )


def test_subprocess_security_no_shell_with_timeout():
    """Cas 6 : subprocess.run appelé avec list, sans shell=True, avec timeout."""
    test_cmd = ["ffmpeg", "-version"]

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(test_cmd, 0, stdout=b"", stderr=b"")
        _run_ffmpeg(test_cmd)

    mock_run.assert_called_once()
    call_args = mock_run.call_args

    # (a) Le premier argument est une liste
    cmd_arg = call_args[0][0] if call_args[0] else call_args[1].get("cmd")
    assert isinstance(cmd_arg, list), "La commande doit être une liste, pas une chaîne"

    # (b) shell n'est pas True
    shell_val = call_args[1].get("shell", False)
    assert shell_val is not True, "shell=True est interdit"

    # (c) timeout est présent et non-None
    timeout_val = call_args[1].get("timeout")
    assert timeout_val is not None, "timeout doit être explicitement défini"
    assert isinstance(timeout_val, (int, float)), "timeout doit être numérique"
    assert timeout_val > 0, "timeout doit être positif"


def test_run_ffmpeg_distinguishes_error_types():
    """Cas complémentaire : _run_ffmpeg distingue CalledProcessError de TimeoutExpired."""
    test_cmd = ["ffmpeg", "-i", "bad_file.mp4"]

    # CalledProcessError → FFmpegProcessError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            1, test_cmd, stderr=b"Invalid data found"
        )
        with pytest.raises(FFmpegProcessError, match="ffmpeg a échoué"):
            _run_ffmpeg(test_cmd)

    # TimeoutExpired → FFmpegTimeoutError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(test_cmd, 120)
        with pytest.raises(FFmpegTimeoutError, match="resté bloqué"):
            _run_ffmpeg(test_cmd)


def test_opencv_missing_cascade_classifier_triggers_fallback(tmp_output_dir: Path):
    """Cas spécifique : cv2 sans CascadeClassifier (ex: OpenCV 5) déclenche un repli propre."""
    out_path = str(tmp_output_dir / "clip_nocascade.mp4")

    mock_cv2 = MagicMock(spec=[])  # aucun attribut par défaut

    with patch("app.clipper._run_ffmpeg") as mock_ffmpeg, \
         patch("app.clipper._ffmpeg_crop_center") as mock_fallback, \
         patch.dict("sys.modules", {"cv2": mock_cv2}):

        mock_ffmpeg.side_effect = _mock_run_ffmpeg_creates_file

        def create_out_file(src, dst):
            Path(dst).write_bytes(b"fallback video")

        mock_fallback.side_effect = create_out_file

        result = make_clip("/fake/source.mp4", 10.0, 40.0, out_path)

    assert result["path"] == out_path
    assert result["warning"] is not None
    assert "CascadeClassifier" in result["warning"] or "Échec technique" in result["warning"]
    mock_fallback.assert_called_once()

