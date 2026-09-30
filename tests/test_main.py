"""Tests unitaires pour le point d'entrée CLI main.py.

100% mocké : zéro dépendance externe, déterministe et rapide.
Vérifie l'aiguillage des sources, l'affichage et la gestion des erreurs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.downloader import DownloaderError
from app.pipeline import PipelineFatalError
from main import cli_main, resolve_project_dir


@pytest.fixture(autouse=True)
def configure_test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Configure un répertoire temporaire pour les jobs lors des tests CLI."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JOBS_DIR", str(jobs_dir))


def test_cli_main_with_local_file_nominal(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Cas 1 : Fichier local existant -> pas de téléchargement, exécution pipeline, code 0."""
    local_video = tmp_path / "local_video.mp4"
    local_video.write_bytes(b"dummy video data")
    out_dir = tmp_path / "out"

    fake_result = {
        "job_id": "fake_job_1",
        "status": "completed",
        "result": {
            "total_clips": 1,
            "clips": [
                {"final_path": str(out_dir / "clip_1.mp4"), "score": 90}
            ],
        },
    }

    with patch("main.downloader.download_youtube") as mock_dl, \
         patch("main.downloader.probe_max_quality") as mock_probe, \
         patch("main.pipeline.run_pipeline", return_value=fake_result) as mock_run:

        ret = cli_main([str(local_video), "-o", str(out_dir)])

    assert ret == 0
    assert mock_dl.call_count == 0
    assert mock_probe.call_count == 0
    assert mock_run.call_count == 1

    captured = capsys.readouterr()
    assert "Fichier local détecté" in captured.out
    assert "BILAN DU TRAITEMENT PIPELINE" in captured.out
    assert "Nombre de clips      : 1" in captured.out


def test_cli_main_with_youtube_url_nominal(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Cas 2 : URL YouTube -> sonde de qualité, téléchargement, exécution pipeline, code 0."""
    out_dir = tmp_path / "out"
    downloaded_video = tmp_path / "downloaded.mp4"
    downloaded_video.write_bytes(b"dummy video data")

    fake_result = {
        "job_id": "fake_job_yt",
        "status": "completed",
        "result": {
            "total_clips": 2,
            "clips": [
                {"final_path": str(out_dir / "c1.mp4"), "score": 85},
                {"final_path": str(out_dir / "c2.mp4"), "score": 92},
            ],
        },
    }

    with patch("main.downloader.probe_max_quality", return_value={"max_height": 1080, "duration": 120}) as mock_probe, \
         patch("main.downloader.download_youtube", return_value=str(downloaded_video)) as mock_dl, \
         patch("main.pipeline.run_pipeline", return_value=fake_result) as mock_run:

        ret = cli_main(["https://www.youtube.com/watch?v=sample123", "-o", str(out_dir)])

    assert ret == 0
    assert mock_probe.call_count == 1
    assert mock_dl.call_count == 1
    assert mock_run.call_count == 1

    captured = capsys.readouterr()
    assert "URL YouTube détectée" in captured.out
    assert "Résolution source dispo : 1080p" in captured.out
    assert "Nombre de clips      : 2" in captured.out


def test_cli_main_displays_warnings(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Cas 3 : Job completed avec warnings -> avertissements visibles dans la sortie console."""
    local_video = tmp_path / "warn_video.mp4"
    local_video.write_bytes(b"dummy video data")

    fake_result = {
        "job_id": "fake_job_warn",
        "status": "completed",
        "result": {"total_clips": 1, "clips": [{"path": "/out/c1.mp4", "score": 80}]},
    }

    with patch("main.pipeline.run_pipeline", return_value=fake_result), \
         patch("main.job_store.get_job", return_value={"warnings": ["Crop centré de repli utilisé", "Sous-titres omis"]}):

        ret = cli_main([str(local_video)])

    assert ret == 0
    captured = capsys.readouterr()
    assert "Avertissements (2) :" in captured.out
    assert "Crop centré de repli utilisé" in captured.out
    assert "Sous-titres omis" in captured.out


def test_cli_main_handles_pipeline_fatal_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Cas 4 : Erreur fatale de pipeline -> message clair, code retour 1, pas de trace brute."""
    local_video = tmp_path / "fatal_video.mp4"
    local_video.write_bytes(b"dummy video data")

    with patch("main.pipeline.run_pipeline", side_effect=PipelineFatalError("Transcription vide détectée")):
        ret = cli_main([str(local_video)])

    assert ret == 1
    captured = capsys.readouterr()
    assert "[ÉCHEC BLOQUANT DU PIPELINE]" in captured.err
    assert "Transcription vide détectée" in captured.err


def test_cli_main_invalid_source_error(capsys: pytest.CaptureFixture):
    """Cas 5 : Source introuvable et non URL -> message d'erreur clair et code retour 1."""
    ret = cli_main(["/chemin/inexistant/video.mp4"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "n'est ni un fichier local existant, ni une URL valide" in captured.err


def test_cli_main_literal_source_keyword_tolerated(tmp_path: Path):
    """Cas 6 : L'utilisateur a tapé 'source <url>' littéralement -> toléré et exécuté avec succès."""
    out_dir = tmp_path / "out"
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"data")

    with patch("main.downloader.download_youtube", return_value=str(video)), \
         patch("main.downloader.probe_max_quality", return_value={"max_height": 720, "duration": 60}), \
         patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}):

        ret = cli_main(["source", "https://youtu.be/W5sIythWU4Y", "-o", str(out_dir)])

    assert ret == 0


def test_cli_main_no_arguments_prints_help(capsys: pytest.CaptureFixture):
    """Cas 7 : Aucun argument -> affiche l'aide et retourne 1."""
    ret = cli_main([])
    assert ret == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.err


def test_cli_main_clean_source_deletes_downloaded_video(tmp_path: Path):
    """Cas 8 : --clean-source supprime bien le fichier téléchargé après succès."""
    out_dir = tmp_path / "out"
    video = tmp_path / "downloaded_vid.mp4"
    video.write_bytes(b"downloaded content")

    with patch("main.downloader.download_youtube", return_value=str(video)), \
         patch("main.downloader.probe_max_quality", return_value={"max_height": 720, "duration": 60}), \
         patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}):

        ret = cli_main(["https://youtu.be/W5sIythWU4Y", "-o", str(out_dir), "--clean-source"])

    assert ret == 0
    assert not video.exists()


def test_cli_main_purge_jobs_flag(capsys: pytest.CaptureFixture):
    """Cas 9 : --purge-jobs exécute la purge et quitte proprement si pas de source."""
    with patch("main.job_store.purge_old_jobs", return_value=3) as mock_purge:
        ret = cli_main(["--purge-jobs", "5"])

    assert ret == 0
    mock_purge.assert_called_once_with(5)


def test_cli_main_non_verbose_flag(tmp_path: Path):
    """Cas 10 : --non-verbose désactive l'affichage verbeux dans ui."""
    from app.ui import ui
    out_dir = tmp_path / "out"
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"data")

    with patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}):
        ret = cli_main([str(video), "-o", str(out_dir), "--non-verbose"])

    assert ret == 0
    assert ui.is_verbose() is False


def test_cli_main_force_transcribe_flag(tmp_path: Path):
    """Cas 11 : --force-transcribe est correctement transmis à pipeline.run_pipeline."""
    out_dir = tmp_path / "out"
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"data")

    with patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}) as mock_pipe:
        ret = cli_main([str(video), "-o", str(out_dir), "--force-transcribe"])

    assert ret == 0
    mock_pipe.assert_called_once()
    assert mock_pipe.call_args.kwargs.get("force_transcribe") is True


def test_resolve_project_dir_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Vérifie la logique de résolution des répertoires de projet sous projects/."""
    monkeypatch.setenv("PROJECTS_DIR", str(tmp_path / "projects"))

    # Cas 1 : Aucun nom ou vide -> projects/default
    assert resolve_project_dir(None) == str(tmp_path / "projects" / "default")
    assert resolve_project_dir("") == str(tmp_path / "projects" / "default")

    # Cas 2 : Nom relatif -> projects/<nom>
    assert resolve_project_dir("num_4") == str(tmp_path / "projects" / "num_4")
    assert resolve_project_dir("podcast_interview") == str(tmp_path / "projects" / "podcast_interview")

    # Cas 3 : Déjà préfixé par projects/ -> conservé sans duplication
    assert resolve_project_dir("projects/num_4") == "projects/num_4"

    # Cas 4 : Chemin absolu (ex: tests) -> préservé
    abs_path = str(tmp_path / "custom_output")
    assert resolve_project_dir(abs_path) == abs_path


def test_cli_main_project_flag_short_and_long(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Vérifie que -p et --project définissent le dossier sous projects/."""
    monkeypatch.setenv("PROJECTS_DIR", str(tmp_path / "projects"))
    video = tmp_path / "test.mp4"
    video.write_bytes(b"data")

    with patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}) as mock_pipe:
        ret = cli_main([str(video), "-p", "mon_projet"])
        assert ret == 0
        assert mock_pipe.call_args.kwargs.get("out_dir") == str(tmp_path / "projects" / "mon_projet")

    with patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}) as mock_pipe:
        ret = cli_main([str(video), "--project", "podcast_ep1"])
        assert ret == 0
        assert mock_pipe.call_args.kwargs.get("out_dir") == str(tmp_path / "projects" / "podcast_ep1")


def test_cli_main_default_project_when_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Vérifie que sans argument -p ou -o, le projet est résolu en projects/default."""
    monkeypatch.setenv("PROJECTS_DIR", str(tmp_path / "projects"))
    video = tmp_path / "test.mp4"
    video.write_bytes(b"data")

    with patch("main.pipeline.run_pipeline", return_value={"status": "completed", "result": {"clips": []}}) as mock_pipe:
        ret = cli_main([str(video)])
        assert ret == 0
        assert mock_pipe.call_args.kwargs.get("out_dir") == str(tmp_path / "projects" / "default")


def test_cli_main_version_flag(capsys: pytest.CaptureFixture):
    """Vérifie que --version et -V affichent la version du projet et quittent avec 0."""
    from app import __version__

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert f"Darwinx_Repurpose v{__version__}" in (captured.out + captured.err)

    with pytest.raises(SystemExit) as exc_info_short:
        cli_main(["-V"])
    assert exc_info_short.value.code == 0
    captured_short = capsys.readouterr()
    assert f"Darwinx_Repurpose v{__version__}" in (captured_short.out + captured_short.err)

