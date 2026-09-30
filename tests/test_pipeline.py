"""Tests unitaires pour l'orchestrateur du pipeline (app.pipeline) et app.job_store.

100% mocké : zéro dépendance externe, déterministe et rapide.
Couvre tous les cas d'acceptation de SKILL.md et AGENTS.md.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app import config, job_store
from app.highlights import NoHighlightsFoundError
from app.pipeline import (
    PipelineError,
    PipelineFatalError,
    PipelineUnexpectedError,
    run_pipeline,
)
from app.transcriber import EmptyTranscriptionError, TranscriberMediaError


@pytest.fixture(autouse=True)
def configure_test_jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isole le répertoire de stockage des jobs dans un dossier temporaire pytest."""
    test_jobs_dir = tmp_path / "test_jobs"
    test_jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "JOBS_DIR", str(test_jobs_dir))
    return test_jobs_dir


@pytest.fixture
def sample_transcript():
    """Exemple de transcript complet."""
    return {
        "duration": 180.0,
        "segments": [
            {
                "start": 10.0,
                "end": 35.0,
                "text": "Extrait numéro un.",
                "words": [{"w": "Extrait", "s": 10.0, "e": 12.0}],
            },
            {
                "start": 50.0,
                "end": 80.0,
                "text": "Extrait numéro deux.",
                "words": [{"w": "Extrait", "s": 50.0, "e": 52.0}],
            },
        ],
    }


@pytest.fixture
def sample_highlights():
    """Exemple de highlights extraits."""
    return [
        {
            "start_time": 10.0,
            "end_time": 35.0,
            "score": 92,
            "hook_sentence": "Hook 1",
            "reason": "Top",
        },
        {
            "start_time": 50.0,
            "end_time": 80.0,
            "score": 85,
            "hook_sentence": "Hook 2",
            "reason": "Cool",
        },
    ]


# =============================================================================
# Tests de l'Orchestrateur (app.pipeline)
# =============================================================================


def test_pipeline_nominal(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 1 : Déroulement nominal complet des 4 modules enchaînés avec succès."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_nominal"
    job_id = job_store.create_job(source)

    mock_clips = [
        {"start_time": 10.0, "end_time": 35.0, "path": str(out_dir / "raw_clips/clip_01.mp4"), "warning": None},
        {"start_time": 50.0, "end_time": 80.0, "path": str(out_dir / "raw_clips/clip_02.mp4"), "warning": None},
    ]

    with patch("app.transcriber.transcribe", return_value=sample_transcript) as mock_transcribe, \
         patch("app.highlights.get_highlights", return_value=sample_highlights) as mock_highlights, \
         patch("app.clipper.make_clips", return_value=mock_clips) as mock_clipper, \
         patch("app.subtitles.generate_ass") as mock_gen_ass, \
         patch("app.subtitles.burn_subtitles") as mock_burn_sub:

        mock_burn_sub.side_effect = lambda video_path, ass_path, out_path: {
            "path": out_path,
            "warning": None,
        }

        result = run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    assert result["status"] == "completed"
    assert result["job_id"] == job_id
    assert result["result"]["total_clips"] == 2

    # Vérification de l'état persisté dans le job_store
    persisted_job = job_store.get_job(job_id)
    assert persisted_job is not None
    assert persisted_job["status"] == "completed"
    assert persisted_job["steps_done"] == ["transcription", "scoring", "clipper", "subtitles"]
    assert persisted_job["current_step"] is None
    assert len(persisted_job["warnings"]) == 0

    mock_transcribe.assert_called_once_with(source)
    mock_highlights.assert_called_once_with(sample_transcript)
    mock_clipper.assert_called_once()
    assert mock_gen_ass.call_count == 2
    assert mock_burn_sub.call_count == 2

    manifest_p = out_dir / "project_manifest.json"
    assert manifest_p.is_file()


def test_pipeline_empty_transcription_fatal(tmp_path: Path):
    """Cas 2 (Bloquant 1/3) : Transcription vide ou média illisible -> job failed immédiat."""
    source = "/media/silent.mp4"
    out_dir = tmp_path / "output_err1"
    job_id = job_store.create_job(source)

    with patch("app.transcriber.transcribe", side_effect=EmptyTranscriptionError("Aucune parole")), \
         patch("app.highlights.get_highlights") as mock_highlights:

        with pytest.raises(PipelineFatalError, match="Échec bloquant de transcription"):
            run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    # Le scoring ne doit jamais être appelé
    mock_highlights.assert_not_called()

    # Le statut persisté doit être failed
    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "failed"
    assert "Échec bloquant de transcription" in persisted["error"]
    assert "transcription" not in persisted["steps_done"]


def test_pipeline_no_highlights_fatal(tmp_path: Path, sample_transcript):
    """Cas 3 (Bloquant 2/3) : Zéro moment fort extrait -> job failed immédiat."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_err2"
    job_id = job_store.create_job(source)

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", side_effect=NoHighlightsFoundError("Zéro highlight")), \
         patch("app.clipper.make_clips") as mock_clipper:

        with pytest.raises(PipelineFatalError, match="Échec bloquant de scoring"):
            run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    mock_clipper.assert_not_called()
    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "failed"
    assert "Échec bloquant de scoring" in persisted["error"]
    assert "transcription" in persisted["steps_done"]
    assert "scoring" not in persisted["steps_done"]


def test_pipeline_all_clips_failed_fatal(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 4 (Bloquant 3/3) : Tous les clips échouent au montage -> job failed."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_err3"
    job_id = job_store.create_job(source)

    mock_clips_failed = [
        {"start_time": 10.0, "end_time": 35.0, "path": None, "warning": "ffmpeg crash 1"},
        {"start_time": 50.0, "end_time": 80.0, "path": None, "warning": "ffmpeg crash 2"},
    ]

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", return_value=sample_highlights), \
         patch("app.clipper.make_clips", return_value=mock_clips_failed), \
         patch("app.subtitles.generate_ass") as mock_gen_ass:

        with pytest.raises(PipelineFatalError, match="tous les clips ont échoué"):
            run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    mock_gen_ass.assert_not_called()
    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "failed"
    assert "tous les clips ont échoué au montage" in persisted["error"]


def test_pipeline_partial_clips_failure_graceful(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 5 : Échec partiel (1 clip échoue, 2 réussissent) -> dégradation gracieuse, job completed."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_partial"
    job_id = job_store.create_job(source)

    mock_clips = [
        {"start_time": 10.0, "end_time": 35.0, "path": str(out_dir / "raw/c1.mp4"), "warning": None},
        {"start_time": 40.0, "end_time": 60.0, "path": None, "warning": "Code d'erreur ffmpeg"},
        {"start_time": 70.0, "end_time": 90.0, "path": str(out_dir / "raw/c3.mp4"), "warning": None},
    ]

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", return_value=sample_highlights), \
         patch("app.clipper.make_clips", return_value=mock_clips), \
         patch("app.subtitles.generate_ass"), \
         patch("app.subtitles.burn_subtitles") as mock_burn:

        mock_burn.side_effect = lambda video_path, ass_path, out_path: {
            "path": out_path,
            "warning": None,
        }

        result = run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    assert result["status"] == "completed"
    assert result["result"]["total_clips"] == 2  # Seulement les 2 valides

    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "completed"
    assert len(persisted["warnings"]) == 1
    assert "Clip 2 échoué" in persisted["warnings"][0]


def test_pipeline_subtitles_warning_graceful(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 6 : Avertissement de sous-titres (clip livré sans sous-titres) -> warning consigné, job completed."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_sub_warn"
    job_id = job_store.create_job(source)

    mock_clips = [
        {"start_time": 10.0, "end_time": 35.0, "path": str(out_dir / "raw/c1.mp4"), "warning": None},
    ]

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", return_value=sample_highlights), \
         patch("app.clipper.make_clips", return_value=mock_clips), \
         patch("app.subtitles.generate_ass"), \
         patch("app.subtitles.burn_subtitles") as mock_burn:

        mock_burn.return_value = {
            "path": str(out_dir / "final_clips/c1.mp4"),
            "warning": "Police manquante : clip livré sans sous-titres",
        }

        result = run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    assert result["status"] == "completed"
    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "completed"
    assert len(persisted["warnings"]) == 1
    assert "Police manquante" in persisted["warnings"][0]


def test_pipeline_unexpected_exception_last_resort(tmp_path: Path, sample_transcript):
    """Cas 7 : Exception totalement imprévue -> filet de dernier recours, job marqué failed."""
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_unexpected"
    job_id = job_store.create_job(source)

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", side_effect=TypeError("Unexpected None inside scoring")):

        with pytest.raises(PipelineUnexpectedError, match="Erreur critique imprévue"):
            run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))

    persisted = job_store.get_job(job_id)
    assert persisted["status"] == "failed"
    assert "TypeError" in persisted["error"]


def test_pipeline_last_resort_survives_job_store_failure(tmp_path: Path, sample_transcript):
    """Cas 8 : Double défaillance (exception imprévue + crash I/O job_store.mark_failed).

    Vérifie que la capture de dernier recours ne plante pas sur sa propre tentative
    d'écriture de statut et lève proprement PipelineUnexpectedError.
    """
    source = "/media/video.mp4"
    out_dir = tmp_path / "output_double_fail"
    job_id = job_store.create_job(source)

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", side_effect=ValueError("Bug inopiné")), \
         patch("app.job_store.mark_failed", side_effect=OSError("Disk full: ENOSPC")):

        with pytest.raises(PipelineUnexpectedError, match="Erreur critique imprévue"):
            run_pipeline(job_id=job_id, source_path=source, out_dir=str(out_dir))


# =============================================================================
# Tests du Magasin de Jobs (app.job_store)
# =============================================================================


def test_job_store_atomic_persistence():
    """Cas 9 : Persistance atomique via fichier temporaire et os.replace."""
    with patch("os.replace", wraps=os.replace) as mock_replace:
        job_id = job_store.create_job("/fake/path.mp4")
        assert mock_replace.called

    data = job_store.get_job(job_id)
    assert data is not None
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    assert data["source_path"] == "/fake/path.mp4"


def test_job_store_state_transitions():
    """Cas 10 : Transitions d'état complètes et accumulation de warnings."""
    job_id = job_store.create_job("/fake/path.mp4")

    job_store.set_step(job_id, "transcription")
    d1 = job_store.get_job(job_id)
    assert d1["status"] == "processing"
    assert d1["current_step"] == "transcription"

    job_store.mark_step_done(job_id, "transcription")
    d2 = job_store.get_job(job_id)
    assert "transcription" in d2["steps_done"]
    assert d2["current_step"] is None

    job_store.add_warning(job_id, "Avertissement cadrage")
    job_store.add_warning(job_id, "Avertissement sous-titres")
    d3 = job_store.get_job(job_id)
    assert len(d3["warnings"]) == 2

    job_store.mark_completed(job_id, {"clips_count": 3})
    d4 = job_store.get_job(job_id)
    assert d4["status"] == "completed"
    assert d4["result"]["clips_count"] == 3

    # Test statut failed
    job_id2 = job_store.create_job("/other/path.mp4")
    job_store.mark_failed(job_id2, "Erreur fatale de test")
    d5 = job_store.get_job(job_id2)
    assert d5["status"] == "failed"
    assert d5["error"] == "Erreur fatale de test"


def test_job_store_concurrent_updates_no_lost_update():
    """Cas 11 : Appels concurrents multi-threads sur le même job_id sans perte de mise à jour."""
    job_id = job_store.create_job("/media/source.mp4")
    num_threads = 10
    warnings_per_thread = 5

    def worker(thread_idx: int):
        for w_idx in range(warnings_per_thread):
            job_store.add_warning(job_id, f"Thread {thread_idx} avertissement {w_idx}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_job = job_store.get_job(job_id)
    assert final_job is not None
    # 10 threads * 5 avertissements = 50 avertissements sans aucune perte
    assert len(final_job["warnings"]) == num_threads * warnings_per_thread
    assert len(set(final_job["warnings"])) == num_threads * warnings_per_thread


def test_job_store_independent_locks_different_jobs():
    """Cas 12 : Le verrouillage d'un job_id ne bloque pas les opérations sur un autre job_id."""
    job_1 = job_store.create_job("/media/source1.mp4")
    job_2 = job_store.create_job("/media/source2.mp4")

    barrier = threading.Barrier(2)
    thread1_holding = threading.Event()

    def hold_job_1_lock():
        with job_store._job_lock(job_1):
            thread1_holding.set()
            barrier.wait(timeout=2.0)
            time.sleep(0.4)

    t1 = threading.Thread(target=hold_job_1_lock)
    t1.start()

    thread1_holding.wait(timeout=2.0)

    # Pendant que job_1 est verrouillé par t1, l'opération sur job_2 doit s'exécuter immédiatement
    start_time = time.time()
    job_store.add_warning(job_2, "Warning rapide sur job 2")
    elapsed = time.time() - start_time

    # Libère t1
    barrier.wait(timeout=2.0)
    t1.join()

    # L'opération sur job_2 n'a pas attendu les 0.4s de t1
    assert elapsed < 0.25
    job_2_data = job_store.get_job(job_2)
    assert "Warning rapide sur job 2" in job_2_data["warnings"]


def test_job_store_lock_timeout_raises_job_store_error(monkeypatch: pytest.MonkeyPatch):
    """Cas 13 : Dépassement du timeout de verrouillage lève explicitement JobStoreError."""
    # Timeout très court pour un test rapide et déterministe
    monkeypatch.setattr(config, "JOB_LOCK_TIMEOUT_SECONDS", 0.1)

    job_id = job_store.create_job("/media/source.mp4")
    lock_acquired_by_t = threading.Event()
    release_event = threading.Event()

    def hold_lock():
        with job_store._job_lock(job_id):
            lock_acquired_by_t.set()
            release_event.wait(timeout=2.0)

    t = threading.Thread(target=hold_lock)
    t.start()
    lock_acquired_by_t.wait(timeout=1.0)

    try:
        with pytest.raises(job_store.JobStoreError, match="Timeout .* lors de l'acquisition du verrou"):
            job_store.add_warning(job_id, "Ne doit pas passer")
    finally:
        release_event.set()
        t.join()


def test_pipeline_cleans_intermediate_raw_clips(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 14 : Vérifie que les clips bruts intermédiaires sont purgés après la finalisation."""
    job_id = job_store.create_job("/media/source.mp4")
    out_dir = tmp_path / "clean_test"

    raw_dir = out_dir / "raw_clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "clip_01.mp4"
    raw_file.write_bytes(b"dummy raw content")

    def mock_make_clips(source, highlights, out_d):
        return [{"start_time": 10.0, "end_time": 30.0, "score": 90, "path": str(raw_file), "warning": None}]

    def mock_burn(video_path, ass_path, out_path):
        Path(out_path).write_bytes(b"final clip content")
        return {"path": out_path, "warning": None}

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", return_value=sample_highlights), \
         patch("app.clipper.make_clips", side_effect=mock_make_clips), \
         patch("app.subtitles.generate_ass"), \
         patch("app.subtitles.burn_subtitles", side_effect=mock_burn):

        res = run_pipeline(job_id, "/media/source.mp4", str(out_dir))

    assert res["status"] == "completed"
    # Le fichier intermédiaire raw_clips doit avoir été supprimé
    assert not raw_file.exists()
    # Le fichier final final_clips doit exister
    final_file = out_dir / "final_clips" / "clip_01.mp4"
    assert final_file.exists()


def test_pipeline_generate_ass_failure_graceful(tmp_path: Path, sample_transcript, sample_highlights):
    """Cas 15 (Dégradation gracieuse) : Si generate_ass échoue, le clip est quand même livré brut."""
    job_id = job_store.create_job("/media/source.mp4")
    out_dir = tmp_path / "ass_fail_test"

    raw_dir = out_dir / "raw_clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "clip_01.mp4"
    raw_file.write_bytes(b"raw clip video content")

    def mock_make_clips(source, highlights, out_d):
        return [{"start_time": 10.0, "end_time": 30.0, "score": 85, "path": str(raw_file), "warning": None}]

    with patch("app.transcriber.transcribe", return_value=sample_transcript), \
         patch("app.highlights.get_highlights", return_value=sample_highlights), \
         patch("app.clipper.make_clips", side_effect=mock_make_clips), \
         patch("app.subtitles.generate_ass", side_effect=RuntimeError("Erreur fatale ASS")):

        res = run_pipeline(job_id, "/media/source.mp4", str(out_dir))

    assert res["status"] == "completed"
    job_data = job_store.get_job(job_id)
    assert any("Sous-titres clip 1 ignorés" in w for w in job_data["warnings"])
    final_file = out_dir / "final_clips" / "clip_01.mp4"
    assert final_file.exists()
    assert final_file.read_bytes() == b"raw clip video content"


def test_job_store_purge_old_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Cas 16 : purge_old_jobs supprime les jobs et locks plus anciens que max_age_days."""
    test_jobs_dir = tmp_path / "jobs"
    test_jobs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "JOBS_DIR", str(test_jobs_dir))

    # Job récent (maintenant)
    recent_job = test_jobs_dir / "recent_job.json"
    recent_job.write_text("{}", encoding="utf-8")

    # Vieux job (il y a 10 jours)
    old_job = test_jobs_dir / "old_job.json"
    old_job.write_text("{}", encoding="utf-8")
    old_lock = test_jobs_dir / "old_job.lock"
    old_lock.write_text("", encoding="utf-8")

    ten_days_ago = time.time() - (10 * 86400)
    os.utime(str(old_job), (ten_days_ago, ten_days_ago))
    os.utime(str(old_lock), (ten_days_ago, ten_days_ago))

    deleted = job_store.purge_old_jobs(max_age_days=7)
    assert deleted == 2
    assert not old_job.exists()
    assert not old_lock.exists()
    assert recent_job.exists()


