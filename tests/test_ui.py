"""Tests unitaires pour le module app.ui (TUI et verbosité)."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from app.ui import UIManager, ui


def test_ui_verbose_toggle():
    """Vérifie l'activation et la désactivation de la verbosité."""
    mgr = UIManager(verbose=True)
    assert mgr.is_verbose() is True

    mgr.set_verbose(False)
    assert mgr.is_verbose() is False

    mgr.set_verbose(True)
    assert mgr.is_verbose() is True


def test_ui_non_verbose_silences_info_and_steps():
    """En mode non-verbose, les étapes et micro-métriques ne sont pas imprimées."""
    mgr = UIManager(verbose=False)

    with patch("app.ui.console.print") as mock_print:
        mgr.print_step_header(1, 4, "Transcription", "detail")
        mgr.print_info("Micro-métrique")
        mgr.print_success("Succès")
        mgr.print_highlights_table([{"score": 90}])
        mgr.print_clip_action(1, 2, "Cadrage", "clip_01")

        # Aucune de ces méthodes ne doit appeler console.print en non-verbose
        mock_print.assert_not_called()


def test_ui_warning_and_error_always_print():
    """Les avertissements et erreurs sont affichés même en mode non-verbose."""
    mgr = UIManager(verbose=False)

    with patch("app.ui.console.print") as mock_print:
        mgr.print_warning("Attention dégradation")
        mgr.print_error("Erreur bloquante")

        assert mock_print.call_count == 2


def test_ui_banner_and_report_rendering():
    """Vérifie le rendu du bandeau d'accueil et du rapport final en mode verbeux."""
    mgr = UIManager(verbose=True)

    with patch("app.ui.console.print") as mock_print:
        mgr.print_banner("https://youtu.be/xyz", "output/dir", "cuda", "groq")
        assert mock_print.call_count >= 1

        mock_print.reset_mock()
        clips = [
            {"score": 95, "start_time": 10.0, "end_time": 30.0, "path": "output/dir/clip_01.mp4"}
        ]
        mgr.print_final_report("job-123", 12.5, clips, ["Warning test"], "output/dir")
        assert mock_print.call_count >= 1
