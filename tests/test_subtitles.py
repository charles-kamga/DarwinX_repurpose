"""Tests unitaires pour le module app.subtitles.

100% mocké : zéro dépendance externe, déterministe et rapide.
Couvre tous les cas d'acceptation de SKILL.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import config
import app.clipper
import app.subtitles
from app.clipper import FFmpegProcessError
from app.subtitles import (
    _escape_ass_text,
    _format_ass_time,
    _group_words_into_blocks,
    burn_subtitles,
    generate_ass,
)


def test_format_ass_time():
    """Vérifie le formatage des timestamps au standard ASS H:MM:SS.cs."""
    assert _format_ass_time(0.0) == "0:00:00.00"
    assert _format_ass_time(5.0) == "0:00:05.00"
    assert _format_ass_time(65.456) == "0:01:05.46"
    assert _format_ass_time(3661.12) == "1:01:01.12"


def test_chunking_short_sentence():
    """Cas 1 : Une phrase courte tient dans un seul bloc."""
    words = [
        {"text": "Bonjour", "start": 0.0, "end": 0.5},
        {"text": "ami", "start": 0.6, "end": 1.0},
    ]
    blocks = _group_words_into_blocks(words)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Bonjour ami"
    assert blocks[0]["start"] == 0.0
    assert blocks[0]["end"] == 1.0


def test_chunking_char_limit():
    """Cas 2 : Découpage par limite de caractères (~20), jamais au milieu d'un mot."""
    words = [
        {"text": "Développement", "start": 0.0, "end": 0.8},  # 13 chars
        {"text": "logiciel", "start": 0.9, "end": 1.4},        # 13 + 1 + 8 = 22 > 20 -> nouveau bloc
        {"text": "moderne", "start": 1.5, "end": 1.9},
    ]
    blocks = _group_words_into_blocks(words)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Développement"
    assert blocks[1]["text"] == "logiciel moderne"


def test_chunking_duration_limit():
    """Cas 3 : Découpage par limite de durée (~2.0s) même si peu de caractères."""
    words = [
        {"text": "Un", "start": 0.0, "end": 0.5},
        {"text": "deux", "start": 2.2, "end": 2.5},  # durée totale 2.5 - 0.0 = 2.5 > 2.0s
    ]
    blocks = _group_words_into_blocks(words)
    assert len(blocks) == 2
    assert blocks[0]["text"] == "Un"
    assert blocks[1]["text"] == "deux"


def test_relative_timestamp_recalibration(tmp_path: Path):
    """Cas 4 : Recalage temporel numérique concret.

    Clip source: start=120.0s, end=150.0s.
    Mot 'Bonjour' (125.0s -> 126.5s) doit apparaître à 0:00:05.00 -> 0:00:06.50.
    Les mots 'Avant' (115.0s) et 'Après' (152.0s) doivent être exclus.
    """
    transcript = {
        "segments": [
            {
                "start": 110.0,
                "end": 160.0,
                "text": "Avant Bonjour monde Après",
                "words": [
                    {"w": "Avant", "s": 115.0, "e": 118.0},
                    {"w": "Bonjour", "s": 125.0, "e": 125.8},
                    {"w": "monde", "s": 125.9, "e": 126.8},
                    {"w": "Après", "s": 152.0, "e": 155.0},
                ],
            }
        ]
    }
    out_ass = tmp_path / "test_recalib.ass"
    generate_ass(transcript, clip_start=120.0, clip_end=150.0, out_path=str(out_ass))

    content = out_ass.read_text(encoding="utf-8")
    assert "Avant" not in content
    assert "Après" not in content
    assert "Bonjour monde" in content
    # Doit débuter à 0:00:05.00 (125.0 - 120.0) et se terminer à 0:00:06.80 (126.8 - 120.0)
    assert "Dialogue: 0,0:00:05.00,0:00:06.80,Default,,0,0,0,,Bonjour monde" in content


def test_words_outside_clip_ignored(tmp_path: Path):
    """Cas 5 : Les mots dont l'intervalle ne chevauche pas le clip sont ignorés."""
    transcript = {
        "segments": [
            {
                "words": [
                    {"w": "TropTot", "s": 10.0, "e": 20.0},
                    {"w": "TropTard", "s": 80.0, "e": 90.0},
                ]
            }
        ]
    }
    out_ass = tmp_path / "empty_clip.ass"
    generate_ass(transcript, clip_start=30.0, clip_end=60.0, out_path=str(out_ass))

    content = out_ass.read_text(encoding="utf-8")
    assert "TropTot" not in content
    assert "TropTard" not in content
    assert "Dialogue:" not in content


def test_words_with_none_timestamps(tmp_path: Path):
    """Cas 6 : Dégradation gracieuse sans crash si un mot a un timestamp None."""
    transcript = {
        "segments": [
            {
                "words": [
                    {"w": "Valide1", "s": 10.0, "e": 10.5},
                    {"w": "SansStart", "s": None, "e": 10.8},
                    {"w": "SansEnd", "s": 10.8, "e": None},
                    {"w": "Valide2", "s": 10.9, "e": 11.4},
                ]
            }
        ]
    }
    out_ass = tmp_path / "none_timestamps.ass"
    # Ne doit lever aucun TypeError: float(None)
    generate_ass(transcript, clip_start=5.0, clip_end=20.0, out_path=str(out_ass))

    content = out_ass.read_text(encoding="utf-8")
    assert "SansStart" not in content
    assert "SansEnd" not in content
    assert "Valide1 Valide2" in content


def test_ass_text_escaping(tmp_path: Path):
    """Cas 7 : Échappement des accolades et antislashs, préservation des virgules.

    Vérifie que :
    - '{' et '}' deviennent '(' et ')'
    - '\\' devient '/'
    - La virgule ',' est conservée intacte dans le champ Text
    - La ligne Dialogue comporte ses 9 virgules initiales de champs fixes
    """
    transcript = {
        "segments": [
            {
                "words": [
                    {"w": "Salut,", "s": 0.0, "e": 0.3},
                    {"w": "{expert}", "s": 0.4, "e": 0.7},
                    {"w": "a\\b", "s": 0.8, "e": 1.1},
                ]
            }
        ]
    }
    out_ass = tmp_path / "escaping.ass"
    generate_ass(transcript, clip_start=0.0, clip_end=5.0, out_path=str(out_ass))

    content = out_ass.read_text(encoding="utf-8")
    # Vérification des remplacements de balises
    assert "{expert}" not in content
    assert "(expert)" in content
    assert "a\\b" not in content
    assert "a/b" in content
    # Vérification de la conservation de la virgule
    assert "Salut," in content

    # Vérification de la syntaxe de la ligne Dialogue (9 virgules avant le champ Text)
    dialogue_line = [l for l in content.splitlines() if l.startswith("Dialogue:")][0]
    parts = dialogue_line.split(",", 9)
    assert len(parts) == 10
    assert parts[9] == "Salut, (expert) a/b"


def test_burn_subtitles_nominal(tmp_path: Path):
    """Cas 8 : Incrustation réussie via ffmpeg."""
    video_in = tmp_path / "input.mp4"
    video_in.write_bytes(b"video data")
    ass_in = tmp_path / "sub.ass"
    ass_in.write_text("[Script Info]", encoding="utf-8")
    video_out = tmp_path / "output.mp4"

    with patch("app.subtitles._run_ffmpeg") as mock_ffmpeg:
        result = burn_subtitles(str(video_in), str(ass_in), str(video_out))

    assert result["path"] == str(video_out)
    assert result["warning"] is None
    mock_ffmpeg.assert_called_once()
    called_cmd = mock_ffmpeg.call_args[0][0]
    assert "ffmpeg" in called_cmd[0]
    assert any("-vf" in arg for arg in called_cmd)


def test_burn_subtitles_failure_fallback(tmp_path: Path):
    """Cas 9 : Repli gracieux si ffmpeg échoue à l'incrustation (livré sans sous-titres)."""
    video_in = tmp_path / "input.mp4"
    video_in.write_bytes(b"original raw video")
    ass_in = tmp_path / "sub.ass"
    ass_in.write_text("[Script Info]", encoding="utf-8")
    video_out = tmp_path / "fallback_output.mp4"

    with patch("app.subtitles._run_ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.side_effect = FFmpegProcessError("ffmpeg filter error: font not found")
        result = burn_subtitles(str(video_in), str(ass_in), str(video_out))

    assert result["path"] == str(video_out)
    assert result["warning"] is not None
    assert "Échec de l'incrustation" in result["warning"]
    assert "sans sous-titres" in result["warning"]
    # Vérifie que le fichier a bien été copié
    assert video_out.exists()
    assert video_out.read_bytes() == b"original raw video"


def test_run_ffmpeg_imported_from_clipper():
    """Cas 10 : Confirmation de la réutilisation de _run_ffmpeg depuis app.clipper."""
    assert app.subtitles._run_ffmpeg is app.clipper._run_ffmpeg
