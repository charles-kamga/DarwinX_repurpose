"""Tests unitaires pour le module app.metadata_studio (Métadonnées SEO & Miniatures 9:16)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import pytest

from app.metadata_studio import (
    build_metadata_prompt,
    create_clip_publishing_kit,
    derive_clip_topic,
    extract_video_frame,
    find_system_font,
    generate_metadata_for_clip,
    generate_thumbnail_image,
    parse_metadata_response,
)


def test_build_metadata_prompt_forbids_chevrons():
    """Vérifie que le prompt exclut formellement les chevrons < >."""
    prompt = build_metadata_prompt(script_text="Script avec <balise>", topic="Topic <test>")
    assert "<" not in prompt
    assert ">" not in prompt
    assert "Sujet : Topic test" in prompt


def test_derive_clip_topic_cleans_oral_intros_and_detects_themes():
    """Vérifie le nettoyage des intros orales et la détection thématique (cas réel Romain)."""
    hook = "Récemment, j'ai reçu un petit message de Romain qui travaille dans la cyber-sécurité."
    reason = "A critical vulnerability allows instant takeover of accounts."
    topic = derive_clip_topic(hook, reason)

    assert "Récemment" not in topic
    assert "petit message" not in topic
    assert "Cybersécurité" in topic or "cyber-sécurité" in topic.lower()
    # Vérification qu'aucun mot n'est coupé en deux
    assert not topic.endswith((" qu", " de", " un"))


def test_derive_clip_topic_twitch_webhook():
    """Vérifie la détection du sujet Twitch / Webhook."""
    hook = "Du coup, ce qui va nous intéresser aujourd'hui, c'est cette faille critique."
    reason = "Webhook challenge handling flaw."
    topic = derive_clip_topic(hook, reason)
    assert "Du coup" not in topic
    assert len(topic) > 5


def test_parse_metadata_response_nominal():
    """Extraction nominale des titres, description et hashtags."""
    raw = (
        "TITRES:\n"
        "- Comment fonctionne l'IA ?\n"
        "- 3 secrets sur l'algorithme !\n"
        "- Tout comprendre en 30s\n\n"
        "DESCRIPTION:\n"
        "Découvrez les coulisses de l'IA en format court. Likez et abonnez-vous !\n\n"
        "HASHTAGS:\n"
        "#Shorts #Tech #AI #Innovation #Viral"
    )
    res = parse_metadata_response(raw, fallback_topic="Intelligence Artificielle")
    assert len(res.titles) == 3
    assert res.titles[0] == "Comment fonctionne l'IA ?"
    assert "Découvrez les coulisses" in res.description
    assert "#Shorts" in res.hashtags
    assert "#TikTok" in res.hashtags
    assert res.is_fallback is False


def test_parse_metadata_response_cleans_chevrons_and_garbage():
    """Nettoyage des chevrons < > et des puces."""
    raw = (
        "TITRES:\n"
        "- <Titre incroyable avec chevron>\n"
        "- < 60 chars catchy title (garbage)\n"
        "- Second bon titre sans balise\n\n"
        "DESCRIPTION:\n"
        "Description avec <lien interdit> et plus de détails.\n\n"
        "HASHTAGS:\n"
        "#Science #Tech"
    )
    res = parse_metadata_response(raw, fallback_topic="Sujet Test")
    for t in res.titles:
        assert "<" not in t
        assert ">" not in t
    assert "<" not in res.description
    assert ">" not in res.description
    assert "#Shorts" in res.hashtags


def test_parse_metadata_response_fallback_avoids_truncated_words():
    """Génération de secours locale réaliste sans coupure de mots et avec tags de niche."""
    res = parse_metadata_response("", fallback_topic="Cybersécurité & Faille")
    assert len(res.titles) >= 3
    for t in res.titles:
        assert not t.endswith((" qu 🤯", " qu", " de", " aujou", " Roma !"))
    assert "#CyberSécurité" in res.hashtags
    assert "#Hacking" in res.hashtags
    assert "#Shorts" in res.hashtags


def test_generate_metadata_for_clip_fail_safe():
    """Vérifie le repli algorithmique local fail-safe si le caller échoue."""
    def buggy_caller(prompt: str) -> str:
        raise RuntimeError("Panne API")

    res = generate_metadata_for_clip(script_text="Texte du clip", topic="Cybersécurité", caller=buggy_caller)
    assert res.is_fallback is True
    assert len(res.titles) >= 3


def test_find_system_font():
    """Vérifie la détection d'une police sans exception."""
    font = find_system_font(size=40)
    assert font is not None


def test_generate_thumbnail_image(tmp_path: Path):
    """Vérifie la composition d'une miniature 1080x1920."""
    out_jpg = tmp_path / "thumb.jpg"
    base_img = tmp_path / "base.jpg"

    Image.new("RGB", (720, 1280), (100, 150, 200)).save(base_img, "JPEG")

    generate_thumbnail_image(
        base_image_path=base_img,
        title="POURQUOI LE CERVEAU CRAQUE ?",
        output_path=out_jpg,
    )

    assert out_jpg.is_file()
    assert out_jpg.stat().st_size > 0

    with Image.open(out_jpg) as img:
        assert img.size == (1080, 1920)
        assert img.format == "JPEG"


def test_create_clip_publishing_kit(tmp_path: Path):
    """Vérifie la création conjointe de la miniature, du .json et du .txt."""
    video = tmp_path / "clip_01.mp4"
    video.write_bytes(b"dummy mp4")

    clip_info = {
        "score": 92,
        "start_time": 10.0,
        "end_time": 35.0,
        "hook_sentence": "Ce secret va changer votre vision du sommeil.",
        "reason": "Explication claire sur le cycle circadien.",
    }

    out_dir = tmp_path / "final_clips"
    out_dir.mkdir()

    with patch("app.metadata_studio.extract_video_frame", return_value=False):
        kit = create_clip_publishing_kit(clip_info, video, out_dir, script_text="Script complet du clip sur le sommeil")

    assert Path(kit["thumbnail_path"]).is_file()
    assert Path(kit["metadata_json_path"]).is_file()
    assert Path(kit["publish_txt_path"]).is_file()

    # Vérification du contenu JSON
    data = json.loads(Path(kit["metadata_json_path"]).read_text(encoding="utf-8"))
    assert data["clip_id"] == "clip_01"
    assert data["score"] == 92
    assert len(data["titles"]) == 3
    assert data["topic"] != ""

    # Vérification du contenu texte
    txt = Path(kit["publish_txt_path"]).read_text(encoding="utf-8")
    assert "SUJET / TOPIC :" in txt
    assert "TITRE RECOMMANDÉ :" in txt
    assert "HASHTAGS :" in txt
    assert "#Shorts" in txt
