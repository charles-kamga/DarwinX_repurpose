"""Module de génération de métadonnées SEO et de miniatures 9:16 pour la publication.

Conforme au guide video_metadata_and_thumbnail_generation_guide.md :
- Dérive un sujet propre (topic) et injecte le script réel du clip dans un prompt SEO Few-Shot.
- Génère 3 titres viraux (< 70 car., sans coupure de mots), description et hashtags ciblés.
- Compose une miniature 1080x1920 (Pillow + FFmpeg) à fort contraste :
  bandeau sombre semi-transparent + typographie jaune (#FFE600) contour noir 6px.
- Exporte un kit de publication complet par clip (.mp4, .jpg, .json, .txt).
- Dégradation gracieuse : repli algorithmique local fail-safe réaliste sans coupure de mots.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import subprocess
import textwrap
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from app import config

logger = logging.getLogger(__name__)


@dataclass
class MetadataResult:
    """Structure des métadonnées générées pour un clip vidéo."""

    titles: List[str]
    description: str
    hashtags: List[str]
    provider: str
    is_fallback: bool


def find_system_font(size: int = 72) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Détecte une police grasse système disponible sur Linux/macOS/Windows."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc", "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for p in candidates:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def extract_video_frame(video_path: Path, output_image_path: Path, time_sec: float = 2.0) -> bool:
    """Extrait une frame clé de la vidéo via FFmpeg."""
    if not video_path.is_file():
        return False
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(max(0.1, time_sec)), "-i", str(video_path.resolve()), "-vframes", "1", "-q:v", "2", str(output_image_path.resolve())]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return res.returncode == 0 and output_image_path.is_file() and output_image_path.stat().st_size > 0
    except Exception as exc:
        logger.warning("Échec d'extraction de frame FFmpeg : %s", exc)
        return False


def generate_thumbnail_image(base_image_path: Optional[Path], title: str, output_path: Path) -> None:
    """Génère la miniature 9:16 (1080x1920) avec Pillow."""
    width, height = 1080, 1920
    if base_image_path and base_image_path.is_file():
        try:
            img = Image.open(base_image_path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        except Exception:
            img = Image.new("RGBA", (width, height), (20, 25, 45, 255))
    else:
        img = Image.new("RGBA", (width, height), (20, 25, 45, 255))

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([(0, 260), (width, 820)], fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    font = find_system_font(size=72)
    lines = textwrap.wrap(re.sub(r"[<>]", "", title).upper().strip(), width=17) or [title.upper().strip()]

    y_offset = 320
    for line in lines[:4]:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            w, h = 600, 70
        draw.text((max(40, (width - w) // 2), y_offset), line, font=font, fill="#FFE600", stroke_width=6, stroke_fill="#000000")
        y_offset += h + 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=95)


def derive_clip_topic(hook: str, reason: str = "") -> str:
    """Extrait un sujet propre et concis à partir du hook et de la raison du clip."""
    raw = hook.strip()
    pat = r"^(?:(?:récemment,?\s*)?j'ai\s*(?:reçu|vu|entendu)\s*(?:un\s*(?:petit\s*)?message\s*de|que)?|(?:du\s*coup,?\s*)?ce\s*qui\s*(?:va\s*nous\s*intéresser|est\s*intéressant)\s*(?:aujourd'hui,?\s*c'est|c'est)?|en\s*fait,?\s*|alors,?\s*|donc,?\s*|vous\s*savez\s*quoi,?\s*|au\s*fait,?\s*)\s*"
    cleaned = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    theme = ""
    combined = f"{raw} {reason}".lower()
    if any(w in combined for w in ["vulnerability", "faille", "security", "sécurité", "breach", "cyber"]):
        theme = "Faille de Cybersécurité"
    elif any(w in combined for w in ["twitch", "webhook"]):
        theme = "Faille Webhook Twitch"
    elif any(w in combined for w in ["ai", "ia", "intelligence artificielle", "algorithm"]):
        theme = "Intelligence Artificielle"

    if theme:
        return theme

    shortened = textwrap.shorten(cleaned.strip(" ,;.:!?-"), width=45, placeholder="", break_on_hyphens=False) if cleaned else ""
    return shortened.capitalize() if shortened else "Tech & Savoir"


def build_metadata_prompt(script_text: str, topic: str) -> str:
    """Construit un prompt structuré inspiré de Faceless Studio sans chevrons."""
    clean_topic = re.sub(r"[<>]", "", topic).strip() or "Tech & Savoir"
    clean_script = re.sub(r"[<>]", "", script_text).strip()[:400]
    return (
        "Tu es un expert SEO et viralité YouTube Shorts, TikTok et Reels.\n"
        "Génère les métadonnées en français pour un Short captivant basé sur cet extrait.\n"
        "Important : Interdiction formelle d'utiliser des chevrons.\n\n"
        f"Sujet : {clean_topic}\nExtrait vidéo :\n{clean_script}\n\n"
        "Réponds STRICTEMENT selon le format suivant (sans texte d'introduction ni métadonnées parasites) :\n"
        "TITRES:\n"
        f"- {clean_topic} : Ce qu'on ne vous dit pas !\n"
        f"- Comment fonctionne {clean_topic.lower()} ?\n"
        f"- Le secret de {clean_topic.lower()} révélé !\n"
        "DESCRIPTION:\n"
        f"Tout comprendre sur {clean_topic.lower()} en moins d'une minute ! Abonne-toi pour plus de découvertes.\n"
        "HASHTAGS:\n#Shorts #Tech #Innovation #Culture #Decouverte\n"
    )


def parse_metadata_response(text: str, fallback_topic: str, provider: str = "llm") -> MetadataResult:
    """Analyse la réponse LLM avec nettoyage anti-fuite, découpage propre et tags thématiques."""
    clean_topic = re.sub(r"[<>]", "", fallback_topic).strip() or "Tech & Savoir"
    titles: List[str] = []
    desc = ""
    hashtags: List[str] = []

    def is_garbage(s: str) -> bool:
        low = s.lower()
        return any(m in low for m in ["catchy title", "titre accrocheur", "< 60", "60 chars", "here are", "voici les"])

    t_match = re.search(r"(?:TITRES?|TITLES?)\s*:\s*(.*?)(?=(?:DESCRIPTION|HASHTAGS?|\Z))", text, re.IGNORECASE | re.DOTALL)
    if t_match:
        for line in t_match.group(1).splitlines():
            c = re.sub(r"^[*\-\d.\s]+", "", line).strip()
            c = re.sub(r"[<>]", "", c).strip()
            if c and len(c) > 4 and not is_garbage(c) and c not in titles:
                titles.append(textwrap.shorten(c, width=70, placeholder="") if len(c) > 70 else c)

    d_match = re.search(r"DESCRIPTION\s*:\s*(.*?)(?=(?:HASHTAGS?|\Z))", text, re.IGNORECASE | re.DOTALL)
    if d_match:
        d_lines = [re.sub(r"[<>]", "", l.strip()) for l in d_match.group(1).splitlines() if l.strip() and not l.startswith(("*", "-")) and not is_garbage(l)]
        desc = " ".join(d_lines).strip()

    h_match = re.search(r"HASHTAGS?\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if h_match:
        hashtags = [t.strip() for t in re.findall(r"#[A-Za-z0-9_À-ÿ]+", h_match.group(1))]
    if not hashtags:
        hashtags = list(dict.fromkeys(re.findall(r"#[A-Za-z0-9_À-ÿ]+", text)))

    for m in ["#Shorts", "#TikTok", "#Reels"]:
        if not any(h.lower() == m.lower() for h in hashtags):
            hashtags.insert(0, m)

    topic_words = re.findall(r"[A-Za-z0-9À-ÿ]+", clean_topic.title())
    topic_tag = ("#" + "".join(topic_words[:3])) if topic_words else ""
    if topic_tag and len(topic_tag) > 2 and not any(h.lower() == topic_tag.lower() for h in hashtags):
        hashtags.append(topic_tag)

    full_context = f"{clean_topic} {text}".lower()
    niche_keywords = [
        ("sécurit", "#CyberSécurité"), ("cyber", "#CyberSécurité"), ("faille", "#Hacking"),
        ("hack", "#Hacking"), ("pirat", "#Hacking"), ("twitch", "#Twitch"),
        ("webhook", "#Dev"), ("code", "#Code"), ("ia", "#IA"), ("argent", "#Finance"),
    ]
    for kw, tag in niche_keywords:
        if kw in full_context and not any(h.lower() == tag.lower() for h in hashtags):
            hashtags.append(tag)

    default_tags = ["#Tech", "#Viral", "#Culture", "#Découverte"]
    for dt in default_tags:
        if len(hashtags) < 6 and not any(h.lower() == dt.lower() for h in hashtags):
            hashtags.append(dt)

    if not titles:
        titles = [
            textwrap.shorten(f"{clean_topic} : L'explication choc !", width=70, placeholder="", break_on_hyphens=False),
            textwrap.shorten(f"Ce qu'on ne vous dit pas sur {clean_topic.lower()}", width=70, placeholder="", break_on_hyphens=False),
            textwrap.shorten(f"Le secret de {clean_topic.lower()} révélé ! 🤯", width=70, placeholder="", break_on_hyphens=False),
        ]
    elif len(titles) < 3:
        titles.append(textwrap.shorten(f"{clean_topic} en 30 secondes", width=70, placeholder="", break_on_hyphens=False))

    if not desc or len(desc) < 10:
        desc = f"Tout comprendre sur {clean_topic.lower()} en moins d'une minute ! Likez et abonnez-vous."

    return MetadataResult(titles=titles[:3], description=desc[:250], hashtags=hashtags[:8], provider=provider, is_fallback=False)


def generate_metadata_for_clip(
    script_text: str,
    topic: str,
    caller: Optional[Callable[[str], str]] = None,
) -> MetadataResult:
    """Génère les métadonnées SEO pour un clip via LLM avec repli algorithmique local fail-safe."""
    prompt = build_metadata_prompt(script_text=script_text, topic=topic)
    if caller is not None:
        try:
            return parse_metadata_response(caller(prompt), fallback_topic=topic, provider="custom")
        except Exception as exc:
            logger.warning("Échec du caller LLM métadonnées : %s -> repli local", exc)
    else:
        try:
            from app import llm_cascade
            llm_fn = llm_cascade.resolve_llm_caller(config.LLM_PROVIDER)
            return parse_metadata_response(llm_fn(prompt), fallback_topic=topic, provider=config.LLM_PROVIDER)
        except Exception as exc:
            logger.warning("Échec cascade LLM métadonnées : %s -> repli local", exc)

    res = parse_metadata_response("", fallback_topic=topic, provider="local_fallback")
    res.is_fallback = True
    return res


def create_clip_publishing_kit(
    clip_info: Dict[str, Any],
    video_path: Path,
    output_dir: Path,
    script_text: str = "",
    caller: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Génère le kit de diffusion complet pour un clip (.jpg, .json, .txt)."""
    clip_stem = video_path.stem
    hook = clip_info.get("hook_sentence") or clip_stem
    reason = clip_info.get("reason") or ""
    start_time = float(clip_info.get("start_time", 0.0))
    end_time = float(clip_info.get("end_time", 0.0))
    duration = end_time - start_time
    score = clip_info.get("score", "N/A")

    topic = derive_clip_topic(hook=hook, reason=reason)
    content_script = script_text.strip() or hook

    meta = generate_metadata_for_clip(script_text=content_script, topic=topic, caller=caller)

    thumb_path = output_dir / f"{clip_stem}_thumbnail.jpg"
    frame_temp = output_dir / f"{clip_stem}_frame_temp.jpg"
    frame_ok = extract_video_frame(video_path, frame_temp, time_sec=min(2.0, duration / 2))
    generate_thumbnail_image(frame_temp if frame_ok else None, meta.titles[0], thumb_path)
    frame_temp.unlink(missing_ok=True)

    json_path = output_dir / f"{clip_stem}_metadata.json"
    metadata_payload = {
        "clip_id": clip_stem,
        "score": score,
        "duration_sec": round(duration, 2),
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "topic": topic,
        "titles": meta.titles,
        "recommended_title": meta.titles[0],
        "description": meta.description,
        "hashtags": meta.hashtags,
        "thumbnail_file": thumb_path.name,
        "provider": meta.provider,
        "is_fallback": meta.is_fallback,
    }
    json_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_path = output_dir / f"{clip_stem}_publish.txt"
    publish_text = (
        f"SUJET / TOPIC :\n{topic}\n\nTITRE RECOMMANDÉ :\n{meta.titles[0]}\n\n"
        f"AUTRES VARIANTES DE TITRES :\n" + "\n".join(f"- {t}" for t in meta.titles[1:]) + "\n\n"
        f"DESCRIPTION :\n{meta.description}\n\nHASHTAGS :\n{' '.join(meta.hashtags)}\n\n"
        f"INFORMATIONS TECHNIQUES :\nFichier vidéo : {video_path.name}\nMiniature     : {thumb_path.name}\nDurée         : {duration:.1f}s | Score : {score}\n"
    )
    txt_path.write_text(publish_text, encoding="utf-8")

    return {
        **metadata_payload,
        "thumbnail_path": str(thumb_path),
        "metadata_json_path": str(json_path),
        "publish_txt_path": str(txt_path),
    }
