"""Configuration globale et seuils produit pour le pipeline de repurposing vidéo.

Conformément à AGENTS.md, les seuils définis ici (durées, retries, fenêtrage)
sont des décisions produit structurantes.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """Charge automatiquement les variables du fichier .env local dans os.environ si présent."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_dotenv_if_present()

# --- Paramètres de Transcription (transcriber) ---
# Forçage du device pour faster-whisper : "auto" (détection), "cpu" ou "cuda"
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "auto")

# --- Paramètres de Fenêtrage du Transcript ---
# Durée cible d'une fenêtre de transcription envoyée au LLM (en secondes)
WINDOW_SECONDS: int = int(os.environ.get("WINDOW_SECONDS", 180))

# Chevauchement cible entre fenêtres consécutives pour capturer les moments à cheval
WINDOW_OVERLAP_SECONDS: int = int(os.environ.get("WINDOW_OVERLAP_SECONDS", 30))

# Seuil en dessous duquel une vidéo est traitée en une seule fenêtre (sans fragmentation)
LONG_VIDEO_THRESHOLD_SECONDS: int = int(os.environ.get("LONG_VIDEO_THRESHOLD_SECONDS", 240))

# --- Contraintes sur les Clips (TikTok / Reels / Shorts) ---
# Durée minimale autorisée pour un clip vertical (secondes)
CLIP_MIN_SECONDS: int = int(os.environ.get("CLIP_MIN_SECONDS", 15))

# Durée maximale autorisée pour un clip vertical (secondes)
CLIP_MAX_SECONDS: int = int(os.environ.get("CLIP_MAX_SECONDS", 60))

# Nombre minimal de clips cibles (plancher qualitatif pour rétention utilisateur)
CLIP_COUNT_MIN: int = int(os.environ.get("CLIP_COUNT_MIN", 4))

# Nombre maximal de clips retournés au final (plafond dur de sélection)
CLIP_COUNT_MAX: int = int(os.environ.get("CLIP_COUNT_MAX", 9))

# --- Paramètres d'appel LLM & Retries ---
# Nombre maximal de tentatives en cas d'erreur transitoire
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", 3))

# Temps de base pour le backoff exponentiel (en secondes)
LLM_BACKOFF_BASE_SECONDS: float = float(os.environ.get("LLM_BACKOFF_BASE_SECONDS", 1.0))

# Facteur multiplicateur pour le backoff exponentiel
LLM_BACKOFF_FACTOR: float = float(os.environ.get("LLM_BACKOFF_FACTOR", 2.0))

# Fournisseur LLM pour le scoring des moments forts :
# "auto" (cascade dynamique Groq -> Gemini -> Mistral -> Anthropic),
# ou fournisseur forcé : "groq", "gemini", "mistral", "anthropic"
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "auto").lower().strip()

# Modèle Claude par défaut pour le scoring de viralité
DEFAULT_ANTHROPIC_MODEL: str = os.environ.get("DEFAULT_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

# Modèle Groq par défaut pour le scoring de viralité
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

# Modèle Google Gemini par défaut pour le scoring de viralité
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# Modèle Mistral AI par défaut pour le scoring de viralité
MISTRAL_MODEL: str = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

# --- Paramètres de Découpe & Cadrage Vertical (clipper) ---
# Timeout pour sous-processus ffmpeg / ffprobe (secondes)
FFMPEG_TIMEOUT_SECONDS: int = int(os.environ.get("FFMPEG_TIMEOUT_SECONDS", 120))

# Facteur de lissage pour suivi de visage (0.0 = pas de lissage, 1.0 = pas d'inertie)
FACE_SMOOTHING_FACTOR: float = float(os.environ.get("FACE_SMOOTHING_FACTOR", 0.15))

# --- Paramètres de Sous-titrage (subtitles) ---
# Nombre maximal de caractères par bloc de sous-titre
SUBTITLE_MAX_CHARS: int = int(os.environ.get("SUBTITLE_MAX_CHARS", 20))

# Durée maximale d'affichage d'un bloc de sous-titre (secondes)
SUBTITLE_MAX_DURATION_SECONDS: float = float(os.environ.get("SUBTITLE_MAX_DURATION_SECONDS", 2.0))

# --- Paramètres de Persistance des Jobs (job_store) ---
# Répertoire de persistance des statuts de jobs
JOBS_DIR: str = os.environ.get("JOBS_DIR", "jobs")

# Timeout pour l'acquisition du verrou inter-processus par job (secondes)
JOB_LOCK_TIMEOUT_SECONDS: float = float(os.environ.get("JOB_LOCK_TIMEOUT_SECONDS", 10.0))

# --- Répertoire Racine des Projets ---
# Répertoire centralisé regroupant tous les projets générés par le pipeline
PROJECTS_DIR: str = os.environ.get("PROJECTS_DIR", "projects")

# --- Paramètres de Téléchargement YouTube (downloader) ---
# Répertoire de stockage temporaire des vidéos sources téléchargées
DOWNLOADS_DIR: str = os.environ.get("DOWNLOADS_DIR", "downloads")

# Nombre maximal de tentatives de téléchargement sur erreur transitoire
DOWNLOAD_MAX_RETRIES: int = int(os.environ.get("DOWNLOAD_MAX_RETRIES", 3))

# Temps de base pour le backoff exponentiel (en secondes)
DOWNLOAD_BACKOFF_BASE_SECONDS: float = float(os.environ.get("DOWNLOAD_BACKOFF_BASE_SECONDS", 1.0))

# Facteur multiplicateur pour le backoff exponentiel
DOWNLOAD_BACKOFF_FACTOR: float = float(os.environ.get("DOWNLOAD_BACKOFF_FACTOR", 2.0))

# Hauteur maximale par défaut pour la sélection de format (1080p)
DOWNLOAD_DEFAULT_MAX_HEIGHT: int = int(os.environ.get("DOWNLOAD_DEFAULT_MAX_HEIGHT", 1080))

# --- Paramètres de Purge & Rétention Disque ---
# Purge automatique des clips intermédiaires après incrustation réussie
CLEAN_INTERMEDIATE_CLIPS: bool = os.environ.get("CLEAN_INTERMEDIATE_CLIPS", "true").lower() in ("1", "true", "yes")

# Durée de rétention par défaut des jobs sur disque (en jours)
JOB_RETENTION_DAYS: int = int(os.environ.get("JOB_RETENTION_DAYS", 7))

# --- Notifications Telegram (Suivi distant temps réel) ---
# Tokens et identifiants de chat (priorité aux variables spécifiques DARWINX_)
TELEGRAM_BOT_TOKEN: Optional[str] = (
    os.environ.get("DARWINX_TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
)
TELEGRAM_CHAT_ID: Optional[str] = (
    os.environ.get("DARWINX_TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
)
TELEGRAM_TIMEOUT_SEC: float = float(os.environ.get("TELEGRAM_TIMEOUT_SEC", 5.0))

