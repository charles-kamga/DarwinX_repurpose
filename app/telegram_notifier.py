"""Module de notifications Telegram autonome et résilient pour Darwinx_repurpose.

Conforme au guide de conception telegram_pipeline_notifications_guide.md :
- Identification claire systématique : préfixe '[Darwinx_repurpose]'.
- Désactivation silencieuse si non configuré (0 latence, 0 crash).
- Confinement strict avec timeout plafonné à 5s (ne bloque jamais le pipeline).
- Repli automatique en texte brut si Telegram rejette le Markdown (HTTP 400).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app import config

logger = logging.getLogger(__name__)

APP_TAG: str = "[Darwinx_repurpose]"


def _get_credentials() -> tuple[Optional[str], Optional[str]]:
    """Récupère les identifiants Telegram (priorité aux clés DARWINX_)."""
    token = (
        os.environ.get("DARWINX_TELEGRAM_BOT_TOKEN")
        or os.environ.get("TELEGRAM_BOT_TOKEN")
        or config.TELEGRAM_BOT_TOKEN
    )
    chat_id = (
        os.environ.get("DARWINX_TELEGRAM_CHAT_ID")
        or os.environ.get("TELEGRAM_CHAT_ID")
        or config.TELEGRAM_CHAT_ID
    )

    clean_token = token.strip() if token else None
    clean_chat_id = chat_id.strip() if chat_id else None
    return clean_token, clean_chat_id


def send_telegram_message(
    text: str,
    parse_mode: Optional[str] = "Markdown",
    timeout: Optional[float] = None,
) -> bool:
    """Envoie un message Telegram avec gestion automatique d'erreurs et repli texte brut.

    Args:
        text: Corps du message.
        parse_mode: 'Markdown', 'HTML' ou None (texte brut).
        timeout: Timeout optionnel en secondes (défaut : config.TELEGRAM_TIMEOUT_SEC).

    Returns:
        True si envoyé avec succès, False sinon (toujours non-bloquant).
    """
    token, chat_id = _get_credentials()

    # 1. Règle d'or n°1 : Désactivation silencieuse si non configuré
    if not token or not chat_id:
        return False

    effective_timeout = timeout if timeout is not None else config.TELEGRAM_TIMEOUT_SEC
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        with httpx.Client(timeout=effective_timeout) as client:
            # 2. Règle d'or n°2 : Première tentative avec formatage Markdown
            response = client.post(url, json=payload)
            if response.status_code == 200:
                return True

            # 3. Règle d'or n°3 : Repli texte brut automatique si rejet Markdown (HTTP 400)
            if response.status_code == 400 and parse_mode:
                logger.debug(
                    "Rejet Markdown Telegram (HTTP 400), tentative de repli en texte brut pour %s",
                    APP_TAG,
                )
                payload.pop("parse_mode", None)
                retry_response = client.post(url, json=payload)
                if retry_response.status_code == 200:
                    return True

            logger.warning(
                "Échec d'envoi Telegram %s (HTTP %d) : %s",
                APP_TAG,
                response.status_code,
                response.text,
            )
            return False

    except (httpx.TimeoutException, httpx.RequestError) as net_err:
        # 4. Règle d'or n°4 : Confinement strict — une panne réseau ne bloque JAMAIS le pipeline
        logger.warning("Notification Telegram %s ignorée (erreur réseau / timeout) : %s", APP_TAG, net_err)
        return False
    except Exception as exc:
        logger.warning("Erreur inattendue lors de l'envoi Telegram %s : %s", APP_TAG, exc)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Alertes Métier Spécialisées pour Darwinx_repurpose
# ──────────────────────────────────────────────────────────────────────────────


def notify_job_start(
    job_id: str,
    source: str,
    whisper_device: str = "auto",
    llm_provider: str = "auto",
) -> bool:
    """Notifie le démarrage du pipeline sur une vidéo avec son identifiant et sa config."""
    src_name = Path(source).name if "://" not in source else source
    lines = [
        f"🚀 *{APP_TAG} Démarrage du pipeline*\n",
        f"🆔 *Job* : `{job_id}`",
        f"📌 *Source* : `{src_name}`",
        f"⚡ *Accélération* : Whisper `{whisper_device.upper()}` | LLM `{llm_provider.upper()}`",
        "📐 *Format cible* : `9:16 Vertical (Shorts / TikTok / Reels)`",
    ]
    return send_telegram_message("\n".join(lines))


def notify_step_done(
    job_id: str,
    step_name: str,
    metrics: Dict[str, Any],
) -> bool:
    """Notifie la complétion d'une étape intermédiaire du pipeline avec ses métriques."""
    lines = [
        f"📦 *{APP_TAG} Étape terminée : {step_name}*\n",
        f"🆔 *Job* : `{job_id}`",
        "\n*Métriques :*",
    ]
    for key, val in metrics.items():
        lines.append(f"• {key} : `{val}`")
    return send_telegram_message("\n".join(lines))


def notify_job_completed(
    job_id: str,
    duration_sec: float,
    clips: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
) -> bool:
    """Notifie le succès final avec la liste des clips générés et les avertissements."""
    warn_list = warnings or []
    warn_badge = f"⚠️ `{len(warn_list)} avertissement(s)`" if warn_list else "✓ `0 avertissement (100% nominal)`"

    lines = [
        f"🎉 *{APP_TAG} Pipeline terminé avec succès !*\n",
        f"🆔 *Job* : `{job_id}`",
        f"⏱️ *Durée totale* : `{duration_sec:.1f}s`",
        f"🎬 *Clips générés* : `{len(clips)} clip(s) 9:16`",
        f"🛡️ *Qualité* : {warn_badge}",
    ]

    if clips:
        lines.append("\n*Clips livrés :*")
        for idx, clip in enumerate(clips, 1):
            score = clip.get("score", "N/A")
            s_time = float(clip.get("start_time", 0.0))
            e_time = float(clip.get("end_time", 0.0))
            dur = e_time - s_time
            c_name = Path(clip.get("final_path") or clip.get("path", f"clip_{idx:02d}")).name
            lines.append(f"  • Clip {idx} : `{c_name}` ({dur:.1f}s · Score {score})")

    if warn_list:
        lines.append("\n*Détails des avertissements :*")
        for w in warn_list:
            lines.append(f"  • {w}")

    return send_telegram_message("\n".join(lines))


def notify_job_error(
    job_id: str,
    step_name: str,
    error_message: str,
) -> bool:
    """Notifie un échec bloquant du pipeline avec identification de l'étape coupable."""
    lines = [
        f"❌ *{APP_TAG} Échec bloquant du pipeline*\n",
        f"🆔 *Job* : `{job_id}`",
        f"🛑 *Étape critique* : `{step_name}`",
        f"⚠️ *Raison* : `{error_message}`",
    ]
    return send_telegram_message("\n".join(lines))
