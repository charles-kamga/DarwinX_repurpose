"""Module de cascade et fallback multi-fournisseurs LLM.

Implémente le patron de cascade résiliente pour le scoring des moments forts :
Groq (priorité 1) -> Google Gemini (priorité 2) -> Mistral AI (priorité 3) -> Anthropic (secours).

Fournit :
- Mode 'auto' : bascule transparente sur rate-limit (429) ou indisponibilité réseau.
- Mode forcé : échec strict sans repli silencieux si un fournisseur précis est exigé.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import httpx

from app import config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception générique pour les appels LLM."""


class PermanentLLMError(LLMError):
    """Erreur permanente (ne doit pas être retentée, ex. clé manquante ou invalide)."""


class TransientLLMError(LLMError):
    """Erreur transitoire (rate-limit 429, timeout, panne réseau)."""


def _call_groq(prompt: str) -> str:
    """Appel au fournisseur Groq via son SDK officiel."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise PermanentLLMError("GROQ_API_KEY non définie dans l'environnement")

    try:
        from groq import Groq
    except ImportError as exc:
        raise PermanentLLMError("Le package 'groq' n'est pas installé dans l'environnement.") from exc

    try:
        client = Groq(api_key=api_key)
        models_to_try = [config.GROQ_MODEL]
        for candidate in ("qwen/qwen3.8-27b", "groq/compound-mini", "llama-3.3-70b-versatile"):
            if candidate not in models_to_try:
                models_to_try.append(candidate)

        last_err = None
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                content = response.choices[0].message.content or ""
                if content.strip():
                    return content
            except Exception as m_exc:
                last_err = m_exc
                err_msg = str(m_exc).lower()
                if "model_not_found" in err_msg or "404" in err_msg:
                    logger.warning("Modèle Groq '%s' non disponible -> bascule modèle...", model_name)
                    continue
                if "rate limit" in err_msg or "429" in err_msg or "timeout" in err_msg:
                    raise TransientLLMError(f"Erreur transitoire Groq : {m_exc}") from m_exc
                if "authentication" in err_msg or "401" in err_msg or "invalid api key" in err_msg:
                    raise PermanentLLMError(f"Erreur d'authentification Groq : {m_exc}") from m_exc
                raise

        if last_err:
            raise last_err
        raise TransientLLMError("Réponse vide reçue de l'API Groq.")
    except (PermanentLLMError, TransientLLMError):
        raise
    except Exception as exc:
        err_msg = str(exc).lower()
        if "rate limit" in err_msg or "429" in err_msg or "timeout" in err_msg:
            raise TransientLLMError(f"Erreur transitoire Groq : {exc}") from exc
        if "authentication" in err_msg or "401" in err_msg or "invalid api key" in err_msg:
            raise PermanentLLMError(f"Erreur d'authentification Groq : {exc}") from exc
        raise TransientLLMError(f"Échec appel Groq : {exc}") from exc


def _call_gemini(prompt: str) -> str:
    """Appel au fournisseur Google Gemini via requête HTTP REST directe (httpx)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PermanentLLMError("Clé d'API Gemini manquante (variable d'environnement GEMINI_API_KEY).")

    models_to_try = [config.GEMINI_MODEL]
    for candidate in ("gemini-3.6-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
        if candidate not in models_to_try:
            models_to_try.append(candidate)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7},
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            last_resp_err = None
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                resp = client.post(url, json=payload)
                if resp.status_code == 404:
                    logger.warning("Modèle Gemini '%s' non disponible (404) -> essai suivant...", model_name)
                    continue
                if resp.status_code == 429:
                    raise TransientLLMError("Rate-limit Google Gemini (HTTP 429) atteint.")
                if resp.status_code in (401, 403):
                    raise PermanentLLMError(f"Clé Gemini invalide ou révoquée (HTTP {resp.status_code}).")
                if resp.status_code >= 500:
                    raise TransientLLMError(f"Erreur serveur Google Gemini (HTTP {resp.status_code}).")

                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise TransientLLMError("Aucun candidat retourné par Gemini.")
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not text.strip():
                    raise TransientLLMError("Réponse textuelle vide reçue de Gemini.")
                return text

            raise TransientLLMError("Tous les modèles Gemini candidats ont renvoyé 404.")
    except (PermanentLLMError, TransientLLMError):
        raise
    except httpx.TimeoutException as exc:
        raise TransientLLMError(f"Timeout lors de l'appel Gemini : {exc}") from exc
    except httpx.RequestError as exc:
        raise TransientLLMError(f"Erreur réseau Gemini : {exc}") from exc
    except Exception as exc:
        raise TransientLLMError(f"Erreur inattendue Gemini : {exc}") from exc


def _call_mistral(prompt: str) -> str:
    """Appel au fournisseur Mistral AI via endpoint compatible OpenAI (httpx)."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise PermanentLLMError("Clé d'API Mistral manquante (variable d'environnement MISTRAL_API_KEY).")

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 429:
                raise TransientLLMError("Rate-limit Mistral AI (HTTP 429) atteint.")
            if resp.status_code in (401, 403):
                raise PermanentLLMError(f"Clé Mistral invalide ou révoquée (HTTP {resp.status_code}).")
            if resp.status_code >= 500:
                raise TransientLLMError(f"Erreur serveur Mistral AI (HTTP {resp.status_code}).")

            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise TransientLLMError("Aucun choix retourné par Mistral.")
            text = choices[0].get("message", {}).get("content", "")
            if not text.strip():
                raise TransientLLMError("Réponse textuelle vide reçue de Mistral.")
            return text
    except (PermanentLLMError, TransientLLMError):
        raise
    except httpx.TimeoutException as exc:
        raise TransientLLMError(f"Timeout lors de l'appel Mistral : {exc}") from exc
    except httpx.RequestError as exc:
        raise TransientLLMError(f"Erreur réseau Mistral : {exc}") from exc
    except Exception as exc:
        raise TransientLLMError(f"Erreur inattendue Mistral : {exc}") from exc


def _call_anthropic(prompt: str) -> str:
    """Appel au fournisseur Anthropic Claude via son SDK officiel."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise PermanentLLMError("ANTHROPIC_API_KEY non définie dans l'environnement")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise PermanentLLMError("Le package 'anthropic' n'est pas installé dans l'environnement.") from exc

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=config.DEFAULT_ANTHROPIC_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        first_block = message.content[0]
        return getattr(first_block, "text", "")
    except Exception as exc:
        err_msg = str(exc).lower()
        if "rate limit" in err_msg or "429" in err_msg or "overloaded" in err_msg or "529" in err_msg:
            raise TransientLLMError(f"Erreur transitoire Anthropic : {exc}") from exc
        if "authentication" in err_msg or "401" in err_msg or "credit balance" in err_msg:
            raise PermanentLLMError(f"Erreur d'authentification ou solde Anthropic : {exc}") from exc
        raise TransientLLMError(f"Échec appel Anthropic : {exc}") from exc


def resolve_llm_caller(provider: Optional[str] = None) -> Callable[[str], str]:
    """Résout le callable LLM approprié selon la stratégie 'auto' (cascade) ou mode forcé.
    
    Args:
        provider: Nom du fournisseur ou 'auto'. Par défaut config.LLM_PROVIDER.
        
    Returns:
        Callable prenant un prompt string et retournant le texte string généré.
    """
    prov = (provider or config.LLM_PROVIDER or "auto").lower().strip()

    # 1. Mode Fournisseur Forcé (échec strict si erreur, pas de repli silencieux)
    if prov not in ("auto", "cascade", "default"):
        provider_map = {
            "groq": _call_groq,
            "gemini": _call_gemini,
            "mistral": _call_mistral,
            "anthropic": _call_anthropic,
        }
        if prov not in provider_map:
            raise PermanentLLMError(
                f"Fournisseur LLM inconnu : '{prov}'. "
                f"Fournisseurs supportés : {', '.join(provider_map.keys())}, 'auto'."
            )
        return provider_map[prov]

    # 2. Mode Cascade Automatique : Groq -> Gemini -> Mistral -> Anthropic
    cascade_chain: List[Tuple[str, str, Callable[[str], str]]] = [
        ("Groq", "GROQ_API_KEY", _call_groq),
        ("Gemini", "GEMINI_API_KEY", _call_gemini),
        ("Mistral", "MISTRAL_API_KEY", _call_mistral),
        ("Anthropic", "ANTHROPIC_API_KEY", _call_anthropic),
    ]

    def _cascade_caller(prompt: str) -> str:
        # Filtrer les fournisseurs ayant une clé configurée
        active_chain = [(name, fn) for name, env_var, fn in cascade_chain if os.environ.get(env_var)]
        if not active_chain:
            raise PermanentLLMError(
                "Aucune clé API LLM configurée pour la cascade automatique. "
                "Veuillez définir au moins l'une des variables : GROQ_API_KEY, GEMINI_API_KEY, "
                "MISTRAL_API_KEY ou ANTHROPIC_API_KEY."
            )

        errors: List[str] = []
        has_transient = False
        has_permanent = False
        for name, fn in active_chain:
            try:
                logger.info("Tentative de scoring via le fournisseur LLM : %s", name)
                res = fn(prompt)
                if res and res.strip():
                    return res
            except TransientLLMError as exc:
                has_transient = True
                logger.warning("Fournisseur %s temporairement indisponible (%s) -> bascule vers le suivant...", name, exc)
                errors.append(f"{name}: {exc}")
            except PermanentLLMError as exc:
                has_permanent = True
                logger.warning("Fournisseur %s non fonctionnel (%s) -> bascule vers le suivant...", name, exc)
                errors.append(f"{name}: {exc}")
            except Exception as exc:
                has_transient = True
                logger.warning("Erreur inattendue sur %s (%s) -> bascule vers le suivant...", name, exc)
                errors.append(f"{name}: {exc}")

        if has_transient and not has_permanent:
            raise TransientLLMError(
                f"Tous les fournisseurs LLM de la cascade ont échoué (erreurs transitoires). Détails : {'; '.join(errors)}"
            )

        raise PermanentLLMError(
            f"Tous les fournisseurs LLM de la cascade ont échoué. Détails : {'; '.join(errors)}"
        )

    return _cascade_caller
