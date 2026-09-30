---
name: ingestion-youtube
description: Compétence à utiliser pour toute tâche de téléchargement de vidéo source via yt-dlp dans le pipeline de repurposing vidéo — retry ciblé, sonde de qualité fail-open. Mots-clés — downloader, yt-dlp, téléchargement, ingestion, YouTube.
---

# Méthode d'ingestion YouTube

Module le plus simple du pipeline — pas de nouvelle logique métier
complexe, principalement de la configuration réfléchie de `yt-dlp`.

## Contrat de sortie

```python
def probe_max_quality(url: str) -> Dict:
    # Retourne {"max_height": int, "duration": int}. Ne lève JAMAIS d'exception.

def download_youtube(url: str, out_dir: Optional[str] = None, max_height: int = 1080) -> str:
    # Retourne le chemin local du fichier téléchargé.
```

## Sonde de qualité — fail-open (jamais bloquante)

`probe_max_quality` sert à connaître la résolution disponible avant de
lancer un traitement long (transcription + rendu peuvent prendre 10+
minutes). Si la sonde elle-même échoue (réseau, vidéo privée, API
YouTube qui change), elle **ne doit jamais lever d'exception** — retourner
des valeurs par défaut (`{"max_height": 0, "duration": 0}`) et laisser
l'appelant décider de continuer quand même.

## Retry ciblé sur le téléchargement réel

- Retenter uniquement sur des erreurs identifiées comme transitoires côté
  YouTube (ex. `403 Forbidden`, souvent lié à un throttling temporaire).
- Ne pas retenter sur une URL structurellement invalide ou une vidéo
  supprimée/privée — ça ne se réparera jamais en réessayant.
- Nombre de tentatives limité (pas de boucle infinie).

## Ce qui est bloquant

Un échec de téléchargement après épuisement des tentatives doit lever une
exception explicite — ce module ne fait pas de dégradation gracieuse sur
son propre échec (il n'y a pas de repli possible si la vidéo source est
introuvable, contrairement aux autres modules qui ont des replis
fonctionnels).

## Tests attendus

- Téléchargement nominal réussi (mock `yt_dlp.YoutubeDL`).
- Retry sur erreur 403 puis succès au deuxième essai.
- Erreur non retryable (URL invalide) → échec immédiat, pas de retry inutile.
- Sonde de qualité qui échoue → retourne les valeurs par défaut, ne lève
  jamais d'exception.
- Sonde de qualité qui réussit → retourne la résolution et la durée
  correctes à partir des formats disponibles.
