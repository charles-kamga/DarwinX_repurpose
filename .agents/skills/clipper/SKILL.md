---
name: cadrage-vertical
description: Compétence à utiliser pour toute tâche de découpe de clips et de cadrage vertical (9:16) dans le pipeline de repurposing vidéo — ffmpeg, détection de visage, suivi de speaker, gestion de sous-processus, nettoyage de fichiers temporaires. Mots-clés — clipper, cadrage, crop, ffmpeg, subprocess, découpe, vertical, visage.
---

# Méthode de découpe + cadrage vertical

Ce module manipule des sous-processus externes (ffmpeg) et des fichiers
temporaires sur disque — la rigueur porte autant sur la sécurité des appels
système que sur la logique de cadrage elle-même.

## Contrat de sortie

```python
def make_clip(source_path: str, start: float, end: float, out_path: str) -> Dict:
    # Retourne {"path": str, "warning": Optional[str]}

def make_clips(source_path: str, highlights: List[Dict], out_dir: str) -> List[Dict]:
    # Retourne une liste, un élément par highlight, chacun enrichi de "path" et "warning"
```

## Sécurité des appels sous-processus (non négociable)

- **Jamais `shell=True`.** Toujours passer la commande sous forme de liste
  (`["ffmpeg", "-i", source_path, ...]`), jamais une chaîne concaténée. Un
  nom de fichier contenant des caractères spéciaux ne doit jamais pouvoir
  être interprété par un shell.
- **Toujours un `timeout` explicite** sur chaque appel `subprocess.run`
  (le traitement vidéo peut bloquer indéfiniment sur un fichier corrompu
  sans ça).
- **Toujours vérifier le code de sortie** (`check=True` ou vérification
  manuelle de `returncode`), et capturer spécifiquement
  `subprocess.CalledProcessError` et `subprocess.TimeoutExpired` plutôt
  qu'un `except Exception` générique — ce sont deux échecs de nature
  différente (l'un dit "ffmpeg a refusé", l'autre dit "ffmpeg est resté
  bloqué").

## Nettoyage des fichiers temporaires (garanti, pas optionnel)

Toute étape qui crée un fichier intermédiaire (clip découpé avant cadrage,
vidéo silencieuse avant réassemblage audio) doit le supprimer dans un bloc
`finally`, même si une étape suivante échoue. Un pipeline qui traite des
dizaines de vidéos sans ce nettoyage garanti remplit le disque en
production silencieusement — un bug qui ne se voit qu'après plusieurs
jours d'utilisation réelle, donc particulièrement dangereux à laisser
passer maintenant.

## Cadrage vertical par visage, avec repli gracieux

- Détection de visage par frame (Haar cascade OpenCV suffit pour un
  speaker unique — pas besoin de suivi de speaker actif multi-personnes
  pour ce module).
- Lissage du centre de recadrage entre frames (facteur de lissage, ex.
  0.15) pour éviter les sauts brusques quand la détection change de
  position d'une frame à l'autre.
- **Si aucun visage n'est détecté sur toute la durée du clip** : repli sur
  un crop centré simple. Ce n'est PAS un échec — c'est un avertissement à
  journaliser (`warning` dans le retour de `make_clip`), le clip est
  livré quand même.
- **Si le suivi de visage plante techniquement** (OpenCV absent, erreur de
  lecture vidéo) : même repli sur crop centré via ffmpeg seul, avec
  avertissement — ne jamais perdre le clip pour cette raison.

## Dégradation gracieuse au niveau du lot (`make_clips`)

Un clip qui échoue **totalement** (le découpage ffmpeg lui-même échoue,
pas seulement le cadrage) ne doit pas interrompre le traitement des autres
clips. Il apparaît dans le résultat avec `"path": None` et un `"warning"`
explicite. Conformément à AGENTS.md, le job entier n'échoue que si **tous**
les clips ont échoué (cette décision-là revient à `pipeline.py`, pas à ce
module — `make_clips` se contente de rapporter fidèlement ce qui a marché
et ce qui n'a pas marché).

## Chemins de fichiers

- Les noms de fichiers de sortie sont générés par le code (ex.
  `clip_{i:02d}.mp4`), jamais construits à partir d'une chaîne fournie par
  l'utilisateur ou par le LLM — pas de risque de traversée de répertoire
  (`../../`) à travers ce module.

## Tests attendus

- Cas nominal : visage détectable sur toute la durée → cadrage suivi,
  pas de warning.
- Cas sans visage : aucune détection sur toute la durée → repli centré,
  warning explicite présent, mais `path` renseigné (le clip existe).
- Cas échec technique du cadrage (ex. OpenCV lève une exception simulée)
  → repli centré via ffmpeg, warning explicite, clip quand même livré.
- Cas échec total du découpage ffmpeg (commande simulée en échec) → ce
  clip a `"path": None` avec warning, mais les AUTRES clips du même lot
  sont quand même traités normalement (tester avec plusieurs highlights,
  dont un seul en échec).
- Nettoyage : vérifier qu'un fichier intermédiaire est bien supprimé même
  quand l'étape suivante lève une exception (simuler l'échec après la
  création du fichier temporaire, puis vérifier son absence sur disque).
- Sécurité subprocess : vérifier (via mock de `subprocess.run`) que les
  appels sont faits avec une liste d'arguments et jamais avec `shell=True`,
  et qu'un `timeout` est bien passé à chaque appel.
