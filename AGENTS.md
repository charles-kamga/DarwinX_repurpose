# AGENTS.md — Repurposing SaaS (long → shorts)

Ce fichier est lu par l'agent (Antigravity) au démarrage de chaque session.
Il encode les standards du projet — pas des préférences vagues, mais des
règles concrètes tirées de l'étude de projets open source de référence
(AI-Youtube-Shorts-Generator, OpenShorts) et des retours de production
qu'ils documentent dans leur propre code.

## Contexte du projet

SaaS de repurposing vidéo : transforme une vidéo longue (podcast, interview,
conférence) en plusieurs clips courts verticaux (9:16) avec sous-titres
animés, prêts pour TikTok/Reels/Shorts.

Pipeline : ingestion (yt-dlp) → transcription (faster-whisper) → scoring
des moments (LLM) → sélection des clips → cadrage vertical → sous-titres →
export (ffmpeg).

Stack : Python (backend pipeline), FastAPI (API web quand elle existera),
ffmpeg, opencv-python, faster-whisper, Anthropic API pour le scoring.

## Principe directeur n°1 : dégradation gracieuse, jamais d'échec en cascade

Une étape non-critique qui échoue ne doit JAMAIS faire échouer tout le job.
Elle doit se replier sur un comportement dégradé et continuer, en
enregistrant un avertissement explicite.

Exemples concrets attendus dans ce projet :
- Le cadrage par suivi de visage échoue (aucun visage détecté, OpenCV
  indisponible) → repli sur un crop centré simple. Le clip est livré, pas
  perdu.
- L'incrustation des sous-titres échoue (police manquante, fichier ASS
  corrompu) → le clip est livré sans sous-titres plutôt que rejeté.
- Une fenêtre de transcript envoyée au LLM de scoring produit un JSON
  invalide après tous les retries → cette fenêtre est ignorée, les autres
  fenêtres continuent d'être traitées.

Les seules étapes autorisées à faire échouer le job ENTIER : téléchargement
impossible, transcription vide (aucune parole détectée), zéro clip généré
après scoring, tous les clips échoués au montage final.

Chaque module qui touche à une étape du pipeline doit documenter
explicitement, dans son plan d'implémentation, quel comportement de repli
il utilise en cas d'échec partiel.

## Principe directeur n°2 : retry ciblé, pas aveugle

Ne jamais retenter une opération sans distinguer erreur transitoire
(réseau, timeout, rate limit) d'erreur permanente (URL invalide, contenu
bloqué par un filtre de sécurité, format de réponse structurellement
incompatible). Un retry sur une erreur permanente gaspille du temps et de
l'argent (appels LLM/API payants) sans jamais réussir.

Tout retry doit avoir un nombre maximum de tentatives et, pour les appels
LLM, un backoff exponentiel entre les tentatives.

## Principe directeur n°3 : état de job traçable

Le traitement d'une vidéo prend plusieurs minutes. Chaque étape du pipeline
doit mettre à jour un état persisté (`queued` / `processing` / `completed`
/ `failed`) avec l'étape en cours, consultable à tout moment — y compris
après un crash ou un redémarrage du process. Ne jamais faire dépendre le
statut du job uniquement de l'état en mémoire d'un process qui peut mourir.

## Conventions de code

- Fichiers Python : viser 300 lignes maximum par fichier, à titre indicatif
  (facilite la relecture humaine complète plutôt qu'un survol partiel — ce
  n'est PAS une garantie anti-bug en soi). Si un module dépasse ce seuil,
  ce n'est pas bloquant, mais le rapport de tâche doit le signaler
  explicitement pour qu'on décide ensemble si un découpage est justifié.
- Toute fonction qui appelle un service externe (LLM, ffmpeg, yt-dlp) doit
  gérer explicitement le cas d'échec — pas de `except Exception: pass`
  silencieux ; toujours logger ou enregistrer l'avertissement.
- Pas de secrets en dur dans le code. Toute clé d'API vient d'une variable
  d'environnement.
- Docstring obligatoire sur chaque fonction publique d'un module, expliquant
  le "pourquoi" du choix technique quand il n'est pas évident (ex. pourquoi
  ce seuil, pourquoi ce repli).

## Exigences de test

- Toute nouvelle fonction de logique métier (scoring, dédup, cadrage,
  génération de sous-titres) doit avoir au moins un test unitaire couvrant
  le cas nominal ET le cas de dégradation gracieuse.
- Lancer la suite de tests avant de marquer une tâche comme terminée.
- Ne jamais considérer une tâche "done" uniquement parce que le code
  s'exécute sans erreur sur un seul cas — tester aussi un cas limite
  (vidéo sans visage détectable, transcript vide, réponse LLM malformée).

## Avant d'implémenter

Avant d'écrire du code pour une étape du pipeline, vérifier s'il existe un
fichier `.agents/skills/<nom-du-module>/SKILL.md` décrivant la méthode
attendue pour ce module (convention standard Antigravity — ces fichiers
sont chargés automatiquement selon le contexte de la tâche, pas besoin de
les invoquer par leur nom). S'il existe, le plan d'implémentation doit s'y
conformer explicitement et signaler tout écart.

Toujours produire le plan d'implémentation avant d'exécuter, et s'arrêter
pour validation avant d'écrire du code — ne pas enchaîner plan et exécution
dans la même passe sans confirmation explicite de l'utilisateur.

## Règles interdites (deny rules)

- Ne jamais modifier les seuils de `config.py` (durée des clips, nombre de
  clips min/max, timeouts) sans confirmation explicite — ce sont des
  décisions produit, pas des détails d'implémentation.
- Ne jamais committer de clé d'API, token, ou fichier `.env`.
- Ne jamais supprimer ou réécrire un test existant pour le faire passer —
  si un test bloque, il faut le signaler et discuter, pas le contourner.
- Ne jamais introduire une dépendance externe non listée dans
  `requirements.txt` sans le signaler dans le rapport de tâche.
