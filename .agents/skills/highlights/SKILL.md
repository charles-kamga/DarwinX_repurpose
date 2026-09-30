---
name: scoring-moments-viraux
description: Compétence à utiliser pour toute tâche liée au scoring des moments forts d'un transcript via LLM dans le pipeline de repurposing vidéo — fenêtrage de transcript, prompt de critères de viralité, dédoublonnage de segments, sélection du nombre de clips. Mots-clés — highlights, scoring, moments viraux, fenêtrage, dédup, LLM, sélection de clips.
---

# Méthode de scoring des moments (module le plus sensible du pipeline)

Ce module est le cœur de la différenciation produit — la rigueur ici compte
plus que partout ailleurs dans le pipeline.

## Contrat de sortie

```python
def get_highlights(transcript: Dict, llm_fn: Optional[Callable[[str], str]] = None) -> List[Dict]:
    ...
```

Retourne une liste de dicts :
```
{"start_time": float, "end_time": float, "score": int, "hook_sentence": str, "reason": str}
```

`llm_fn` doit être injectable (paramètre optionnel, valeur par défaut =
l'appel réel à l'API). C'est ce qui permet de tester toute la logique de
fenêtrage/dédup/sélection sans jamais appeler un vrai LLM.

## Fenêtrage du transcript

- Découper le transcript en fenêtres d'environ `WINDOW_SECONDS` (config),
  **alignées sur les frontières de segments** — ne jamais couper une
  fenêtre au milieu d'un segment de parole.
- Chevauchement de `WINDOW_OVERLAP_SECONDS` entre fenêtres consécutives,
  pour ne pas rater un moment fort à cheval sur deux fenêtres.
- Si la vidéo est courte (durée sous `LONG_VIDEO_THRESHOLD_SECONDS`), une
  seule fenêtre couvrant tout le transcript suffit — pas la peine de
  fragmenter inutilement et de multiplier les appels LLM.

## Retry — distinguer transitoire de permanent (principe AGENTS.md appliqué ici concrètement)

- **Transitoire (retry justifié, avec backoff exponentiel)** : timeout
  réseau, erreur 5xx, rate limit (429).
- **Permanent (ne JAMAIS retry)** : contenu bloqué par un filtre de
  sécurité de l'API, erreur d'authentification, requête structurellement
  invalide (ex. prompt vide). Retenter ces cas gaspille des appels payants
  sans jamais réussir — le module doit les identifier explicitement, pas
  les traiter avec le même compteur de retry que les erreurs transitoires.
- Nombre maximum de tentatives et backoff configurables (voir `config.py`).

## Parsing de la réponse LLM

- La sortie attendue est du JSON, mais un LLM peut produire du texte
  autour (balises markdown, préambule). Le parsing doit tenter une
  extraction tolérante avant d'abandonner (ex. isoler le premier `{` et le
  dernier `}` si le parsing strict échoue).
- Toute entrée dont `end_time <= start_time`, ou dont le score est hors de
  l'intervalle 0-100, ou dont la durée ne respecte pas
  `CLIP_MIN_SECONDS`/`CLIP_MAX_SECONDS`, doit être rejetée silencieusement
  (pas de crash) — ne jamais faire confiance aveuglément à la sortie du LLM
  pour des contraintes qu'on peut vérifier soi-même en code.

## Dégradation gracieuse par fenêtre

Si une fenêtre échoue définitivement (JSON invalide après tous les
retries, ou erreur permanente détectée) : cette fenêtre est ignorée, un
avertissement est loggué, et le traitement continue sur les fenêtres
suivantes. Le job entier n'échoue QUE si **aucune** fenêtre n'a produit de
highlight valide au final (voir AGENTS.md : "aucun clip généré après
scoring" est un des rares échecs bloquants).

## Dédoublonnage

- Trier les highlights par score décroissant.
- Deux highlights sont considérés en conflit si leur recouvrement temporel
  dépasse 50% de la durée du plus court des deux.
- En cas de conflit, garder celui avec le score le plus élevé, écarter
  l'autre.

## Plancher/plafond sur le nombre de clips retenus

Après dédup et tri par score, ne garder qu'entre `CLIP_COUNT_MIN` et
`CLIP_COUNT_MAX` clips (config). Ce n'est pas un détail arbitraire : une
donnée de production observée chez un projet de référence montre que la
rétention utilisateur dépend du nombre de clips livrés (4-9 = bon taux de
retour, 1-3 = quasi nul), pas seulement de leur qualité perçue individuelle.

## Extension — support d'un second fournisseur LLM (Groq)

`get_highlights` utilise déjà un mécanisme de fournisseur par défaut
(`_default_anthropic_caller`) quand `llm_fn` n'est pas fourni. Ce
mécanisme doit être étendu pour choisir dynamiquement entre Anthropic et
Groq selon `config.LLM_PROVIDER` (`"anthropic"` | `"groq"`), **sans
modifier la signature de `get_highlights` ni celle de `pipeline.py`** —
`pipeline.py` continue d'appeler `highlights.get_highlights(transcript)`
sans rien savoir du fournisseur choisi.

**Exigences** :
- Nouvelle fonction `_default_groq_caller(prompt: str) -> str`, même
  contrat exact que `_default_anthropic_caller` (une chaîne en entrée,
  une chaîne en sortie).
- Utilise le SDK officiel `groq` (interface calquée sur celle d'OpenAI).
- Clé API lue depuis `GROQ_API_KEY`. Absence de clé → `PermanentLLMError`
  explicite, même pattern que `_default_anthropic_caller` (pas de retry
  inutile sur une erreur de configuration).
- Modèle configurable via `config.GROQ_MODEL` (config par défaut à un
  modèle gratuit rapide du catalogue Groq).
- **Ne pas toucher à `_is_transient_error` ni à `_call_llm_with_retry`** —
  vérifier d'abord (et l'écrire dans le plan) que les exceptions du SDK
  Groq exposent bien un attribut `status_code` ou un nom de classe
  reconnu par la classification existante, plutôt que de supposer que ça
  fonctionne sans vérification.
- Sélection du fournisseur par défaut au sommet de `get_highlights` :
  `caller = llm_fn or (_default_groq_caller if config.LLM_PROVIDER == "groq" else _default_anthropic_caller)`.

**Tests attendus pour cette extension** :
- `config.LLM_PROVIDER="groq"` sans `llm_fn` fourni → `_default_groq_caller`
  est bien utilisé (vérifiable par mock).
- Absence de `GROQ_API_KEY` → `PermanentLLMError` explicite.
- Une erreur simulée du SDK Groq avec un `status_code=429` (ou équivalent
  rate limit) est bien classée comme transitoire par `_is_transient_error`
  sans aucune modification de cette fonction — preuve que l'intégration
  est réellement compatible avec la classification existante, pas
  supposée compatible.

## Tests attendus

- Fenêtrage : un transcript de segments donné produit des fenêtres qui ne
  coupent jamais un segment en deux, avec le chevauchement attendu.
- Retry transitoire : `llm_fn` simulée qui échoue 2 fois puis réussit → le
  résultat final est correct, avec exactement 3 appels.
- Erreur permanente : `llm_fn` simulée qui lève une erreur marquée comme
  permanente → aucun retry, une seule tentative, fenêtre ignorée.
- JSON malformé (texte autour, guillemets cassés) → soit récupéré par le
  parsing tolérant, soit la fenêtre est proprement ignorée sans crash.
- Rejet des entrées hors contraintes : score à 150, ou `end_time <
  start_time`, ou durée de 200s → l'entrée est filtrée, pas de crash, pas
  de clip invalide dans le résultat final.
- Dédup : deux highlights très chevauchants → seul celui au score le plus
  haut est conservé.
- Plancher/plafond : simuler 15 highlights valides distincts → le résultat
  final ne dépasse pas `CLIP_COUNT_MAX`.
- Cas totalement bloquant : toutes les fenêtres échouent → le module lève
  une exception explicite plutôt que de retourner une liste vide
  silencieuse (c'est `pipeline.py` qui décide ensuite que c'est un échec
  de job, mais le module doit signaler clairement l'absence de résultat).
