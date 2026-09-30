---
name: orchestration-pipeline
description: Compétence à utiliser pour toute tâche d'orchestration du pipeline de repurposing vidéo — enchaînement transcription/scoring/cadrage/sous-titres, machine à états de job, dégradation gracieuse au niveau global, persistance de statut. Mots-clés — pipeline, orchestrateur, job, statut, état, run_pipeline, warnings.
---

# Méthode d'orchestration du pipeline

Ce module relie les 4 briques déjà validées (`transcriber`, `highlights`,
`clipper`, `subtitles`) et gère la machine à états de job exigée par le
Principe n°3 d'AGENTS.md. Aucune nouvelle logique métier ici — uniquement
de l'enchaînement, de la traduction d'erreurs, et de la persistance de
statut.

## Contrats exacts des modules déjà validés (à respecter tels quels)

```python
transcriber.transcribe(media_path: str, language=None, force=False, model_size=None) -> Dict
# lève FileNotFoundError, EmptyTranscriptionError, TranscriberMediaError

highlights.get_highlights(transcript: Dict, llm_fn=None) -> List[Dict]
# lève NoHighlightsFoundError

clipper.make_clips(source_path: str, highlights: List[Dict], out_dir: str) -> List[Dict]
# chaque élément enrichi de "path" (Optional[str]) et "warning" (Optional[str])

subtitles.generate_ass(transcript: Dict, clip_start: float, clip_end: float, out_path: str) -> None
subtitles.burn_subtitles(video_path: str, ass_path: str, out_path: str) -> Dict
# {"path": str, "warning": Optional[str]}
```

`clip_start`/`clip_end` pour `subtitles` sont les timestamps **absolus**
dans la vidéo source — ce sont `highlight["start_time"]` et
`highlight["end_time"]`, pas des valeurs recalées manuellement par ce
module (le recalage relatif est déjà fait à l'intérieur de
`subtitles.generate_ass`).

## Découplage volontaire de l'ingestion

Ce module orchestre à partir d'un **fichier déjà présent sur disque**
(`source_path`), pas d'une URL YouTube. Le téléchargement (`downloader.py`)
est un module séparé, pas encore construit à ce stade — ce découplage
permet de tester et valider l'orchestrateur sans dépendre d'un module qui
n'existe pas encore.

## Contrat de sortie

```python
def run_pipeline(job_id: str, source_path: str, out_dir: str) -> Dict:
    ...
```

## Machine à états de job (nouveau module `app/job_store.py` à créer)

Statuts : `queued` → `processing` (avec l'étape en cours) →
`completed` | `failed`. Persisté sur disque en JSON, écriture **atomique**
(écrire dans un fichier temporaire puis `os.replace`, jamais écrire
directement dans le fichier final — un crash pendant l'écriture ne doit
jamais laisser un fichier de statut à moitié écrit et donc corrompu).

Fonctions attendues : `create_job`, `get_job`, `set_step`,
`mark_step_done`, `add_warning`, `mark_completed`, `mark_failed`.

## Garantie absolue : un job ne reste jamais bloqué en "processing"

C'est la règle la plus importante de ce module. Toute l'orchestration doit
être encadrée par une capture large en tout dernier recours : si une
exception totalement imprévue s'échappe d'une étape (bug, erreur non
anticipée), le job doit être marqué `failed` avec le message d'erreur,
**jamais laissé indéfiniment en `processing`**. Un job bloqué sans
résolution est pire qu'un job en échec explicite — l'utilisateur ne saurait
même pas qu'il doit relancer.

## Ce qui est bloquant pour le JOB ENTIER (les 3 seuls cas, per AGENTS.md)

1. Transcription vide ou média illisible (`EmptyTranscriptionError`,
   `TranscriberMediaError`, `FileNotFoundError` remontant de
   `transcriber.transcribe`).
2. Aucun moment fort extrait (`NoHighlightsFoundError` de
   `highlights.get_highlights`).
3. Tous les clips ont échoué au montage (`path is None` pour chaque
   élément retourné par `clipper.make_clips`).

Dans chacun de ces 3 cas : `job_store.mark_failed(job_id, message)`,
propager ou non l'exception selon ce qui est le plus pratique pour
l'appelant, mais le statut de job doit refléter l'échec.

## Ce qui N'est PAS bloquant (dégradation gracieuse au niveau global)

- Un clip individuel en échec (`path is None`) alors que d'autres ont
  réussi → ce clip est exclu du résultat final, un avertissement est
  ajouté au job, le traitement continue sur les autres clips.
- Un warning renvoyé par `clipper.make_clips` (repli crop centré) ou par
  `subtitles.burn_subtitles` (clip livré sans sous-titres) → capturé dans
  les avertissements du job, jamais transformé en échec.

## Concurrence — verrouillage par job (ajout après relecture du code livré)

`os.replace` garantit qu'une écriture individuelle n'est jamais lue à
moitié faite, mais ne protège **pas** contre une course lecture-modification-
écriture : si deux appels concurrents (ex. deux threads futurs traitant
plusieurs jobs en parallèle) lisent le même état avant que l'un des deux
n'écrive, le second à écrire efface silencieusement la modification du
premier — une perte de mise à jour, pas une corruption de fichier.

Le pipeline actuel est strictement séquentiel (ce risque est aujourd'hui
théorique), mais comme un traitement parallèle est prévu plus tard
(`ThreadPoolExecutor` ou équivalent), ce point doit être corrigé
maintenant plutôt que reporté.

**Exigences** :
- Chaque opération de mise à jour d'un job (`set_step`, `mark_step_done`,
  `add_warning`, `mark_completed`, `mark_failed`, et le cycle interne de
  `_update_job`) doit être protégée par un verrou **scindé par `job_id`**
  — verrouiller un job ne doit jamais bloquer les opérations sur un autre
  job.
- Le verrou doit avoir un **timeout explicite** (cohérent avec le reste du
  projet : jamais d'attente indéfinie). Si le verrou ne peut être acquis
  dans le délai, lever une `JobStoreError` explicite plutôt que de rester
  bloqué indéfiniment.
- Solution recommandée : un verrou de fichier inter-process (ex. la
  librairie `filelock`, légère et standard) sur un fichier `.lock` dédié
  par `job_id`, plutôt qu'un simple `threading.Lock` qui ne protégerait
  pas contre plusieurs process séparés (utile si le pipeline tourne un
  jour derrière plusieurs workers). Si une autre solution est choisie,
  la justifier explicitement dans le plan.
- Ajouter `filelock` à `requirements.txt` si retenu — nouvelle dépendance
  à signaler explicitement (voir deny rules d'AGENTS.md).

**Tests attendus pour cette correction** :
- Deux appels concurrents (simulés via threads) sur le **même** `job_id`
  (ex. deux `add_warning` avec des messages différents) → après les deux
  appels, les deux avertissements sont bien présents (aucune perte de
  mise à jour).
- Le verrouillage d'un `job_id` ne bloque pas une opération sur un
  **autre** `job_id` simultané (vérifier que ce n'est pas un verrou
  global déguisé).
- Un verrou maintenu artificiellement (simulé) au-delà du timeout
  configuré déclenche une `JobStoreError` claire plutôt qu'un blocage
  indéfini.

## Tests attendus

- Cas nominal : les 4 modules mockés en succès → job `completed`, toutes
  les étapes marquées `mark_step_done`, résultat contient tous les clips
  avec leur `final_path`.
- Transcription vide : mock de `transcriber.transcribe` qui lève
  `EmptyTranscriptionError` → job `failed`, message explicite, aucune
  étape suivante n'est tentée (le mock de `highlights.get_highlights`
  n'est jamais appelé).
- Aucun highlight : mock de `highlights.get_highlights` qui lève
  `NoHighlightsFoundError` → job `failed`.
- Tous les clips échouent : mock de `clipper.make_clips` qui retourne une
  liste où chaque élément a `path: None` → job `failed`, message explicite.
- Dégradation partielle : mock de `clipper.make_clips` qui retourne 3
  clips dont 1 avec `path: None` → job `completed`, résultat final contient
  2 clips, 1 avertissement enregistré pour le clip en échec.
- Warning de sous-titres : mock de `subtitles.burn_subtitles` qui retourne
  `{"path": ..., "warning": "..."}` pour un clip → le clip est quand même
  inclus dans le résultat final, le warning est enregistré sur le job, pas
  d'échec.
- Exception totalement imprévue (ex. `TypeError` simulé au milieu du
  traitement d'un clip) → le job est marqué `failed` explicitement, jamais
  laissé en `processing`.
- Persistance atomique : écrire un job, simuler une interruption pendant
  l'écriture (ou au minimum vérifier qu'un fichier temporaire est utilisé
  puis renommé), puis vérifier que `get_job` relit un état cohérent.
