---
name: sous-titres-animes
description: Compétence à utiliser pour toute tâche de génération ou d'incrustation de sous-titres dans le pipeline de repurposing vidéo — format ASS, calage mot-par-mot, burn-in ffmpeg, recalage temporel relatif au clip. Mots-clés — sous-titres, ASS, burn-in, karaoke, captions, incrustation, ffmpeg.
---

# Méthode de génération et incrustation des sous-titres

## Contrat de sortie

```python
def generate_ass(transcript: Dict, clip_start: float, clip_end: float, out_path: str) -> None:
    ...

def burn_subtitles(video_path: str, ass_path: str, out_path: str) -> Dict:
    # Retourne {"path": str, "warning": Optional[str]}
```

## Recalage temporel — le piège le plus probable de ce module

Les timestamps du transcript sont **absolus** (position dans la vidéo
source complète), mais un clip commence à `clip_start` et se termine à
`clip_end`. Tout timestamp de mot doit être recalé par rapport au **début
du clip**, pas de la vidéo source :
```
rel_start = max(0.0, word.start - clip_start)
rel_end = min(clip_end - clip_start, word.end - clip_start)
```
Un oubli de ce recalage produit des sous-titres qui apparaissent au
mauvais moment (souvent bien après la fin du clip) — une régression
silencieuse, aucune erreur ne sera levée nulle part.

Ignorer les mots dont l'intervalle ne recouvre pas `[clip_start,
clip_end]` (mots appartenant à d'autres portions du transcript).

## Découpage en blocs affichables

- Regrouper les mots en blocs de sous-titres.
- Contraintes de blocage (calibrage repris d'un projet de référence
  étudié) : maximum ~20 caractères et ~2 secondes par bloc — au-delà, un
  nouveau bloc commence.
- Les frontières de bloc doivent tomber entre deux mots, jamais couper un
  mot en deux.

## Mots avec timestamp manquant (cas déjà rencontré dans `transcriber.py`)

Certains mots peuvent avoir `start`/`end` absents ou `None` (même cause
que dans le module de transcription). Ce module doit gérer ce cas
gracieusement : ignorer le mot pour le calage précis plutôt que de lever
une exception qui ferait échouer tout le clip.

## Échappement du texte pour le format ASS

Le format ASS utilise des caractères spéciaux pour ses balises de style
(`{`, `}`, `\`) et sa syntaxe de champs (`,` sépare les colonnes d'un
`Dialogue:`). Un mot transcrit contenant l'un de ces caractères doit être
échappé ou neutralisé avant d'être inséré dans le fichier `.ass` — sinon
il peut soit casser le parsing du fichier par ffmpeg, soit être interprété
à tort comme une balise de style.

## Dégradation gracieuse à l'incrustation (principe déjà appliqué dans `clipper.py`)

Si `burn_subtitles` échoue (police manquante, fichier ASS mal formé,
ffmpeg refuse la commande) : **le clip est livré sans sous-titres**
(copie simple du fichier vidéo d'entrée vers `out_path`) plutôt que perdu.
Le retour inclut un `warning` explicite. Ce n'est pas un échec bloquant
pour le clip, encore moins pour le job.

## Réutilisation — ne pas dupliquer la sécurité subprocess

`app/clipper.py` contient déjà `_run_ffmpeg` (wrapper sécurisé :
liste d'arguments, jamais `shell=True`, timeout explicite, distinction
`CalledProcessError`/`TimeoutExpired`). Ce module doit **importer et
réutiliser cette fonction** plutôt que de dupliquer sa propre version —
une deuxième implémentation qui diverge silencieusement de la première
est un risque de maintenance, pas une amélioration.

## Tests attendus

- Découpage nominal : une phrase courte tient dans un seul bloc.
- Découpage par limite de caractères : une phrase longue est bien coupée
  en plusieurs blocs à ~20 caractères, jamais au milieu d'un mot.
- Découpage par limite de durée : des mots espacés dans le temps (silences
  entre eux) déclenchent un nouveau bloc après ~2 secondes même si peu de
  caractères ont été accumulés.
- Recalage temporel : un mot à la seconde 125 de la vidéo source, dans un
  clip qui commence à la seconde 120, doit apparaître à la seconde 5 dans
  le fichier ASS généré (pas à la seconde 125).
- Mots hors du clip : un mot dont l'intervalle ne recouvre pas
  `[clip_start, clip_end]` est ignoré, pas inclus dans le fichier ASS.
- Mot avec timestamp `None` : ne casse pas la génération, le mot est
  simplement omis du calage fin.
- Échappement : un mot contenant une virgule ou une accolade ne casse pas
  la structure du fichier ASS généré (vérifier que le fichier reste
  parsable/valide après génération).
- Incrustation réussie : `burn_subtitles` retourne `warning=None`.
- Incrustation en échec : simuler un échec ffmpeg → le clip est livré
  sans sous-titres (fichier copié), `warning` explicite présent, pas
  d'exception qui remonte.
- Réutilisation confirmée : vérifier que ce module importe bien
  `_run_ffmpeg` depuis `app.clipper` plutôt que de le redéfinir localement.
