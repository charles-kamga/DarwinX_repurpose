---
name: transcription-audio
description: Compétence à utiliser pour toute tâche de transcription audio ou vidéo avec timestamps précis via faster-whisper dans le pipeline de repurposing vidéo. Mots-clés pertinents — transcription, whisper, timestamps mot-par-mot, cache de transcript, VAD, segments, sous-titres.
---

# Méthode de transcription (faster-whisper)

## Contrat de sortie

La fonction doit retourner :
```
{
  "duration": float,
  "segments": [
    {"start": float, "end": float, "text": str,
     "words": [{"w": str, "s": float, "e": float}, ...]}
  ]
}
```

Les timestamps mot-par-mot (`words`) sont **obligatoires**, pas optionnels :
le module de sous-titres et le calage précis des coupes de clips en
dépendent en aval. Ne jamais livrer un transcript sans cette information
sous prétexte que "start/end de segment suffit".

## Sélection du device

- Détecter CUDA automatiquement si disponible, repli CPU sinon.
- Ce repli CPU n'est **pas** une dégradation à signaler comme avertissement
  de job — c'est un comportement normal et attendu, pas un échec partiel.
- `compute_type="int8"` sur CPU, `"float16"` sur CUDA.

## Cache

- Le résultat est mis en cache sur disque (clé = nom du fichier source), pour
  éviter de retranscrire à chaque relance en développement.
- Un cache vide ou invalide (0 segments alors que le fichier source existe)
  doit être traité comme **absent**, pas comme un résultat valide à
  retourner tel quel — retranscrire plutôt que retourner un résultat vide
  silencieusement.

## Ce qui constitue un échec BLOQUANT pour ce module (voir AGENTS.md)

- Aucune parole détectée dans tout le fichier (0 segments) → c'est un des
  rares cas où le **job entier** doit échouer, pas une dégradation
  gracieuse : un pipeline de repurposing n'a rien à repurposer sans
  transcript.
- Fichier audio/vidéo illisible ou corrompu → échec bloquant également.

## Ce qui N'est PAS un échec bloquant

- Langue non fournie explicitement → utiliser la détection automatique de
  faster-whisper par défaut, ne pas bloquer le job pour ça.
- Device CUDA indisponible → repli CPU silencieux (voir ci-dessus).

## Tests attendus

- Cas nominal : extrait audio court avec parole claire → au moins 1 segment
  avec `words` non vide.
- Cas cache : appeler deux fois sur le même fichier → le deuxième appel
  doit réutiliser le cache (pas de re-transcription) et retourner un
  résultat identique.
- Cas cache corrompu : écrire manuellement un cache avec 0 segments, puis
  appeler → doit détecter l'invalidité et retranscrire plutôt que
  retourner le cache vide.
- Cas limite : fichier audio silencieux (pas de parole) → doit lever une
  erreur explicite plutôt que retourner un résultat vide comme si de rien
  n'était.
