"""Module d'interface terminale riche (TUI) et de transparence verbale du pipeline.

Fournit une hiérarchie visuelle, un thème sémantique strict et un affichage
transparent en temps réel conformément au guide tui_and_pipeline_verbosity_guide.md.

Comportement :
- Verbosité activée par défaut (annonces d'intentions, micro-métriques live, tables de clips).
- Désactivable via l'argument CLI --non-verbose.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Thème sémantique conforme au guide de conception
_THEME = Theme({
    "info": "dim",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "brand": "bold bright_blue",
    "key": "cyan",
    "accent": "bold magenta",
})

console = Console(theme=_THEME, highlight=False)


class UIManager:
    """Gestionnaire central de l'affichage terminal du pipeline."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def set_verbose(self, enabled: bool) -> None:
        """Active ou désactive le mode verbeux."""
        self.verbose = enabled

    def is_verbose(self) -> bool:
        """Retourne l'état de verbosité actuel."""
        return self.verbose

    def print_banner(
        self,
        source: str,
        out_dir: str,
        whisper_device: str,
        llm_provider: str,
        encoder: Optional[str] = None,
    ) -> None:
        """Affiche le panneau d'accueil stylé avec les badges techniques."""
        if not self.verbose:
            return

        device_badge = f"[success]{whisper_device.upper()}[/success]" if "cuda" in whisper_device.lower() else f"[key]{whisper_device.upper()}[/key]"
        provider_badge = f"[accent]{llm_provider.upper()}[/accent]"
        source_display = source if len(source) <= 65 else f"{source[:62]}..."
        enc_part = f"  ·  Encodage : [success]{encoder}[/success]" if encoder else ""

        body = (
            f"[bold]Source[/bold]      : [key]{source_display}[/key]\n"
            f"[bold]Projet[/bold]      : [key]{out_dir}[/key]\n"
            f"[bold]Accélération[/bold]: Whisper sur {device_badge}{enc_part}  ·  Scoring LLM : {provider_badge}  ·  Format : [bold cyan]9:16 Vertical[/bold cyan]"
        )

        console.print(
            Panel(
                body,
                title="[brand]◆  Darwinx Repurpose — Pipeline Vidéo Intelligent[/brand]",
                subtitle="[dim]TikTok · Reels · YouTube Shorts[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        console.print()

    def print_step_header(self, step_idx: int, total_steps: int, title: str, detail: str = "") -> None:
        """Déclare l'intention d'une étape majeure du pipeline."""
        if not self.verbose:
            return
        detail_str = f" [dim]({detail})[/dim]" if detail else ""
        console.print(f"[brand]◆ [{step_idx}/{total_steps}] {title}[/brand]{detail_str}")

    def print_info(self, message: str) -> None:
        """Affiche une micro-métrique ou une information secondaire."""
        if not self.verbose:
            return
        console.print(f"  [info]→[/info] {message}")

    def print_success(self, message: str) -> None:
        """Affiche une validation ou un succès factuel."""
        if not self.verbose:
            return
        console.print(f"  [success]✔[/success] {message}")

    def print_warning(self, message: str) -> None:
        """Affiche un avertissement ou une dégradation gracieuse."""
        # Les avertissements sont toujours affichés, même en non-verbose
        console.print(f"  [warning]⚠[/warning] {message}")

    def print_error(self, message: str) -> None:
        """Affiche un échec bloquant."""
        console.print(f"  [error]✗[/error] {message}")

    def print_highlights_table(self, highlights: List[Dict[str, Any]]) -> None:
        """Affiche un tableau Rich soigné des moments viraux retenus après scoring."""
        if not self.verbose or not highlights:
            return

        table = Table(
            title="[brand]Moments Viraux Retenus pour Montage[/brand]",
            box=box.ROUNDED,
            border_style="bright_blue",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Intervalle", justify="center", width=17)
        table.add_column("Durée", justify="right", width=8)
        table.add_column("Phrase d'Accroche (Hook)", style="dim")

        for idx, h in enumerate(highlights, 1):
            score = h.get("score", 0)
            score_style = "bold green" if score >= 85 else ("bold yellow" if score >= 70 else "dim")
            s_time = h.get("start_time", 0.0)
            e_time = h.get("end_time", 0.0)
            dur = e_time - s_time
            interval_str = f"{int(s_time // 60):02d}:{s_time % 60:04.1f} → {int(e_time // 60):02d}:{e_time % 60:04.1f}"
            hook = h.get("hook_sentence") or h.get("reason", "")
            if len(hook) > 48:
                hook = f"{hook[:45]}..."

            table.add_row(
                str(idx),
                f"[{score_style}]{score}/100[/{score_style}]",
                interval_str,
                f"{dur:.1f}s",
                hook,
            )

        console.print(table)
        console.print()

    def print_clip_action(
        self,
        idx: int,
        total: int,
        action_name: str,
        clip_name: str,
        detail: str = "",
        warning: Optional[str] = None,
    ) -> None:
        """Affiche la progression live d'une opération sur un clip (cadrage ou sous-titres)."""
        if not self.verbose:
            return

        if warning:
            console.print(
                f"  [warning]⚠[/warning] Clip {idx}/{total} ({clip_name}) [{action_name}] : [yellow]{warning}[/yellow]"
            )
        else:
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            console.print(
                f"  [success]✔[/success] Clip {idx}/{total} ({clip_name}) [{action_name}]{detail_str}"
            )

    def print_final_report(
        self,
        job_id: str,
        duration_sec: float,
        clips: List[Dict[str, Any]],
        warnings: List[str],
        out_dir: str,
    ) -> None:
        """Affiche le grand bilan final du pipeline avec chemins cliquables."""
        table = Table(
            title="[brand]BILAN DE PRODUCTION — CLIPS GÉNÉRÉS[/brand]",
            box=box.DOUBLE_EDGE,
            border_style="bright_blue",
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Score", justify="center", width=8)
        table.add_column("Durée", justify="right", width=8)
        table.add_column("Titre Recommandé & Fichier Vidéo", style="key")

        for idx, clip in enumerate(clips, 1):
            score = clip.get("score", "N/A")
            s_time = float(clip.get("start_time", 0.0))
            e_time = float(clip.get("end_time", 0.0))
            dur = e_time - s_time
            clip_path = clip.get("final_path") or clip.get("path", "")
            rec_title = clip.get("recommended_title")
            title_display = f"[bold]{rec_title}[/bold]\n[dim link=file://{clip_path}]{clip_path}[/dim link]" if rec_title else f"[link=file://{clip_path}]{clip_path}[/link]"
            table.add_row(
                str(idx),
                f"[bold green]{score}[/bold green]",
                f"{dur:.1f}s",
                title_display,
            )

        console.print()
        console.print(table)
        console.print()

        # Grand bilan standard pour lisibilité et compatibilité
        console.print("=" * 60)
        console.print("          BILAN DU TRAITEMENT PIPELINE")
        console.print("=" * 60)
        console.print(f"  Identifiant Job      : {job_id}")
        console.print("  Statut final         : COMPLETED")
        console.print(f"  Durée totale         : {duration_sec:.2f} secondes")
        console.print(f"  Nombre de clips      : {len(clips)}")

        if clips:
            console.print("\n  Clips générés :")
            for idx, clip in enumerate(clips, start=1):
                c_path = clip.get("final_path") or clip.get("path")
                c_score = clip.get("score", "N/A")
                console.print(f"    [{idx}] Score: {c_score} | Fichier: {c_path}")
                if clip.get("recommended_title"):
                    console.print(f"        Titre : {clip.get('recommended_title')}")
                if clip.get("thumbnail_path"):
                    console.print(f"        Miniature : {clip.get('thumbnail_path')}")
                if clip.get("publish_txt_path"):
                    console.print(f"        Fiche diffusion : {clip.get('publish_txt_path')}")

        if warnings:
            console.print(f"\n  Avertissements ({len(warnings)}) :")
            for w in warnings:
                console.print(f"    - {w}")
        else:
            console.print("\n  Avertissements       : Aucun (traitement nominal sans dégradation)")

        console.print("=" * 60 + "\n")


# Instance singleton partagée
ui = UIManager(verbose=True)
