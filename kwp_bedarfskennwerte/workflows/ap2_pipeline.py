# -*- coding: utf-8 -*-
"""kwp_bedarfskennwerte.workflows.ap2_pipeline

WP2/AP2-Pipeline: Gebäudetypisierung (IWU) + Bedarfskennwerte Wärme/TWW + Kälte.

Diese Pipeline ist so geschrieben, dass sie *direkt* mit dem Context/Settings-Objekt
zusammenarbeitet, das durch die CLI (kwp_bedarfskennwerte.cli) erzeugt wird.

Erwartete Eingaben (aus ctx.settings.data)
-----------------------------------------
Pflicht:
- ap1_gpkg: Pfad zum AP1-/Enrichment-Gebäudelayer (GPKG)
- ap2_zensus_csv: CSV mit Zensus-Attributen (building_id-basiert)
- ap2_geom_csv: CSV mit Geometrie-Analyse (building_id-basiert)

Optional (falls nicht gesetzt, werden Defaults verwendet):
- ap2_out_gpkg: Zielpfad für den getypten Layer
- ap2_heat_demand_gpkg: Zielpfad für Wärme/TWW-Ausgabe
- ap2_cold_demand_gpkg: Zielpfad für Final-Ausgabe (Wärme/TWW + Kälte)
- iwu_base_dir: Basisverzeichnis der IWU-Tabellen (Default: data/Bedarfskennwerte_IWU)

Outputs (Default unter <settings.out_dir>/ap2)
---------------------------------------------
- ap2_buildings_typed.gpkg
- ap2_buildings_heat_demand.gpkg
- ap2_buildings_heat_cold_demand.gpkg

Hinweis zu QA
-------------
heat_demand.compute_heat_demand_for_ap2 schreibt QA-Dateien unter <out_dir>/ap2/qa.
Damit diese an der erwarteten Stelle landen, übergeben wir an heat_demand bewusst den
*Basis*-out_dir (settings.out_dir) und nicht das ap2-Unterverzeichnis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..methodology import building_typing, cold_demand, heat_demand, refurbishment
from ..config.paths import rel_bedarfskennwerte_iwu_dir


def _resolve_ap2_dir(base_out_dir: Path) -> Path:
    """Ermittelt das AP2-Ausgabeverzeichnis robust.

    In manchen Setups wird settings.out_dir bereits als ".../out/ap2" übergeben.
    Damit keine Doppelung ".../out/ap2/ap2" entsteht, prüfen wir den letzten
    Pfadbestandteil.
    """
    if base_out_dir.name.lower() == "ap2":
        return base_out_dir
    return base_out_dir / "ap2"


def _opt_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    v = str(value).strip()
    return Path(v) if v else None


def _root_out_dir(base_out_dir: Path) -> Path:
    """Bestimmt das 'Projekt-out' Verzeichnis.

    In manchen Setups ist settings.out_dir bereits .../out/ap2. Für Eingänge aus AP1
    brauchen wir dann das Parent-Verzeichnis (.../out).
    """
    return base_out_dir.parent if base_out_dir.name.lower() == "ap2" else base_out_dir


def _resolve_existing_input(
    p: Path,
    *,
    base_out_dir: Path,
    kind: str,
    glob_patterns: list[str],
    verbose: bool = False,
) -> Path:
    """Versucht fehlende Eingabepfade robust zu reparieren.

    Hauptfälle:
    1) Pfad existiert -> unverändert zurück.
    2) Doppeltes 'ap1/ap1' (oder ähnlich) in der Pipeline -> versuche bekannte Alternativen.
    3) Fallback: glob-Suche im out/ap1/* anhand von Patterns (neueste Datei gewinnt).
    """
    if p.exists():
        return p

    tried: list[Path] = []
    root = _root_out_dir(base_out_dir)

    # 2) Häufiger Pfadfehler: out/ap1/ap1/geometry_analysis statt out/ap1/geometry_analysis (oder umgekehrt)
    #    Wir versuchen beide Richtungen, sofern ein '.../ap1/geometry_analysis' Segment vorkommt.
    s = str(p)
    needle_a = str(root / "ap1" / "geometry_analysis")
    needle_b = str(root / "ap1" / "ap1" / "geometry_analysis")
    if needle_a in s:
        cand = Path(s.replace(needle_a, needle_b))
        tried.append(cand)
        if cand.exists():
            if verbose:
                print(f"[ap2] WARN: {kind} nicht gefunden, nutze stattdessen: {cand}")
            return cand
    if needle_b in s:
        cand = Path(s.replace(needle_b, needle_a))
        tried.append(cand)
        if cand.exists():
            if verbose:
                print(f"[ap2] WARN: {kind} nicht gefunden, nutze stattdessen: {cand}")
            return cand

    # 3) Glob-Fallback (suche unter out/ap1)
    ap1_dir = root / "ap1"
    matches: list[Path] = []
    if ap1_dir.exists():
        for pat in glob_patterns:
            matches.extend(Path(m) for m in glob.glob(str(ap1_dir / "**" / pat), recursive=True))

    matches = [m for m in matches if m.is_file()]
    if matches:
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        chosen = matches[0]
        if verbose:
            print(f"[ap2] WARN: {kind} nicht gefunden: {p}")
            if tried:
                print("[ap2]       geprüft:")
                for t in tried:
                    print(f"[ap2]         - {t}")
            print(f"[ap2]       glob-Fallback -> {chosen}")
        return chosen

    # Nichts gefunden -> harte Fehlermeldung mit Hinweisen
    msg = [f"{kind} nicht gefunden: {p}"]
    if tried:
        msg.append("Geprüfte Alternativen:")
        msg.extend([f"  - {t}" for t in tried])
    msg.append(f"Kein Treffer via Glob in: {ap1_dir}")
    msg.append("Hinweis: In deinem Log liegt die Geometrieanalyse offenbar unter out/ap1/ap1/geometry_analysis. ")
    raise FileNotFoundError("\n".join(msg))


def run(ctx: Any, verbose: bool = False) -> Path:
    """Führt die WP2/AP2-Pipeline aus und gibt den Pfad zur finalen GPKG zurück."""

    settings = ctx.settings
    data = settings.data

    # ---------------------------------------------------------------------
    # 0) Basisverzeichnisse
    # ---------------------------------------------------------------------
    base_out_dir = Path(settings.out_dir)
    ap2_out_dir = _resolve_ap2_dir(base_out_dir)
    ap2_out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Eingänge
    # ---------------------------------------------------------------------
    ap1_gpkg = Path(getattr(data, "ap1_gpkg"))
    zensus_csv = Path(getattr(data, "ap2_zensus_csv"))
    geom_csv = Path(getattr(data, "ap2_geom_csv"))

    # Robustheit: fehlende Eingänge reparieren (häufig: out/ap1/ap1/geometry_analysis vs out/ap1/geometry_analysis)
    ap1_gpkg = _resolve_existing_input(
        ap1_gpkg,
        base_out_dir=base_out_dir,
        kind="AP1-GPKG",
        glob_patterns=["*.gpkg"],
        verbose=verbose,
    )
    zensus_csv = _resolve_existing_input(
        zensus_csv,
        base_out_dir=base_out_dir,
        kind="Zensus-CSV",
        glob_patterns=["*enriched_zensus*.csv", "*zensus*.csv"],
        verbose=verbose,
    )
    geom_csv = _resolve_existing_input(
        geom_csv,
        base_out_dir=base_out_dir,
        kind="Geometrie-CSV",
        glob_patterns=["*geomstats_attributes.csv", "*geomstats*attributes*.csv"],
        verbose=verbose,
    )

    # ---------------------------------------------------------------------
    # 2) Ziele (können von CLI überschrieben werden)
    # ---------------------------------------------------------------------
    typed_gpkg = (
        _opt_path(getattr(data, "ap2_out_gpkg", None))
        or _opt_path(getattr(data, "out_gpkg", None))
        or (ap2_out_dir / "ap2_buildings_typed.gpkg")
    )

    heat_gpkg = (
        _opt_path(getattr(data, "ap2_heat_demand_gpkg", None))
        or (ap2_out_dir / "ap2_buildings_heat_demand.gpkg")
    )

    heat_refurb_gpkg = (
        _opt_path(getattr(data, "ap2_refurb_gpkg", None))
        or (ap2_out_dir / "ap2_buildings_heat_demand_refurb.gpkg")
    )

    final_gpkg = (
        _opt_path(getattr(data, "ap2_cold_demand_gpkg", None))
        or _opt_path(getattr(data, "ap2_demand_gpkg", None))
        or (ap2_out_dir / "ap2_buildings_heat_cold_demand.gpkg")
    )

    typed_gpkg.parent.mkdir(parents=True, exist_ok=True)
    heat_gpkg.parent.mkdir(parents=True, exist_ok=True)
    heat_refurb_gpkg.parent.mkdir(parents=True, exist_ok=True)
    final_gpkg.parent.mkdir(parents=True, exist_ok=True)

    # IWU Basisverzeichnis (erst settings, dann data, sonst Default)
    iwu_base_dir = (
        _opt_path(getattr(settings, "iwu_base_dir", None))
        or _opt_path(getattr(data, "iwu_base_dir", None))
        or rel_bedarfskennwerte_iwu_dir()
    )

    if verbose:
        print("[ap2] Starte AP2/WP2")
        print(f"[ap2] out_dir            : {ap2_out_dir}")
        print(f"[ap2] AP1-GPKG           : {ap1_gpkg}")
        print(f"[ap2] Zensus-CSV         : {zensus_csv}")
        print(f"[ap2] Geometrie-CSV      : {geom_csv}")
        print(f"[ap2] IWU base dir       : {iwu_base_dir}")
        print(f"[ap2] typed_gpkg         : {typed_gpkg}")
        print(f"[ap2] heat_gpkg          : {heat_gpkg}")
        print(f"[ap2] heat_refurb_gpkg   : {heat_refurb_gpkg}")
        print(f"[ap2] demand_gpkg (final): {final_gpkg}")

    # ---------------------------------------------------------------------
    # 3) Schritt 1/3: Gebäudetypisierung
    # ---------------------------------------------------------------------
    if verbose:
        print("[ap2] Schritt 1/3: Gebäudetypisierung (building_typing)")

    building_typing.run_building_typing(
        ap1_gpkg=ap1_gpkg,
        zensus_csv=zensus_csv,
        geom_csv=geom_csv,
        out_gpkg=typed_gpkg,
    )

    # ---------------------------------------------------------------------
    # 4) Schritt 2/3: Wärme + Trinkwarmwasser
    # ---------------------------------------------------------------------
    if verbose:
        print("[ap2] Schritt 2/3: Wärme + Trinkwarmwasser (heat_demand)")

    # IMPORTANT: out_dir an heat_demand ist Basis-out_dir, damit QA nach <out>/ap2/qa
    # geschrieben wird (nicht nach <out>/ap2/ap2/qa).
    heat_demand.compute_heat_demand_for_ap2(
        ap2_gpkg_path=typed_gpkg,
        iwu_base_dir=iwu_base_dir,
        out_gpkg_path=heat_gpkg,
        out_dir=base_out_dir,
        layer_name=None,
    )
    # ---------------------------------------------------------------------
    # 4b) Schritt 2.5/3: Sanierungsgrad-Korrektur (refurbishment)
    # ---------------------------------------------------------------------
    if verbose:
        print("[ap2] Schritt 2.5/3: Sanierungsgrad-Korrektur (refurbishment)")

    refurbishment.compute_refurbishment_for_ap2(
        ap2_heat_gpkg_path=heat_gpkg,
        out_gpkg_path=heat_refurb_gpkg,
        layer_name=None,
    )


    # ---------------------------------------------------------------------
    # 5) Schritt 3/3: Kälte
    # ---------------------------------------------------------------------
    if verbose:
        print("[ap2] Schritt 3/3: Kühlbedarf (cold_demand)")

    cold_demand.compute_cold_demand_for_ap2(
        ap2_gpkg_path=heat_refurb_gpkg,
        typed_gpkg_path=typed_gpkg,
        iwu_base_dir=iwu_base_dir,
        out_gpkg_path=final_gpkg,
    )

    # ---------------------------------------------------------------------
    # 6) Outputs im Context ablegen (praktisch für spätere Schritte)
    # ---------------------------------------------------------------------
    try:
        data.ap2_out_gpkg = str(typed_gpkg)
        data.ap2_heat_demand_gpkg = str(heat_gpkg)
        data.ap2_refurb_gpkg = str(heat_refurb_gpkg)
        data.ap2_cold_demand_gpkg = str(final_gpkg)
    except Exception:
        # Context muss nicht zwingend beschreibbar sein
        pass

    if verbose:
        print("[ap2] AP2/WP2 abgeschlossen.")

    return final_gpkg
