"""
Kommandozeileninterface f?r die KWP-Workflows.

Enth?lt Subcommands f?r:
- AP1 (Geometrie-/Quellenaufnahme und Analyse)
- AP1-Enrichment (Zensus, GHS-OBAT, DIVIS, Adressen)
- AP2 (Typisierung, W?rme-/K?ltebedarfe)
- AP2-Climate (Klimakorrektur)
- Forecast 2045 (Szenariofortschreibung)

Args und Defaults sind auf reproduzierbare Batch-L?ufe ausgelegt.
"""
# kwp_bedarfskennwerte/cli.py
from __future__ import annotations

import os
import argparse
from types import SimpleNamespace
from pathlib import Path

# Pipelines
from .workflows.ap1_pipeline import ap1_pipeline
# Basemap-Konfig (wird in Settings abgelegt)
from .data_catalog.sources import BasemapCfg
# AP1 - Enrichment (DIVIS + Zensus + weitere Kennwerte)
from .workflows.ap1_enrich import run_enrichment
# from .workflows.ap1_enrich import run_enrichment_ghs_obat
# AP2 – Gebäudetypisierung (IWU)
from .workflows import ap2_pipeline
from .workflows.ap2_climate_correction import run_climate_correction



def _make_ctx(args) -> SimpleNamespace:
    """
    Baut ein minimales Context-Objekt mit .settings-Attribut,
    das von den Pipelines (AP1/AP2) und dem Enrichment (ap1-enrich) verwendet wird.
    """
    # Region (nur BBOX 25833; für ap1-enrich optional)
    bbox = getattr(args, "bbox", None)
    region = SimpleNamespace(
        bbox_25833=tuple(map(float, bbox)) if bbox else None
    )

    # Basemap-Konfiguration
    basemap_mvt_template = getattr(
        args,
        "basemap_mvt_template",
        "https://sgx.geodatenzentrum.de/gdz_basemapde_vektor/tiles/v2/bm_web_de_3857/{z}/{x}/{y}.pbf",
    )
    basemap = BasemapCfg(
        mvt_url_template=basemap_mvt_template,
        headers=None,
    )

    # Verzeichnisse
    out_dir_base = Path(getattr(args, "out_dir", "out"))
    cache_dir = Path(getattr(args, "cache_dir", "cache"))
    work_dir = Path(getattr(args, "work_dir", "work"))

    # AP-spezifische Output-Struktur:
    cmd = getattr(args, "cmd", None)
    if cmd in ("ap1", "ap1-enrich"):
        out_dir_effective = out_dir_base / "ap1"
    elif cmd == "ap2":
        out_dir_effective = out_dir_base / "ap2"
    else:
        out_dir_effective = out_dir_base

    # AP2: Default-Ausgabe-GPKG (falls nicht explizit gesetzt)
    ap2_out_gpkg_arg = getattr(args, "out_gpkg", None)
    ap2_out_gpkg = Path(ap2_out_gpkg_arg) if ap2_out_gpkg_arg else None
    if cmd == "ap2" and ap2_out_gpkg is None:
        ap2_out_gpkg = out_dir_effective / "ap2_buildings_typed.gpkg"

    # Datenpfade (AP1 + Enrichment + AP2)
    lod2_path = getattr(args, "lod2_path", None)
    divis_path = getattr(args, "divis_path", None)
    zensus_path = getattr(args, "zensus_path", None)

    # AP2-spezifische Pfade
    ap1_gpkg = getattr(args, "ap1_gpkg", None)
    ap2_zensus_csv = getattr(args, "zensus_csv", None)
    ap2_geom_csv = getattr(args, "geom_csv", None)

    data_cfg = SimpleNamespace(
        lod2_path=str(Path(lod2_path)) if lod2_path else None,
        divis_path=str(Path(divis_path)) if divis_path else None,
        zensus_path=str(Path(zensus_path)) if zensus_path else None,

        # AP2 – Typisierung
        ap1_gpkg=str(Path(ap1_gpkg)) if ap1_gpkg else None,
        ap2_zensus_csv=str(Path(ap2_zensus_csv)) if ap2_zensus_csv else None,
        ap2_geom_csv=str(Path(ap2_geom_csv)) if ap2_geom_csv else None,
        ap2_out_gpkg=str(ap2_out_gpkg) if ap2_out_gpkg else None,
    )

    target_epsg = int(getattr(args, "target_epsg", 25833))
    verbose = bool(getattr(args, "verbose", False))

    # Analyse-Schalter (Legacy-Flag: wird von main.py für ap1/ap1-enrich mitgegeben)
    analyse = bool(getattr(args, "analyse", False))

    # Enrichment-Steuerung für ap1-enrich
    enrich_only_divis = bool(getattr(args, "only_divis", False))
    enrich_only_zensus = bool(getattr(args, "only_zensus", False))
    # Für Kompatibilität: --analyse in ap1-enrich triggert auch die Enrichment-Auswertung
    enrich_analyse = analyse

    # HK-DE Adressanreicherung (für Klimakorrektur/PLZ etc.)
    enrich_with_hk_addresses = not bool(getattr(args, "no_hk", False))
    hk_path = getattr(args, "hk_path", None)
    hk_place_filter = getattr(args, "hk_place", None)
    hk_building_id_col = getattr(args, "hk_building_id_col", "LOD_UNITID")

    settings = SimpleNamespace(
        # Verzeichnisse
        out_dir=str(out_dir_effective),
        out_dir_base=str(out_dir_base),
        out_dir_ap1=str(out_dir_base / "ap1"),
        out_dir_ap2=str(out_dir_base / "ap2"),
        cache_dir=str(cache_dir),
        work_dir=str(work_dir),

        # Ziel-CRS
        target_epsg=target_epsg,

        # Schalter / Logging
        verbose=verbose,
        analyse=analyse,
        enrich_only_divis=enrich_only_divis,
        enrich_only_zensus=enrich_only_zensus,
        enrich_analyse=enrich_analyse,

        # HK-DE Adressen
        enrich_with_hk_addresses=enrich_with_hk_addresses,
        hk_path=hk_path,
        hk_place_filter=hk_place_filter,
        hk_building_id_col=hk_building_id_col,

        # Datenpfade
        data=data_cfg,

        # Quellen-Configs
        overpass_url=getattr(args, "overpass_url", None),
        basemap=basemap,

        # Region
        region=region,
    )

    # FAST-Modus ggf. zusätzlich per ENV setzen, damit auch tiefere Schichten ihn sehen
    if getattr(args, "fast", False):
        os.environ["KWP_FAST"] = "1"

    # optional per ENV: bestimmte Quellen deaktivieren
    if getattr(args, "no_osm", False):
        os.environ["KWP_NO_OSM"] = "1"
    if getattr(args, "no_basemap", False):
        os.environ["KWP_NO_BASEMAP"] = "1"

    return SimpleNamespace(settings=settings)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kwp_bedarfskennwerte",
        description="CLI für AP-Workflows (AP1 + Enrichment + AP2)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---------------- AP1 ----------------
    ap1 = sub.add_parser("ap1", help="AP1 – Datenaufnahme/Analyse (LoD2, Basemap, OSM)")
    ap1.add_argument(
        "--bbox",
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        type=float,
        required=False,
        help="BBOX in EPSG:25833",
    )
    ap1.add_argument("--lod2-path", required=False, help="Pfad zu LoD2/LoD1/Shape")
    ap1.add_argument(
        "--overpass-url",
        required=False,
        default="https://overpass.kumi.systems/api/interpreter",
        help=(
            "Overpass API URL – einzeln ODER kommagetrennt für Fallbacks "
            "(z. B. 'https://kumi...,...overpass-api.de/api/interpreter')"
        ),
    )
    ap1.add_argument(
        "--basemap-mvt-template",
        required=False,
        default=(
            "https://sgx.geodatenzentrum.de/gdz_basemapde_vektor/tiles/v2/"
            "bm_web_de_3857/{z}/{x}/{y}.pbf"
        ),
        help="Basemap.de MVT-Tile-Template",
    )
    ap1.add_argument("--out-dir", default="out")
    ap1.add_argument("--cache-dir", default="cache")
    ap1.add_argument("--work-dir", default="work")
    ap1.add_argument("--target-epsg", default=25833, type=int)
    ap1.add_argument("--verbose", action="store_true")
    ap1.add_argument(
        "--analyse",
        action="store_true",
        help="Nur Geometrieanalyse der Eingangs-LoD2-Datei ausführen (Legacy-Flag, kompatibel zu main.py).",
    )

    # Performance-Schalter
    ap1.add_argument(
        "--fast",
        action="store_true",
        help="Schneller Testmodus (weniger Tiles, Overpass out geom)",
    )
    ap1.add_argument(
        "--no-osm",
        action="store_true",
        help="OSM vorübergehend deaktivieren",
    )
    ap1.add_argument(
        "--no-basemap",
        action="store_true",
        help="Basemap vorübergehend deaktivieren",
    )

    # ---------------- AP1-Enrichment ----------------
    ap1_enrich = sub.add_parser(
        "ap1-enrich",
        help=(
            "AP1-Enrichment – Anreicherung des AP1-Gebäudelayers mit DIVIS- und "
            "Zensus-Informationen (Baujahr, Denkmalstatus, Leerstands-/Eigentumsquoten, "
            "Anlagen-/Gebäudekennwerte)"
        ),
    )

    # Akzeptiere Debug-Defaults aus main.py (werden nur teilweise genutzt)
    ap1_enrich.add_argument(
        "--bbox",
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        type=float,
        required=False,
        help="Optionale BBOX in EPSG:25833 (für DIVIS-WMS o. Debug).",
    )
    ap1_enrich.add_argument("--lod2-path", required=False, help="Kompatibilität (wird ignoriert).")
    ap1_enrich.add_argument("--overpass-url", required=False, help="Kompatibilität (wird ignoriert).")
    ap1_enrich.add_argument("--basemap-mvt-template", required=False, help="Kompatibilität (wird ignoriert).")

    ap1_enrich.add_argument(
        "--out-dir",
        default="out",
        help="Basis-Ausgabeverzeichnis (muss mit AP1-Ausgabe korrespondieren).",
    )
    ap1_enrich.add_argument("--cache-dir", default="cache", help="Cache-Verzeichnis (wie in AP1).")
    ap1_enrich.add_argument("--work-dir", default="work", help="Arbeitsverzeichnis (wie in AP1).")
    ap1_enrich.add_argument("--target-epsg", default=25833, type=int, help="Ziel-EPSG (wie in AP1).")
    ap1_enrich.add_argument("--verbose", action="store_true", help="Ausführliche Log-Ausgaben.")
    ap1_enrich.add_argument(
        "--analyse",
        action="store_true",
        help="Legacy-Flag (wird von main.py mitgegeben; triggert ggf. Enrichment-Auswertung).",
    )

    ap1_enrich.add_argument("--divis-path", required=False, help="Pfad zur DIVIS-Geodatei (optional).")

    ap1_enrich.add_argument(
        "--skip-divis",
        action="store_true",
        help="DIVIS-Enrichment überspringen (spart Zeit, wenn DIVIS nicht benötigt wird).",
    )
    ap1_enrich.add_argument("--zensus-path", required=False, help="Pfad zur vorbereiteten Zensus-100m-Geodatei (optional).")

    ap1_enrich.add_argument("--only-divis", action="store_true", help="Nur DIVIS-Enrichment ausführen.")
    ap1_enrich.add_argument("--only-zensus", action="store_true", help="Nur Zensus-Enrichment ausführen.")

    ap1_enrich.add_argument("--fast", action="store_true", help="Reserviert für spätere Performance-Optimierungen.")
    ap1_enrich.add_argument("--no-osm", action="store_true", help="Kompatibilität (wird ignoriert).")
    ap1_enrich.add_argument("--no-basemap", action="store_true", help="Kompatibilität (wird ignoriert).")
    ap1_enrich.add_argument("--only-obat", action="store_true", help="Nur GHS-OBAT-Anreicherung ausführen.")

    # --- HK-DE Adressanreicherung (für PLZ / Klimakorrektur) ---
    ap1_enrich.add_argument(
        "--hk-path",
        required=False,
        help=(
            "Pfad zur HK-DE Adressdatei (CSV/TXT, ';' getrennt). "
            "Wenn nicht gesetzt: Auto-Suche in <project>/Data/ bzw. <project>/data/ "
            "oder ENV KWP_HK_PATH."
        ),
    )
    ap1_enrich.add_argument(
        "--hk-place",
        action="append",
        default=None,
        help=(
            "Optionaler Ortsfilter (mehrfach nutzbar): filtert HK-DE Zeilen nach "
            "postonm (postalischer Ort) oder gmd (Gemeinde). Beispiel: --hk-place Chemnitz"
        ),
    )
    ap1_enrich.add_argument(
        "--hk-building-id-col",
        default="LOD_UNITID",
        help="Join-Key im Gebäudelayer (Default: LOD_UNITID) für HK-DE Join gegen oid.",
    )
    ap1_enrich.add_argument(
        "--no-hk",
        action="store_true",
        help="HK-DE Adressanreicherung deaktivieren (standardmäßig aktiv).",
    )


    # ---------------- AP2 ----------------
    ap2 = sub.add_parser(
        "ap2",
        help=(
            "AP2 – Gebäudetypisierung (IWU) auf Basis des angereicherten "
            "AP1-Gebäudelayers und der ergänzenden Zensus-/Geometrie-CSV."
        ),
    )
    ap2.add_argument("--ap1-gpkg", required=True, help="Pfad zum angereicherten AP1-Gebäudelayer (GPKG).")
    ap2.add_argument("--zensus-csv", required=True, help="Pfad zur Zensus-Attribut-CSV.")
    ap2.add_argument("--geom-csv", required=True, help="Pfad zur Geometrie-Analyse-CSV.")
    ap2.add_argument("--out-gpkg", required=False, help="Pfad zur Ausgabe-GPKG. Default: <out-dir>/ap2/ap2_buildings_typed.gpkg")
    ap2.add_argument("--out-dir", default="out")
    ap2.add_argument("--cache-dir", default="cache")
    ap2.add_argument("--work-dir", default="work")
    ap2.add_argument("--target-epsg", default=25833, type=int)
    ap2.add_argument("--verbose", action="store_true")
    # ---------------- AP2-Klimakorrektur ----------------

    climate = sub.add_parser(
        "ap2-climate",
        help=(
            "AP2-Klimakorrektur – standortbezogene Anpassung der Raumheizungs-"
            "Bedarfskennwerte anhand der DWD-Klimafaktoren (Referenz Potsdam)."
        ),
    )

    # Für Kompatibilität mit main.py / _make_ctx akzeptieren wir die üblichen Standard-Args
    climate.add_argument("--bbox", nargs=4, type=float, required=False)
    climate.add_argument("--lod2-path", required=False)
    climate.add_argument("--overpass-url", required=False)
    climate.add_argument("--basemap-mvt-template", required=False)

    climate.add_argument("--out-dir", default="out")
    climate.add_argument("--cache-dir", default="cache")
    climate.add_argument("--work-dir", default="work")
    climate.add_argument("--target-epsg", default=25833, type=int)
    climate.add_argument("--verbose", action="store_true")

    climate.add_argument("--input-gpkg", required=True)
    climate.add_argument("--output-gpkg", required=False, default=None)
    climate.add_argument(
        "--layer",
        action="append",
        default=None,
        help="Zu bearbeitende Layer (mehrfach möglich). Wenn nicht gesetzt: alle Layer.",
    )
    climate.add_argument(
        "--no-keep-ref",
        action="store_true",
        help="Keine *_ref Sicherungsspalten anlegen.",
    )

    climate.add_argument("--fast", action="store_true")
    climate.add_argument("--no-osm", action="store_true")
    climate.add_argument("--no-basemap", action="store_true")

    # ---------------- Forecast 2045 ----------------
    fc2045 = sub.add_parser(
        "forecast-2045",
        help=(
            "Forecast 2045 – Szenario-Transformation (Sanierung/Leerstand/"
            "Klimawandel-Faktor) auf Basis der AP2-Heat-Demand GPKG. "
            "Schreibt zusätzliche *_2045 Spalten und Metadatenfelder."
        ),
    )

    # Standard-Args (Kompatibilität zu main.py/_make_ctx)
    fc2045.add_argument("--bbox", nargs=4, type=float, required=False)
    fc2045.add_argument("--lod2-path", required=False)
    fc2045.add_argument("--overpass-url", required=False)
    fc2045.add_argument("--basemap-mvt-template", required=False)

    fc2045.add_argument("--out-dir", default="out")
    fc2045.add_argument("--cache-dir", default="cache")
    fc2045.add_argument("--work-dir", default="work")
    fc2045.add_argument("--target-epsg", default=25833, type=int)
    fc2045.add_argument("--verbose", action="store_true")

    fc2045.add_argument("--input-gpkg", required=True)
    fc2045.add_argument("--output-gpkg", required=False, default=None)
    fc2045.add_argument(
        "--layer",
        action="append",
        default=None,
        help="Zu bearbeitende Layer (mehrfach möglich). Wenn nicht gesetzt: alle Layer.",
    )

    # Klimawandel-Faktor (Zeitfaktor)
    fc2045.add_argument(
        "--delta-t-climate-k",
        type=float,
        default=1.6,
        help="Temperaturerhöhung ΔT_Klima bis 2045 in Kelvin (Default 1.6).",
    )
    fc2045.add_argument(
        "--delta-t-ref-k",
        type=float,
        default=None,
        help=(
            "Mittlere Temperaturdifferenz ΔT_ref in der Heizperiode des Referenzklimas (K). "
            "Wenn nicht gesetzt: wird aus --hgt-ref-kd / --heating-period-days abgeleitet."
        ),
    )
    fc2045.add_argument(
        "--hgt-ref-kd",
        type=float,
        default=3000.0,
        help="Referenz-Heizgradtage HGT_ref des Referenzklimas (K·d) (Default 3000).",
    )
    fc2045.add_argument(
        "--heating-period-days",
        type=float,
        default=200.0,
        help="Heizperiodendauer N_HP (Tage) zur Ableitung ΔT_ref (Default 200).",
    )

    # Platzhalter-Module aktivieren/deaktivieren
    fc2045.add_argument(
        "--no-refurb",
        action="store_true",
        help="Refurbishment/Sanierungsmodulation deaktivieren.",
    )
    fc2045.add_argument(
        "--enable-vacancy",
        action="store_true",
        help="Leerstands-/Occupancy-Platzhalter aktivieren (noch ohne Wirkung auf Q).",
    )
    fc2045.add_argument(
        "--target-vacancy-rate",
        type=float,
        default=0.10,
        help="Ziel-Leerstandsquote 2045 (0..1) für Platzhalter (Default 0.10).",
    )

    # ---------------- AP2-Klimakorrektur ----------------

    climate = sub.add_parser(
        "ap2-climate",
        help=(
            "AP2-Klimakorrektur – standortbezogene Anpassung der Raumheizungs-"
            "Bedarfskennwerte anhand der DWD-Klimafaktoren (Referenz Potsdam)."
        ),
    )

    # Für Kompatibilität mit main.py / _make_ctx akzeptieren wir die üblichen Standard-Args
    climate.add_argument("--bbox", nargs=4, type=float, required=False)
    climate.add_argument("--lod2-path", required=False)
    climate.add_argument("--overpass-url", required=False)
    climate.add_argument("--basemap-mvt-template", required=False)

    climate.add_argument("--out-dir", default="out")
    climate.add_argument("--cache-dir", default="cache")
    climate.add_argument("--work-dir", default="work")
    climate.add_argument("--target-epsg", default=25833, type=int)
    climate.add_argument("--verbose", action="store_true")

    climate.add_argument("--input-gpkg", required=True)
    climate.add_argument("--output-gpkg", required=False, default=None)
    climate.add_argument(
        "--layer",
        action="append",
        default=None,
        help="Zu bearbeitende Layer (mehrfach möglich). Wenn nicht gesetzt: alle Layer.",
    )
    climate.add_argument(
        "--no-keep-ref",
        action="store_true",
        help="Keine *_ref Sicherungsspalten anlegen.",
    )

    climate.add_argument("--fast", action="store_true")
    climate.add_argument("--no-osm", action="store_true")
    climate.add_argument("--no-basemap", action="store_true")

    return p


def main(argv=None):
    """
    CLI-Entry: parst Args und startet die gewünschte Pipeline.

    - Wenn argv=None: echte sys.argv verwenden (Standard)
    - Wenn argv=list[str]: diese Liste als Argumente nutzen (Debug-Start aus main.py)
    """
    parser = _build_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(argv)

    ctx = _make_ctx(args)

    try:
        if args.cmd == "ap1":
            return ap1_pipeline.run(ctx)

        if args.cmd == "ap1-enrich":
            verbose = bool(getattr(args, "verbose", False))
            only_obat = bool(getattr(args, "only_obat", False))
            if only_obat:
                return run_enrichment_ghs_obat(ctx, verbose=verbose)
            # skip_divis ist optional; wenn nicht gesetzt, bleibt es False.
            return run_enrichment(
                ctx,
                verbose=verbose,
                skip_divis=bool(getattr(args, "skip_divis", False)),
            )

        if args.cmd == "ap2":
            return ap2_pipeline.run(ctx, verbose=bool(getattr(args, "verbose", False)))

        if args.cmd == "ap2-climate":
            verbose = bool(getattr(args, "verbose", False))
            input_gpkg = getattr(args, "input_gpkg")
            output_gpkg = getattr(args, "output_gpkg", None)
            layers = getattr(args, "layer", None)
            keep_ref = not bool(getattr(args, "no_keep_ref", False))

            res = run_climate_correction(
                ctx,
                input_gpkg=input_gpkg,
                output_gpkg=output_gpkg,
                layers=layers,
                keep_ref_columns=keep_ref,
                verbose=verbose,
            )

        if args.cmd == "forecast-2045":
            from kwp_bedarfskennwerte.methodology.forecast_2045 import (
                ClimateFutureConfig,
                Forecast2045Config,
                RefurbishmentFutureConfig,
                VacancyFutureConfig,
                run_forecast_2045,
            )

            verbose = bool(getattr(args, "verbose", False))
            input_gpkg = getattr(args, "input_gpkg")
            output_gpkg = getattr(args, "output_gpkg", None)
            layers = getattr(args, "layer", None)

            climate_cfg = ClimateFutureConfig(
                target_year=2045,
                delta_t_ref_k=getattr(args, "delta_t_ref_k", None),
                hgt_ref_kd=float(getattr(args, "hgt_ref_kd", 3000.0)),
                heating_period_days=float(getattr(args, "heating_period_days", 200.0)),
                delta_t_climate_k=float(getattr(args, "delta_t_climate_k", 1.6)),
            )

            refurb_cfg = RefurbishmentFutureConfig(enabled=not bool(getattr(args, "no_refurb", False)))
            vacancy_cfg = VacancyFutureConfig(
                enabled=bool(getattr(args, "enable_vacancy", False)),
                target_vacancy_rate=float(getattr(args, "target_vacancy_rate", 0.10)),
            )

            cfg = Forecast2045Config(climate=climate_cfg, refurb=refurb_cfg, vacancy=vacancy_cfg)

            res = run_forecast_2045(
                ctx,
                input_gpkg=input_gpkg,
                output_gpkg=output_gpkg,
                layers=layers,
                cfg=cfg,
                verbose=verbose,
            )
        return res
    except KeyboardInterrupt:
        print("[kwp] Abbruch durch Nutzer (Ctrl+C).")
        return 130

    parser.error(f"Unbekanntes Kommando: {args.cmd}")


if __name__ == "__main__":
    main(None)
