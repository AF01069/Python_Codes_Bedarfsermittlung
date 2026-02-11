"""
main.py

Batch-Entry-Point für die Workflows (Jobliste):
- AP1            (Datenaufnahme/Analyse LoD2, OSM, Basemap)
- AP1-Enrichment (Zensus, HK-Join, Analyse)
- AP2            (Gebäudetypisierung nach IWU)
- AP2-Climate    (Klimakorrektur Raumheizung: Standort vs. Referenz Potsdam)

Aktueller Modus:
- Wenn main.py ohne CLI-Args gestartet wird (z.B. PyCharm/Debug), wird IMMER
  die Jobliste (_build_jobs) ausgeführt.
- Wenn main.py mit CLI-Args gestartet wird (Shell), wird die normale CLI
  unverändert durchgereicht (cli_main(None)).

Hinweis:
- Die Jobliste ist für reproduzierbare Batch-Läufe gedacht und kann pro Stadt
  über Konstanten (LOD2_BASENAME / TEST_LOD2_PATH) angepasst werden.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

from kwp_bedarfskennwerte.cli import main as cli_main
from kwp_bedarfskennwerte.config.paths import (
    rel_data_dir,
    rel_adressen_dir,
    rel_geometrie_lod2_dir,
)

def _bool_env(name: str, default: bool = False) -> bool:
    """Liest boolesche ENV-Flags robust (1/true/yes/on)."""
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}



# ---------------------------------------------------------------------------
# 1) Projektpfade & Stadt-spezifische Konfiguration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(".")
DATA_DIR = rel_data_dir()
LOD2_DIR = rel_geometrie_lod2_dir()

# >>> HIER pro Stadt anpassen: LoD2-Basename (ohne Endung) – nur für relative Pfade
LOD2_BASENAME = os.environ.get("KWP_LOD2_BASENAME", "lod2_33498_5666_2_sn")

# Optional: absoluter Pfad (z.B. aus deiner Windows-Umgebung). Wenn gesetzt und existent, hat er Vorrang.
TEST_LOD2_PATH = Path(
    os.environ.get(
        "KWP_LOD2_PATH",
        str(LOD2_DIR / f"{LOD2_BASENAME}.shp"),
    )
)

TEST_META_CSV = Path(
    os.environ.get(
        "KWP_LOD2_META_CSV",
        str(LOD2_DIR / f"{LOD2_BASENAME}_akt.csv"),
    )
)

OUT_DIR = PROJECT_ROOT / "out"
OUT_AP1_DIR = OUT_DIR / "ap1"
OUT_AP1_GEOM_DIR = OUT_AP1_DIR / "geometry_analysis"
OUT_ZENSUS_DIR = OUT_AP1_DIR / "zensus"
OUT_AP2_DIR = OUT_DIR / "ap2"

AP1_GEOM_STATS_CSV = OUT_AP1_GEOM_DIR / f"{LOD2_BASENAME}_geomstats_attributes.csv"
ZENSUS_ENRICHED_CSV = OUT_ZENSUS_DIR / "ap1_buildings_enriched_zensus.csv"
AP1_BUILDINGS_ENRICHED_GPKG = OUT_ZENSUS_DIR / "ap1_buildings_enriched_zensus.gpkg"

AP2_TYPED_GPKG = OUT_AP2_DIR / "ap2_buildings_typed.gpkg"
AP2_HEAT_GPKG = OUT_AP2_DIR / "ap2_buildings_heat_demand.gpkg"
AP2_HEAT_CLIMATE_GPKG = OUT_AP2_DIR / "ap2_buildings_heat_demand_climate.gpkg"
AP2_HEAT_FORECAST2045_GPKG = OUT_AP2_DIR / "ap2_buildings_heat_demand_forecast2045.gpkg"

FALLBACK_BBOX_25833: Tuple[float, float, float, float] = (
    498000.0,
    5666000.0,
    500000.0,
    5668000.0,
)

DEFAULT_OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
DEFAULT_BASEMAP_MVT = (
    "https://sgx.geodatenzentrum.de/gdz_basemapde_vektor/tiles/v2/"
    "bm_web_de_3857/{z}/{x}/{y}.pbf"
)


# ---------------------------------------------------------------------------
# HK-DE Adressanreicherung (für Klimakorrektur / Auswertungen)
# ---------------------------------------------------------------------------
# Standard: HK-Join ist in ap1-enrich aktiviert (wenn eine HK-Datei gefunden wird).
# Du kannst hier optional explizit konfigurieren:
#
# - HK_ENABLE:   False setzt in ap1-enrich den Schalter --no-hk (Join deaktiviert)
# - HK_PATH:     expliziter Pfad zur HK-DE Datei (sonst Auto-Suche/ENV KWP_HK_PATH)
# - HK_PLACE_FILTER: optional nur bestimmte Orte (für Tests), z.B. ["Chemnitz"]
#
HK_ENABLE = True
HK_PATH = str(rel_adressen_dir() / "hk_sn_adressen_20250918.txt")
HK_PLACE_FILTER: List[str] = []


# ---------------------------------------------------------------------------
# 2) Hilfsfunktionen
# ---------------------------------------------------------------------------

def _read_bbox_from_csv(csv_path: Path) -> Tuple[float, float, float, float] | None:
    """
    Liest die BBOX aus einer CSV-Spalte 'Ausdehnung' (Format: "minx miny maxx maxy").
    Erwartet Semikolon als Trennzeichen.
    """
    if not csv_path.exists():
        return None

    import csv

    for enc in ("cp1252", "utf-8"):
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    raw = (row.get("Ausdehnung") or "").strip()
                    if not raw:
                        continue
                    parts = raw.replace(",", ".").split()
                    if len(parts) != 4:
                        continue
                    minx, miny, maxx, maxy = map(float, parts)
                    return (minx, miny, maxx, maxy)
        except Exception:
            continue
    return None


def _set_env_defaults_for_overpass() -> None:
    """
    Setzt robuste Defaults für den Overpass-Fetch, ohne neue CLI-Flags zu benötigen.
    Wird bei AP1 genutzt (AP2 braucht die Werte nicht, schadet aber nicht).
    """
    os.environ.setdefault(
        "KWP_OVERPASS_URLS",
        ",".join(
            [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter",
                "https://overpass.nchc.org.tw/api/interpreter",
            ]
        ),
    )
    os.environ.setdefault("KWP_OSM_DEG", "0.02")
    os.environ.setdefault("KWP_FAST", "1")


def _hk_cli_args() -> List[str]:
    """Baut optionale CLI-Argumente für HK-DE Adressanreicherung in ap1-enrich."""
    args: List[str] = []
    if not HK_ENABLE:
        return ["--no-hk"]

    hk_path = (HK_PATH or "").strip()
    if hk_path:
        args += ["--hk-path", hk_path]

    for place in HK_PLACE_FILTER:
        p = str(place).strip()
        if p:
            args += ["--hk-place", p]
    return args


# ---------------------------------------------------------------------------
# 3) Job-Liste für Batch-Ausführung
# ---------------------------------------------------------------------------

def _build_bbox() -> Tuple[float, float, float, float]:
    """
    Bestimmt die BBOX:
    - bevorzugt aus der Metadatei <LOD2_BASENAME>_akt.csv
    - sonst Fallback-BBOX.
    """
    bbox = _read_bbox_from_csv(TEST_META_CSV)
    if bbox is not None:
        return bbox
    return FALLBACK_BBOX_25833


def _build_jobs() -> List[List[str]]:
    """
    Baut eine Liste von Argumentlisten (argv), die nacheinander
    an die CLI übergeben werden.

    Konfiguriert:
    1) AP1                (Basislauf)
    2) AP1 --analyse      (Analyse-Run)
    3) AP1-Enrich         (Zensus + HK + Analyse)
    4) AP2                (IWU-Typisierung)
    5) AP2-Climate         (Standort-Klimakorrektur Raumheizung)
    6) Forecast 2045        (Sanierung/Leerstand-Scaffold + Klimawandel-Zeitfaktor)
    """
    if not TEST_LOD2_PATH.exists():
        raise FileNotFoundError(f"LoD2-Datei nicht gefunden: {TEST_LOD2_PATH}")

    minx, miny, maxx, maxy = _build_bbox()

    jobs: List[List[str]] = []

    # ---------------- AP1 ----------------
    jobs.append(
        [
            "ap1",
            "--verbose",
            "--lod2-path",
            str(TEST_LOD2_PATH),
            "--bbox",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "--overpass-url",
            os.environ.get("KWP_OVERPASS_URLS", DEFAULT_OVERPASS_URL),
            "--basemap-mvt-template",
            DEFAULT_BASEMAP_MVT,
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )

    # ---------------- AP1 (Analyse) ----------------
    jobs.append(
        [
            "ap1",
            "--analyse",
            "--verbose",
            "--lod2-path",
            str(TEST_LOD2_PATH),
            "--bbox",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "--overpass-url",
            os.environ.get("KWP_OVERPASS_URLS", DEFAULT_OVERPASS_URL),
            "--basemap-mvt-template",
            DEFAULT_BASEMAP_MVT,
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )

    # ---------------- AP1-Enrich ----------------
    # Optional: DIVIS überspringen (sehr zeitaufwendig)
    # Aktivieren via ENV: KWP_SKIP_DIVIS=1  (oder true/yes/on)
    skip_divis = _bool_env("KWP_SKIP_DIVIS", default=False)

    jobs.append(
        [
            "ap1-enrich",
            "--verbose",
            *(_hk_cli_args()),
            "--analyse",
            *(['--skip-divis'] if skip_divis else []),
            "--bbox",
            str(minx),
            str(miny),
            str(maxx),
            str(maxy),
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )


    # ---------------- AP2 – Gebäudetypisierung (IWU) ----------------
    jobs.append(
        [
            "ap2",
            "--verbose",
            "--ap1-gpkg",
            str(AP1_BUILDINGS_ENRICHED_GPKG),
            "--zensus-csv",
            str(ZENSUS_ENRICHED_CSV),
            "--geom-csv",
            str(AP1_GEOM_STATS_CSV),
            "--out-gpkg",
            str(AP2_TYPED_GPKG),
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )

    # ---------------- AP2-Climate – Standortkorrektur Raumheizung ----------------
    # Korrigiert die bereits berechneten Bedarfskennwerte (Heat-Demand).
    jobs.append(
        [
            "ap2-climate",
            "--verbose",
            "--input-gpkg",
            str(AP2_HEAT_GPKG),
            "--output-gpkg",
            str(AP2_HEAT_CLIMATE_GPKG),
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )

    # ---------------- Forecast 2045 ----------------
    jobs.append(
        [
            "forecast-2045",
            "--verbose",
            "--input-gpkg",
            str(AP2_HEAT_CLIMATE_GPKG),
            "--output-gpkg",
            str(AP2_HEAT_FORECAST2045_GPKG),
            # Klimawandel-Parameter (Default kann auch via ENV/CLI gesetzt werden)
            "--delta-t-climate-k",
            os.environ.get("KWP_DELTA_T_CLIMATE_K", "1.6"),
            # Optional: ΔT_ref direkt oder Ableitung aus HGT_ref/N_HP
            # "--delta-t-ref-k", "15.0",
            "--hgt-ref-kd",
            os.environ.get("KWP_HGT_REF_KD", "3000"),
            "--heating-period-days",
            os.environ.get("KWP_HEATING_PERIOD_DAYS", "200"),
            # Leerstand-Platzhalter (noch ohne Wirkung auf Q)
            # "--enable-vacancy",
            # "--target-vacancy-rate", "0.10",
            "--out-dir",
            str(OUT_DIR),
            "--cache-dir",
            "cache",
            "--work-dir",
            "work",
            "--target-epsg",
            "25833",
        ]
    )

    return jobs


# ---------------------------------------------------------------------------
# 4) Programmstart
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _set_env_defaults_for_overpass()

    if len(sys.argv) > 1:
        # Shell-Aufruf: echte sys.argv verwenden (klassischer CLI-Modus)
        cli_main(None)
        raise SystemExit(0)

    # Debug/Batch-Start ohne CLI-Args: Jobliste ausführen
    jobs = _build_jobs()

    for idx, argv in enumerate(jobs, start=1):
        print("\n" + "=" * 72)
        print(f"[kwp] Starte Job {idx}/{len(jobs)}: {' '.join(argv)}")
        print("=" * 72)
        cli_main(argv)
