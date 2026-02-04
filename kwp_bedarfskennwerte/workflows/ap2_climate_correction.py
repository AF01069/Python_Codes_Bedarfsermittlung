"""
AP2 – Klimakorrektur (Standortkorrektur, Referenz Potsdam)

Dieses Modul nimmt eine Geopackage-Datei (typischerweise AP2-Ergebnis) und
korrigiert raumheizungsbezogene Bedarfskennwerte auf Basis der DWD-Klimafaktoren
(Referenz: TRY Potsdam).

Es wird **nur** Raumheizung korrigiert; Trinkwarmwasser (TWW/DHW) und Kühlung
werden nicht angepasst.

Die DWD-Klimafaktoren sind definiert als:
    KF = G(TRY, Potsdam) / G(Standort)

Da IWU-Bedarfskennwerte auf dem Referenzklima basieren, wird für die
Übertragung auf den Standort verwendet:
    f_loc = 1 / KF
    Q_RH_loc = Q_RH_ref * f_loc

Die Routine versucht den Klimafaktor über eine PLZ-Spalte im Layer zuzuordnen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd

from ..config.runtime import PipelineContext
from ..utils.climate import (
    ensure_dwd_kf_dataset,
    load_kf_mapping,
    normalize_plz,
)


PLZ_CANDIDATES: Tuple[str, ...] = (
    "plz",
    "PLZ",
    "postcode",
    "Postleitzahl",
    "addr_postcode",
    "ADDR_POSTCODE",
    "OSM_PLZ",
    "osm_plz",
    # HK-DE (v5.2) Standardspalte aus addresses_hk.py
    "HK_postplz",
)

_INCLUDE_PATTERNS = (
    r"(^|_)RH(_|$)",
    r"raumw",
    r"raumwaerme",
    r"heiz",
    r"space.->heat",
    r"(^|_)QH(_|$)",
)

_EXCLUDE_PATTERNS = (
    r"tww",
    r"dhw",
    r"warmwasser",
    r"rcool",
    r"cool",
    r"kalt",
)


def _get_settings(ctx: PipelineContext):
    return getattr(ctx, "settings", ctx)


def _detect_plz_column(gdf: gpd.GeoDataFrame) -> Optional[str]:
    cols = set(gdf.columns)
    for c in PLZ_CANDIDATES:
        if c in cols:
            return c
    # fuzzy
    lower_map = {str(c).lower(): c for c in gdf.columns}
    for cand in ("plz", "postleitzahl", "postcode", "zip"):
        if cand in lower_map:
            return lower_map[cand]
    return None


def _is_heating_column(col: str) -> bool:
    s = str(col)
    sl = s.lower()

    for pat in _EXCLUDE_PATTERNS:
        if pd.notna(s) and pd.Series([sl]).str.contains(pat, case=False, regex=True).iloc[0]:
            return False

    for pat in _INCLUDE_PATTERNS:
        if pd.notna(s) and pd.Series([sl]).str.contains(pat, case=False, regex=True).iloc[0]:
            return True

    if any(k in sl for k in ("kwh_m2a", "kwh_a")) and any(k in sl for k in ("heat", "heiz", "rh")):
        return True

    return False


def _numeric_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    out: List[str] = []
    for c in gdf.columns:
        if c == "geometry":
            continue
        if pd.api.types.is_numeric_dtype(gdf[c]):
            out.append(c)
    return out


def _apply_factor_to_columns(
    gdf: gpd.GeoDataFrame,
    factor: np.ndarray,
    keep_ref_columns: bool = True,
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    touched: List[str] = []
    numeric_cols = _numeric_columns(gdf)
    for col in numeric_cols:
        if not _is_heating_column(col):
            continue

        ref_col = f"{col}_ref"
        if keep_ref_columns and ref_col not in gdf.columns:
            gdf[ref_col] = gdf[col]

        gdf[col] = gdf[col].astype(float) * factor
        touched.append(col)

    return gdf, touched


def run_climate_correction(
    ctx: PipelineContext,
    input_gpkg: str,
    output_gpkg: Optional[str] = None,
    layers: Optional[Sequence[str]] = None,
    keep_ref_columns: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Klimakorrigiert eine GPKG (alle oder ausgewählte Layer).

    - DWD-KF wird einmalig geladen (cache_dir/dwd_kf/)
    - Pro Layer wird per PLZ-Spalte ein Faktor 1/KF zugeordnet
    - Raumheizungsbezogene numerische Spalten werden korrigiert
      (Heuristik; TWW/DHW/Cooling wird ausgeschlossen)
    """
    settings = _get_settings(ctx)
    cache_dir = Path(getattr(settings, "cache_dir", "cache"))

    ds, local_csv = ensure_dwd_kf_dataset(cache_dir=cache_dir)
    mapping = load_kf_mapping(local_csv)

    input_gpkg = str(Path(input_gpkg))
    if output_gpkg is None:
        p = Path(input_gpkg)
        output_gpkg = str(p.with_name(p.stem + "_climate.gpkg"))

    # Layerliste
    all_layers = fiona.listlayers(input_gpkg)
    target_layers = list(all_layers) if not layers else [l for l in layers if l in all_layers]
    if not target_layers:
        raise RuntimeError(f"Keine passenden Layer gefunden. Verfügbar: {all_layers}")

    # Output neu schreiben
    out_path = Path(output_gpkg)
    if out_path.exists():
        out_path.unlink()

    summary = {
        "input_gpkg": input_gpkg,
        "output_gpkg": str(out_path),
        "dwd_dataset": ds.filename,
        "dwd_period": ds.period,
        "layers": [],
    }

    for lyr in target_layers:
        gdf = gpd.read_file(input_gpkg, layer=lyr)
        if gdf.empty:
            if verbose:
                print(f"[climate] Layer '{lyr}' ist leer – wird unverändert geschrieben.")
            gdf.to_file(out_path, layer=lyr, driver="GPKG")
            summary["layers"].append(
                {"layer": lyr, "n": 0, "plz_col": None, "unique_plz": 0, "touched_cols": []}
            )
            continue

        plz_col = _detect_plz_column(gdf)
        if plz_col is None:
            raise RuntimeError(
                f"Keine PLZ-Spalte in Layer '{lyr}' gefunden. "
                f"Erwartet z. B. eine der Spalten: {PLZ_CANDIDATES}"
            )

        plz_norm = gdf[plz_col].apply(normalize_plz)
        gdf["_CLIMATE_PLZ"] = plz_norm

        uniq = sorted([p for p in plz_norm.dropna().unique().tolist() if p])
        if not uniq:
            raise RuntimeError(f"Layer '{lyr}': keine gültigen PLZ-Werte in Spalte '{plz_col}'.")

        # Faktorvektor je Feature
        kf = plz_norm.map(mapping).astype(float)

        # Fallback: falls einzelne PLZ nicht im DWD-Mapping sind -> median der vorhandenen
        if kf.isna().any():
            med = float(np.nanmedian(kf.values))
            kf = kf.fillna(med)
            if verbose:
                missing_n = int(kf.isna().sum())
                print(f"[climate] Layer '{lyr}': {missing_n} PLZ ohne KF – Fallback median={med:.3f}")

        # Standortfaktor: 1/KF
        factor = (1.0 / kf.values.astype(float))

        gdf["climate_kf"] = kf.values.astype(float)
        gdf["climate_factor"] = factor
        gdf["climate_source"] = "DWD_climate_correction_factor_recent"
        gdf["climate_period"] = ds.period

        gdf, touched_cols = _apply_factor_to_columns(
            gdf=gdf,
            factor=factor,
            keep_ref_columns=keep_ref_columns,
        )

        gdf.to_file(out_path, layer=lyr, driver="GPKG")

        summary["layers"].append(
            {
                "layer": lyr,
                "n": int(len(gdf)),
                "plz_col": plz_col,
                "unique_plz": int(len(uniq)),
                "touched_cols": touched_cols,
            }
        )

        if verbose:
            print(
                f"[climate] Layer '{lyr}': n={len(gdf)}, plz_col={plz_col}, "
                f"unique_plz={len(uniq)}, touched_cols={len(touched_cols)}"
            )

    return summary
