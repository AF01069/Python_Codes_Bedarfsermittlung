"""kwp_bedarfskennwerte.methodology.forecast_2045

Forecast / Szenario 2045.

Dieses Modul implementiert eine **kompositionelle** Szenario-Transformation für
Bedarfskennwerte (Kennwertverfahren). Es bündelt drei Themenblöcke:

1) **Sanierung / Modernisierung** (Refurbishment)
   - nutzt die vorhandene Modulation des impliziten Sanierungsanteils \u03c6 -> \u03c6'
     und die Ansatz-1-Skalierung aus :mod:`kwp_bedarfskennwerte.methodology.refurbishment`.

2) **Leerstand / Nutzungsgrad** (Vacancy / occupancy)
   - ist als *Struktur* vorgesehen (Platzhalter). In der 2045-Prognose soll
     Leerstand perspektivisch über ein Nutzungs-/Beheizungsgrad-Multiplikator
     in die *absoluten* Jahresbedarfe (kWh/a) eingehen.

3) **Klimawandel** (Time climate factor 2045)
   - implementiert den im Bericht beschriebenen Ansatz über einen *zeitlichen*
     Klimawandel-Faktor, der als Verhältnis zweier Heizgradtagszahlen definiert
     ist und mit dem bereits vorhandenen *räumlichen* Standortfaktor
     (DWD-Klimakorrektur, Referenz Potsdam) kombiniert werden kann.

Wichtig
-------
Dieses Modul ist absichtlich **TRY-unabhängig**. Es liefert aber einen
Ankerpunkt, um später Heizgradtage direkt aus Zukunfts-TRY (2031-2060) abzuleiten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config.runtime import PipelineContext
from .refurbishment import FieldMap as RefurbFieldMap
from .refurbishment import RefurbishmentConfig, apply_refurbishment


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------


@dataclass
class ClimateFutureConfig:
    """Parameter zur Bestimmung des zeitlichen Klimawandel-Faktors (2045).

    Implementiert die im Bericht verwendete Beziehung:

        f_klima(2045) = HGT_2045 / HGT_ref
                     = (\u0394T_ref - \u0394T_klima) / \u0394T_ref

    wobei \u0394T_ref die mittlere Temperaturdifferenz in der Heizperiode des
    Referenzklimas (Potsdam) repräsentiert.

    Da im Kennwertmodell keine TRY-Zeitreihen verwendet werden, wird \u0394T_ref
    als konfigurierbarer Parameter gesetzt. Alternativ kann \u0394T_ref aus
    HGT_ref / N_HP abgeleitet werden.
    """

    target_year: int = 2045

    # Option A: direkt als mittlere Temperaturdifferenz in der Heizperiode (K)
    delta_t_ref_k: Optional[float] = None

    # Option B: Ableitung aus HGT_ref und Heizperiodendauer (Tage)
    hgt_ref_kd: float = 3000.0
    heating_period_days: float = 200.0

    # \u0394T_Klima (Erwärmung bis Zieljahr relativ zum Referenzklima), in Kelvin
    # Default: konservativer Mittelwert; im Projekt i.d.R. über Szenarioparameter
    # gesetzt (z. B. Band +1.0..+2.5 K für 2021-2050 in Sachsen).
    delta_t_climate_k: float = 1.6

    # Untergrenze, damit f nicht negativ wird (numerische Sicherheit)
    min_factor: float = 0.05


@dataclass
class VacancyFutureConfig:
    """Platzhalter für Leerstands-/Nutzungsgradmodellierung.

    Idee (später):
      - occupancy_factor in 0..1
      - absolute Jahresbedarfe (Q_*_kWh_a) werden mit occupancy_factor skaliert
      - spezifische Kennwerte (kWh/m²a) bleiben unverändert.
    """

    enabled: bool = False
    # z. B. Ziel-Leerstand 2045 (0..1); später regional differenziert
    target_vacancy_rate: float = 0.10


@dataclass
class RefurbishmentFutureConfig:
    """Konfiguration der Sanierungsmodulation.

    Der bestehende Refurbishment-Ansatz moduliert \u03c6 objekt-/standortspezifisch
    aus Denkmal, Leerstand, Eigentum und skaliert RW-Kennwerte.

    Für 2045 soll perspektivisch zusätzlich eine *zeitliche Sanierungsdynamik*
    (Sanierungsquoten) abgebildet werden. Das ist hier als Struktur vorgesehen.
    """

    enabled: bool = True
    cfg: RefurbishmentConfig = RefurbishmentConfig()
    fm: RefurbFieldMap = RefurbFieldMap()

    # Platzhalter: jährliche Sanierungsquote, Zieljahr-Pfad, etc.
    # Diese Parameter werden später genutzt, um \u03c6' weiter in Richtung eines
    # Szenariopfades zu verschieben.
    annual_refurb_rate_pct: Optional[float] = None


@dataclass
class Forecast2045Config:
    """
    Datenklasse f?r forecast2045 config.
    """
    climate: ClimateFutureConfig = ClimateFutureConfig()
    vacancy: VacancyFutureConfig = VacancyFutureConfig()
    refurb: RefurbishmentFutureConfig = RefurbishmentFutureConfig()

    # Wenn True: bestehende Werte werden in *_<year> geschrieben und Basis bleibt erhalten.
    # Wenn False: Werte werden überschrieben (nicht empfohlen).
    write_year_suffix_columns: bool = True

    # Namenskonventionen
    col_factor_time: str = "climate_factor_time"
    col_factor_total: str = "climate_factor_total"
    col_target_year: str = "forecast_year"


# -----------------------------------------------------------------------------
# Klimawandel-Faktor (Zeit)
# -----------------------------------------------------------------------------


def compute_time_climate_factor(cfg: ClimateFutureConfig) -> float:
    """Berechnet den zeitlichen Klimawandel-Faktor f_klima(target_year).

    f = (\u0394T_ref - \u0394T_klima) / \u0394T_ref

    \u0394T_ref wird entweder direkt angegeben oder aus HGT_ref/N_HP abgeleitet.
    """

    if cfg.delta_t_ref_k is not None:
        delta_ref = float(cfg.delta_t_ref_k)
    else:
        # Ableitung aus HGT_ref / N_HP
        n_hp = float(cfg.heating_period_days)
        if n_hp <= 0:
            raise ValueError("heating_period_days muss > 0 sein")
        delta_ref = float(cfg.hgt_ref_kd) / n_hp

    if delta_ref <= 0:
        raise ValueError("delta_t_ref_k (oder hgt_ref_kd/heating_period_days) muss > 0 sein")

    delta_clim = float(cfg.delta_t_climate_k)
    f = (delta_ref - delta_clim) / delta_ref
    return float(max(cfg.min_factor, min(1.5, f)))


# -----------------------------------------------------------------------------
# Anwendung auf DataFrame / GPKG
# -----------------------------------------------------------------------------


def _is_abs_energy_column(name: str) -> bool:
    s = str(name)
    sl = s.lower()
    if sl.endswith("_kwh_a") or "kwh_a" in sl:
        return True
    return False


def _is_rw_column(name: str) -> bool:
    """Heuristik: Raumwärme-Spalten (RW/RH/QH) – TWW/DHW ausschließen."""
    sl = str(name).lower()
    if any(k in sl for k in ("tww", "dhw", "warmwasser")):
        return False
    if any(k in sl for k in ("rw", "rh", "raumw", "raumwaerme", "spaceheat", "qh")):
        return True
    return False


def apply_time_climate_to_df(
    df: pd.DataFrame,
    *,
    cfg: Forecast2045Config,
    verbose: bool = False,
) -> pd.DataFrame:
    """Wendet den zeitlichen Klimawandel-Faktor auf Raumwärme-Spalten an.

    Erwartung:
    - Standortkorrektur (DWD) ist optional bereits enthalten; wenn Spalte
      'climate_factor' vorhanden ist, wird sie zur QA in einen Gesamtfaktor
      übernommen.
    - Es werden ausschließlich RW/RH-Spalten korrigiert; TWW bleibt unverändert.
    - Wenn 'write_year_suffix_columns=True', werden neue Spalten *_<year>
      geschrieben und die Originale bleiben erhalten.
    """

    out = df.copy()
    year = int(cfg.climate.target_year)
    f_time = compute_time_climate_factor(cfg.climate)

    # Standortfaktor, falls vorhanden (aus ap2-climate-correction)
    f_loc = None
    if "climate_factor" in out.columns:
        try:
            f_loc = pd.to_numeric(out["climate_factor"], errors="coerce")
        except Exception:
            f_loc = None

    out[cfg.col_target_year] = year
    out[cfg.col_factor_time] = float(f_time)
    if f_loc is not None:
        out[cfg.col_factor_total] = (f_loc.astype(float) * float(f_time)).astype(float)
    else:
        out[cfg.col_factor_total] = float(f_time)

    # Korrektur anwenden
    touched = 0
    for col in list(out.columns):
        if col == "geometry":
            continue
        if not _is_rw_column(col):
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue

        new_col = f"{col}_{year}" if cfg.write_year_suffix_columns else col
        if cfg.write_year_suffix_columns and new_col in out.columns:
            # bereits vorhanden -> überspringen
            continue

        out[new_col] = pd.to_numeric(out[col], errors="coerce") * float(f_time)
        touched += 1

    # HTW neu bilden, falls passende Spalten existieren (RW+TWW)
    # Spezifisch
    for base_rw, base_tww, out_htw in (
        ("spec_RW_NE_kWh_m2a", "spec_TWW_NE_kWh_m2a", "spec_HTW_NE_kWh_m2a"),
        ("spec_RW_END_kWh_m2a", "spec_TWW_END_kWh_m2a", "spec_HTW_END_kWh_m2a"),
        ("spec_RW_NE_refurb_kWh_m2a", "spec_TWW_NE_kWh_m2a", "spec_HTW_NE_refurb_kWh_m2a"),
        ("spec_RW_END_refurb_kWh_m2a", "spec_TWW_END_kWh_m2a", "spec_HTW_END_refurb_kWh_m2a"),
    ):
        rw_col = f"{base_rw}_{year}" if cfg.write_year_suffix_columns else base_rw
        if rw_col in out.columns and base_tww in out.columns:
            htw_col = f"{out_htw}_{year}" if cfg.write_year_suffix_columns else out_htw
            out[htw_col] = (
                pd.to_numeric(out[rw_col], errors="coerce")
                + pd.to_numeric(out[base_tww], errors="coerce")
            )

    # Absolut
    for base_rw, base_tww, out_htw in (
        ("Q_RW_NE_kWh_a", "Q_TWW_NE_kWh_a", "Q_HTW_NE_kWh_a"),
        ("Q_RW_END_kWh_a", "Q_TWW_END_kWh_a", "Q_HTW_END_kWh_a"),
        ("Q_RW_NE_refurb_kWh_a", "Q_TWW_NE_kWh_a", "Q_HTW_NE_refurb_kWh_a"),
        ("Q_RW_END_refurb_kWh_a", "Q_TWW_END_kWh_a", "Q_HTW_END_refurb_kWh_a"),
    ):
        rw_col = f"{base_rw}_{year}" if cfg.write_year_suffix_columns else base_rw
        if rw_col in out.columns and base_tww in out.columns:
            htw_col = f"{out_htw}_{year}" if cfg.write_year_suffix_columns else out_htw
            out[htw_col] = (
                pd.to_numeric(out[rw_col], errors="coerce")
                + pd.to_numeric(out[base_tww], errors="coerce")
            )

    if verbose:
        print(f"[forecast2045] climate_time_factor={f_time:.4f}, touched_cols={touched}")

    return out


def apply_vacancy_placeholder(df: pd.DataFrame, cfg: VacancyFutureConfig) -> pd.DataFrame:
    """Platzhalter: schreibt occupancy/vacancy Zielwerte (noch ohne Wirkung)."""
    out = df.copy()
    if not cfg.enabled:
        out["vacancy_model_enabled"] = False
        return out

    out["vacancy_model_enabled"] = True
    out["vacancy_rate_target"] = float(cfg.target_vacancy_rate)
    out["occupancy_factor"] = float(max(0.0, min(1.0, 1.0 - cfg.target_vacancy_rate)))
    return out


def apply_refurbishment_scenario(df: pd.DataFrame, cfg: RefurbishmentFutureConfig) -> pd.DataFrame:
    """Wendet die bestehende Refurbishment-Logik an (zeitliche Quoten später)."""
    if not cfg.enabled:
        return df
    out = apply_refurbishment(df, fm=cfg.fm, cfg=cfg.cfg)
    # Platzhalter: jährliche Sanierungsquote (noch ohne Wirkung auf \u03c6')
    if cfg.annual_refurb_rate_pct is not None:
        out["annual_refurb_rate_pct"] = float(cfg.annual_refurb_rate_pct)
    return out


# -----------------------------------------------------------------------------
# GeoPackage IO – Entry Point (für Workflow/CLI)
# -----------------------------------------------------------------------------


def run_forecast_2045(
    ctx: PipelineContext,
    *,
    input_gpkg: str,
    output_gpkg: Optional[str] = None,
    layers: Optional[Sequence[str]] = None,
    cfg: Forecast2045Config = Forecast2045Config(),
    verbose: bool = False,
) -> dict:
    """Erstellt eine Szenario-GPKG mit 2045-Prognose-Spalten.

    Erwartet als Input typischerweise eine AP2-Heat-Demand-GPKG (ggf. bereits
    standort-klimakorrigiert via `ap2-climate`).

    Pipeline:
      1) Refurbishment (\u03c6 -> \u03c6', RW/HTW-Skalierung)  [optional]
      2) Leerstand/Occupancy Platzhalter                       [optional]
      3) Zeitlicher Klimawandel-Faktor 2045 auf RW anwenden
      4) HTW 2045 neu bilden (RW2045 + TWW)
    """

    import fiona
    import geopandas as gpd

    input_gpkg = str(Path(input_gpkg))
    if output_gpkg is None:
        p = Path(input_gpkg)
        output_gpkg = str(p.with_name(p.stem + "_forecast2045.gpkg"))

    all_layers = fiona.listlayers(input_gpkg)
    target_layers = list(all_layers) if not layers else [l for l in layers if l in all_layers]
    if not target_layers:
        raise RuntimeError(f"Keine passenden Layer gefunden. Verfügbar: {all_layers}")

    out_path = Path(output_gpkg)
    if out_path.exists():
        out_path.unlink()

    summary = {
        "input_gpkg": input_gpkg,
        "output_gpkg": str(out_path),
        "target_year": int(cfg.climate.target_year),
        "layers": [],
    }

    for lyr in target_layers:
        gdf = gpd.read_file(input_gpkg, layer=lyr)
        if gdf.empty:
            gdf.to_file(out_path, layer=lyr, driver="GPKG")
            summary["layers"].append({"layer": lyr, "n": 0})
            continue

        df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

        # 1) Sanierung (bestehende Logik)
        df = apply_refurbishment_scenario(df, cfg.refurb)

        # 2) Leerstand (Platzhalter)
        df = apply_vacancy_placeholder(df, cfg.vacancy)

        # 3) Klimawandel-Zeitfaktor
        df = apply_time_climate_to_df(df, cfg=cfg, verbose=verbose)

        # Zurück in GeoDataFrame
        gdf_out = gdf.copy()
        for c in df.columns:
            if c == "geometry":
                continue
            gdf_out[c] = df[c].values

        gdf_out.to_file(out_path, layer=lyr, driver="GPKG")
        summary["layers"].append({"layer": lyr, "n": int(len(gdf_out))})

        if verbose:
            print(f"[forecast2045] wrote layer '{lyr}' n={len(gdf_out)}")

    return summary
