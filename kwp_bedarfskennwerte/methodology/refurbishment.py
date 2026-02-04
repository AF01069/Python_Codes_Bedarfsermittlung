"""kwp_bedarfskennwerte.methodology.refurbishment

AP2 / WP2 – Modulation des impliziten Sanierungsanteils und Ableitung
szenariokonsistenter Bedarfskennwerte ("Ansatz 1").

Motivation
---------
Die IWU-Typologie weist für Wohngebäude (und analog abgeleitet für NWG)
Bedarfskennwerte als *Mischzustand* des Bestands aus. Der zugehörige
Modernisierungs-/Sanierungsanteil \u03c6 ist dabei ein Bestandsanteil (nicht
Einsparprozentsatz). Für Szenarien soll \u03c6 objekt-/standortspezifisch
moduliert (Denkmalschutz, Leerstand, Eigentum) und daraus ein neuer
Mischkennwert abgeleitet werden.

Ansatz 1 (strukturelle Skalierung ohne explizites q_uns/q_mod)
------------------------------------------------------------
Wenn im Gebäudedatensatz nur der Mischkennwert q_mix (bzw. Q_mix) und der
implizite Sanierungsanteil \u03c6 vorliegen, aber keine expliziten Zustandkennwerte
(q_uns, q_mod), kann der neue Mischkennwert q' konsistent über eine
Skalierung berechnet werden.

Wir nehmen ein (konfigurierbares) Verhältnis r = q_mod / q_uns (0<r<1) an.
Dann gilt:

  q_mix = q_uns * ((1-\u03c6) + \u03c6*r)
  q'    = q_uns * ((1-\u03c6') + \u03c6'*r)

Eliminieren von q_uns liefert die Skalierung:

  q' = q_mix * ((1-\u03c6') + \u03c6'*r) / ((1-\u03c6) + \u03c6*r)

Damit wird nur die Bestandsstruktur (Anteile) verschoben; die Zustandswirkung
(Verhaeltnis r) bleibt unveraendert. Trinkwarmwasser (TWW) bleibt unangetastet.

Ausgabe
-------
- sani_share_corr_pct : \u03c6' in % (0..100)
- refurb_corrected    : \u03c6' in 0..1
- f_denk, f_leer, f_eig (optional fuer QA)
- Korrigierte Raumwaerme-Kennwerte mit Suffix *_refurb_*
- Korrigierte Gesamtkennwerte HTW = RW + TWW (TWW bleibt Original)

Die Pipeline ruft compute_refurbishment_for_ap2(...) auf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Feldzuordnung (GPKG)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldMap:
    """Zuordnung der Feldnamen in der von heat_demand erzeugten GPKG."""

    # impliziter Sanierungsanteil aus heat_demand (0..1 oder 0..100)
    sani_share_pct: str = "sani_share_pct"

    # Ausgaben (Anteile)
    sani_share_corr_pct: str = "sani_share_corr_pct"  # \u03c6' in %
    refurb_corrected: str = "refurb_corrected"        # \u03c6' in 0..1

    # Zensus-Quoten
    zensus_leer: str = "ZENSUS_LeerQuote"  # v
    zensus_eig: str = "ZENSUS_EigQuote"    # o


    # Gebäudegruppe
    wg_nwg: str = "WG_NWG"  # erwartet Werte wie "WG" / "NWG"

    # Denkmal: in euren Daten ggf. nicht vorhanden -> dann f_denk=1.0
    denkmal_candidates: Tuple[str, ...] = (
        "Final_Denkmal",
        "Denkmalschutz",
        "Einzeldenkmal",
        "denkmal",
        "denkmalstatus",
        # Achtung: DIVIS_flag ist fachlich nur ein Proxy, falls ihr nichts anderes habt
        "DIVIS_flag",
    )

    # Spezifische Kennwerte (kWh/m2a)
    spec_rw_ne: str = "spec_RW_NE_kWh_m2a"
    spec_rw_end: str = "spec_RW_END_kWh_m2a"
    spec_tww_ne: str = "spec_TWW_NE_kWh_m2a"
    spec_tww_end: str = "spec_TWW_END_kWh_m2a"

    # Totale Endenergie (falls vorhanden)
    q_rw_end_total: str = "q_rw_end_total_kwh_m2a"
    q_tww_end_total: str = "q_tww_end_total_kwh_m2a"

    # Absolute Jahreswerte (kWh/a)
    Q_rw_ne: str = "Q_RW_NE_kWh_a"
    Q_rw_end: str = "Q_RW_END_kWh_a"
    Q_tww_ne: str = "Q_TWW_NE_kWh_a"
    Q_tww_end: str = "Q_TWW_END_kWh_a"


# -----------------------------------------------------------------------------
# Parametrisierung
# -----------------------------------------------------------------------------


@dataclass
class RefurbishmentConfig:
    """Parameter fuer die Modulation \u03c6 -> \u03c6' und Ansatz-1-Skalierung."""

    # f_denk
    f_denk_if_true: float = 0.3
    f_denk_if_false: float = 1.0

    # Leerstand: f_leer = max(0.6, 1 - 4*v)
    leer_min: float = 0.6
    leer_slope: float = 4.0

    # Eigentum (WG): f_eig = 0.85 + 0.30*o
    eig_base_wg: float = 0.85
    eig_gain_wg: float = 0.30

    # Eigentum (NWG): Eigentümerquote wirkt i.d.R. schwächer (mehr institutionelle/unternehmerische
    # Eigentümer, andere Investitionslogiken). Deshalb engerer Korridor um 1.0.
    # f_eig_nwg = 0.95 + 0.10*o  -> 0.95..1.05
    eig_base_nwg: float = 0.95
    eig_gain_nwg: float = 0.10

    # Defaults, falls Zensuswerte fehlen
    default_leer: float = 0.06  # 6%
    default_eig: float = 0.40   # 40%

    # Ansatz 1: Verhältnis modernisiert/unsaniert (q_mod / q_uns)
    # (Konstante Modellannahme; kann bei Bedarf nach WG/NWG differenziert werden.)
    modernized_ratio_r_wg: float = 0.55
    modernized_ratio_r_nwg: float = 0.55

    # QA-Spalten schreiben->
    write_factor_columns: bool = True

    # Dezimalstellen
    round_spec: int = 3
    round_abs: int = 2


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def _clamp01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _normalize_rate_01(s: pd.Series) -> pd.Series:
    """Normalisiert Quote auf 0..1; akzeptiert 0..1 oder 0..100."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    med = float(s.dropna().median())
    if med > 1.5:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0)


def _parse_boolish(series: pd.Series) -> pd.Series:
    """Robuste Bool-Interpretation (True/False, 1/0, ja/nein, ...)."""
    if series is None:
        return pd.Series(dtype="bool")
    s = series.copy()
    if s.dtype == bool:
        return s.fillna(False)
    s = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "ja", "j", "wahr"}
    false_set = {"0", "false", "f", "no", "n", "nein", "falsch", ""}
    out = s.map(lambda v: True if v in true_set else (False if v in false_set else False))
    return out.fillna(False)


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _scale_series(series: pd.Series, scale: pd.Series, *, ndigits: int) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") * scale).round(ndigits)


# -----------------------------------------------------------------------------
# Kernlogik
# -----------------------------------------------------------------------------


def apply_refurbishment(
    gdf: "pd.DataFrame",
    *,
    fm: FieldMap = FieldMap(),
    cfg: RefurbishmentConfig = RefurbishmentConfig(),
) -> "pd.DataFrame":
    """Wendet die Sanierungsmodulation und Ansatz-1-Skalierung an.

    Erwartet, dass `heat_demand` bereits folgende Basiswerte geschrieben hat:
      - sani_share_pct (\u03c6)
      - spec_RW_*, Q_RW_* sowie TWW-Groessen

    Ergebnis:
      - schreibt \u03c6' sowie korrigierte RW/HTW-Kennwerte mit Suffix *_refurb_*
    """

    df = gdf.copy()

    if fm.sani_share_pct not in df.columns:
        raise KeyError(f"Erwartetes Feld fehlt: {fm.sani_share_pct}")

    # --- \u03c6 (implizit, aus IWU) ------------------------------------------------
    phi = _normalize_rate_01(df[fm.sani_share_pct]).fillna(0.0)

    # --- Denkmalstatus -----------------------------------------------------------
    denk_col = _first_existing(df, fm.denkmal_candidates)
    if denk_col is None:
        is_denk = pd.Series(False, index=df.index)
    else:
        is_denk = _parse_boolish(df[denk_col])

    f_denk = pd.Series(np.where(is_denk, cfg.f_denk_if_true, cfg.f_denk_if_false).astype(float), index=df.index)

    # --- Leerstand / Eigentum ----------------------------------------------------
    v = _normalize_rate_01(df[fm.zensus_leer]) if fm.zensus_leer in df.columns else pd.Series(np.nan, index=df.index)
    o = _normalize_rate_01(df[fm.zensus_eig]) if fm.zensus_eig in df.columns else pd.Series(np.nan, index=df.index)

    v = v.fillna(cfg.default_leer)
    o = o.fillna(cfg.default_eig)

    f_leer = (1.0 - cfg.leer_slope * v).clip(lower=cfg.leer_min, upper=1.0)
    # Gebäudegruppe (WG/NWG) zur differenzierten Parametrisierung
    if fm.wg_nwg in df.columns:
        grp = df[fm.wg_nwg].astype(str).str.strip().str.upper()
        is_nwg = grp.eq("NWG") | grp.str.contains("NWG")
    else:
        is_nwg = pd.Series(False, index=df.index)

    f_eig = pd.Series(
        np.where(
            is_nwg,
            cfg.eig_base_nwg + cfg.eig_gain_nwg * o,
            cfg.eig_base_wg + cfg.eig_gain_wg * o,
        ),
        index=df.index,
    ).astype(float).clip(lower=0.0, upper=2.0)

    # --- \u03c6' (moduliert) --------------------------------------------------------
    phi_prime = _clamp01(phi * f_denk * f_leer * f_eig)

    df[fm.refurb_corrected] = phi_prime
    df[fm.sani_share_corr_pct] = (phi_prime * 100.0).round(3)

    if cfg.write_factor_columns:
        df["f_denk"] = f_denk.round(3)
        df["f_leer"] = f_leer.round(3)
        df["f_eig"] = f_eig.round(3)

    # --- Ansatz 1: Skalierungsfaktor --------------------------------------------
    # scale = ((1-\u03c6') + \u03c6'*r) / ((1-\u03c6) + \u03c6*r)
    # Verhältnis r kann nach WG/NWG variieren (Default identisch).
    r_wg = float(cfg.modernized_ratio_r_wg)
    r_nwg = float(cfg.modernized_ratio_r_nwg)
    r_wg = min(max(r_wg, 0.05), 0.95)
    r_nwg = min(max(r_nwg, 0.05), 0.95)
    r_series = pd.Series(np.where(is_nwg, r_nwg, r_wg), index=df.index).astype(float)

    denom_base = (1.0 - phi) + phi * r_series
    denom_scen = (1.0 - phi_prime) + phi_prime * r_series

    # Vermeide /0; in diesen Faellen bleibt scale NaN, Kennwerte werden NaN.
    scale = (denom_scen / denom_base.replace({0.0: np.nan})).astype(float)

    # --- Raumwaerme: korrigierte spezifische Kennwerte --------------------------
    if fm.spec_rw_ne in df.columns:
        df["spec_RW_NE_refurb_kWh_m2a"] = _scale_series(df[fm.spec_rw_ne], scale, ndigits=cfg.round_spec)
    if fm.spec_rw_end in df.columns:
        df["spec_RW_END_refurb_kWh_m2a"] = _scale_series(df[fm.spec_rw_end], scale, ndigits=cfg.round_spec)
    if fm.q_rw_end_total in df.columns:
        df["q_rw_end_total_refurb_kwh_m2a"] = _scale_series(df[fm.q_rw_end_total], scale, ndigits=cfg.round_spec)

    # --- Raumwaerme: korrigierte absolute Jahreswerte ---------------------------
    if fm.Q_rw_ne in df.columns:
        df["Q_RW_NE_refurb_kWh_a"] = _scale_series(df[fm.Q_rw_ne], scale, ndigits=cfg.round_abs)
    if fm.Q_rw_end in df.columns:
        df["Q_RW_END_refurb_kWh_a"] = _scale_series(df[fm.Q_rw_end], scale, ndigits=cfg.round_abs)

    # --- Gesamt (HTW = RW + TWW), TWW bleibt Original ---------------------------
    # Spezifisch
    if ("spec_RW_NE_refurb_kWh_m2a" in df.columns) and (fm.spec_tww_ne in df.columns):
        df["spec_HTW_NE_refurb_kWh_m2a"] = (
            pd.to_numeric(df["spec_RW_NE_refurb_kWh_m2a"], errors="coerce")
            + pd.to_numeric(df[fm.spec_tww_ne], errors="coerce")
        ).round(cfg.round_spec)

    if ("spec_RW_END_refurb_kWh_m2a" in df.columns) and (fm.spec_tww_end in df.columns):
        df["spec_HTW_END_refurb_kWh_m2a"] = (
            pd.to_numeric(df["spec_RW_END_refurb_kWh_m2a"], errors="coerce")
            + pd.to_numeric(df[fm.spec_tww_end], errors="coerce")
        ).round(cfg.round_spec)

    if ("q_rw_end_total_refurb_kwh_m2a" in df.columns) and (fm.q_tww_end_total in df.columns):
        df["q_htw_end_total_refurb_kwh_m2a"] = (
            pd.to_numeric(df["q_rw_end_total_refurb_kwh_m2a"], errors="coerce")
            + pd.to_numeric(df[fm.q_tww_end_total], errors="coerce")
        ).round(cfg.round_spec)

    # Absolut
    if ("Q_RW_NE_refurb_kWh_a" in df.columns) and (fm.Q_tww_ne in df.columns):
        df["Q_HTW_NE_refurb_kWh_a"] = (
            pd.to_numeric(df["Q_RW_NE_refurb_kWh_a"], errors="coerce")
            + pd.to_numeric(df[fm.Q_tww_ne], errors="coerce")
        ).round(cfg.round_abs)

    if ("Q_RW_END_refurb_kWh_a" in df.columns) and (fm.Q_tww_end in df.columns):
        df["Q_HTW_END_refurb_kWh_a"] = (
            pd.to_numeric(df["Q_RW_END_refurb_kWh_a"], errors="coerce")
            + pd.to_numeric(df[fm.Q_tww_end], errors="coerce")
        ).round(cfg.round_abs)

    # Dokumentiere r in der Tabelle (optional, hilfreich fuer QA)
    df["refurb_r_mod_uns_ratio"] = r_series.round(3)

    return df


# -----------------------------------------------------------------------------
# GeoPackage IO – AP2 Entry Point
# -----------------------------------------------------------------------------


def compute_refurbishment_for_ap2(
    *,
    ap2_heat_gpkg_path: Path,
    out_gpkg_path: Path,
    layer_name: Optional[str] = None,
    fm: FieldMap = FieldMap(),
    cfg: RefurbishmentConfig = RefurbishmentConfig(),
) -> Path:
    """Liest `ap2_heat_gpkg_path`, wendet apply_refurbishment an und schreibt nach out."""
    import geopandas as gpd

    ap2_heat_gpkg_path = Path(ap2_heat_gpkg_path)
    out_gpkg_path = Path(out_gpkg_path)

    if layer_name is None:
        layer_name = _guess_first_layer(ap2_heat_gpkg_path)

    gdf = gpd.read_file(ap2_heat_gpkg_path, layer=layer_name)
    gdf2 = apply_refurbishment(gdf, fm=fm, cfg=cfg)

    out_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf2.to_file(out_gpkg_path, layer=layer_name, driver="GPKG")
    return out_gpkg_path


def _guess_first_layer(gpkg_path: Path) -> str:
    """Robust: ermittelt den ersten Layernamen eines GeoPackages."""
    gpkg_path = Path(gpkg_path)

    # 1) Fiona (Standard)
    try:
        import fiona

        layers = list(fiona.listlayers(str(gpkg_path)))
        if layers:
            return layers[0]
    except Exception:
        pass

    # 2) pyogrio (Fallback)
    try:
        import pyogrio

        layers = pyogrio.list_layers(str(gpkg_path))
        if isinstance(layers, list) and layers:
            first = layers[0]
            return first[0] if isinstance(first, (tuple, list)) else str(first)
    except Exception:
        pass

    # 3) Notfalls
    return "buildings"
