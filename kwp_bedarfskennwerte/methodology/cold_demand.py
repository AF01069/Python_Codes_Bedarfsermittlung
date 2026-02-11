# -*- coding: utf-8 -*-
"""
cold_demand.py

Ergänzung des Kühlbedarfs (NWG) auf Basis der Tabelle
data/Bedarfskennwerte_IWU/FraunhoferUmsicht_Tabellenkennwerte_NWG_Kaelte.txt

Es werden **ausschließlich** folgende Felder neu geschrieben (in genau dieser Reihenfolge):
1) Q_RCool_NE_kWh_a
2) Q_RCool_END_kWh_a
3) spec_RCool_NE_kWh_m2a
4) spec_RCool_END_kWh_m2a

Definitionen:
- END = Endenergie (Strom) für Kältebereitstellung [kWh/(m²·a)] bzw. [kWh/a]
- NE  = Nutz-/Nettokälte (thermische Kälte) [kWh_th/(m²·a)] bzw. [kWh_th/a]

Um eine physikalisch konsistente Beziehung sicherzustellen, wird angesetzt:
    spec_RCool_NE_kWh_m2a = spec_RCool_END_kWh_m2a * SEER_angenommen

Hinweise:
- Nutzfläche wird bevorzugt aus 'Final_ANutz' gelesen (Fallbacks vorhanden).
- Zuordnung erfolgt primär über IWU-Typname (Text), sekundär über HK-Code (1..11).
- Es werden keine weiteren Kälte-Metadatenfelder geschrieben; temporäre Hilfsspalten werden vor dem Schreiben entfernt.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from typing import Optional, Sequence

import pandas as pd
import geopandas as gpd

try:  # optional
    import fiona
except Exception:  # pragma: no cover
    fiona = None

ENERGY_REF_AREA_FACTOR = 0.8  # EBF = 80% der BruttogeschossflÃ¤che


def _norm_text(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\s/,_\-–—()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_sep_from_header(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    cands = {",": head.count(","), ";": head.count(";"), "\t": head.count("\t")}
    sep = max(cands, key=cands.get)
    return sep if cands[sep] > 0 else ","


def _read_ref_table(iwu_base_dir: Path) -> pd.DataFrame:
    path = Path(iwu_base_dir) / "FraunhoferUmsicht_Tabellenkennwerte_NWG_Kaelte.txt"
    if not path.exists():
        raise FileNotFoundError(f"Kälte-Kennwerttabelle nicht gefunden: {path}")

    sep = _detect_sep_from_header(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8")

    # Retry wenn Separator falsch (alles in 1 Spalte)
    if df.shape[1] == 1:
        for alt in [",", ";", "\t"]:
            if alt == sep:
                continue
            df2 = pd.read_csv(path, sep=alt, encoding="utf-8")
            if df2.shape[1] > 1:
                df = df2
                break

    need = {"IWU_HK_Geb", "IWU_Typname", "qE_Kuehlung_kWh_m2a", "SEER_angenommen"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Kälte-Kennwerttabelle: Fehlende Spalten {missing}. Vorhanden: {list(df.columns)}")

    df = df.copy()
    df["IWU_HK_Geb"] = pd.to_numeric(df["IWU_HK_Geb"], errors="coerce").astype("Int64")
    df["qE_Kuehlung_kWh_m2a"] = pd.to_numeric(df["qE_Kuehlung_kWh_m2a"], errors="coerce")
    df["SEER_angenommen"] = pd.to_numeric(df["SEER_angenommen"], errors="coerce")
    df["__typ_norm"] = df["IWU_Typname"].map(_norm_text)
    return df


def _list_layers(path: Path) -> Sequence[str]:
    if fiona is not None:
        try:
            return list(fiona.listlayers(path))
        except Exception:
            pass
    try:
        import pyogrio  # type: ignore
        return list(pyogrio.list_layers(path)[0])
    except Exception:
        return []


def _pick_layer(path: Path, preferred: Sequence[str] = ("buildings",)) -> str:
    layers = _list_layers(path)
    if not layers:
        return preferred[0]
    for p in preferred:
        if p in layers:
            return p
    return layers[0]


def _find_area_col(cols: Sequence[str]) -> Optional[str]:
    favorites = [
        "Final_ANutz",
        "Final_Nutzflaeche_m2",
        "Final_Nutzflaeche",
        "ANutz",
        "Nutzflaeche_m2",
        "area_m2",
        "Area_m2",
    ]
    for c in favorites:
        if c in cols:
            return c
    pat = re.compile(r"(final_)?anutz|nutz.*fl|area", re.IGNORECASE)
    for c in cols:
        if pat.search(c):
            return c
    return None


def _find_typname_col(cols: Sequence[str]) -> Optional[str]:
    favorites = [
        "IWU_NWG_Typ",
        "IWU_Typname",
        "demand_iwu_type",
        "demand_iwu_type_name",
        "IWU_Hauptfunktion",
    ]
    for c in favorites:
        if c in cols:
            return c
    pat = re.compile(r"(iwu.*typ|typname|hauptfunktion)", re.IGNORECASE)
    for c in cols:
        if pat.search(c):
            return c
    return None


def _find_hk_code_col(cols: Sequence[str]) -> Optional[str]:
    for c in ("IWU_HK_Geb", "HK_Geb", "demand_iwu_hk", "demand_iwu_type"):
        if c in cols:
            return c
    return None


def _extract_hk_code(s: pd.Series) -> pd.Series:
    def to_code(v):
        """
        F?hrt to_code aus.
        
        Args:
            v: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        if pd.isna(v):
            return pd.NA
        if isinstance(v, (int, float)) and not pd.isna(v):
            try:
                iv = int(v)
                return iv if 1 <= iv <= 11 else pd.NA
            except Exception:
                return pd.NA
        m = re.search(r"\b(\d{1,2})\b", str(v))
        if not m:
            return pd.NA
        iv = int(m.group(1))
        return iv if 1 <= iv <= 11 else pd.NA

    return s.apply(to_code).astype("Int64")


def compute_cold_demand_for_ap2(
    ap2_gpkg_path: Path,
    out_gpkg_path: Path,
    iwu_base_dir: Path,
    typed_gpkg_path: Optional[Path] = None,
    layer: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """
    Berechnet cold demand for AP2.
    
    Args:
        ap2_gpkg_path: Beschreibung.
        out_gpkg_path: Beschreibung.
        iwu_base_dir: Beschreibung.
        typed_gpkg_path: Beschreibung.
        layer: Beschreibung.
        verbose: Beschreibung.
    
    Returns:
        Beschreibung.
    
    Raises:
        Exception: Bei Fehlerbedingungen.
    """
    ap2_gpkg_path = Path(ap2_gpkg_path)
    out_gpkg_path = Path(out_gpkg_path)
    iwu_base_dir = Path(iwu_base_dir)

    ref = _read_ref_table(iwu_base_dir)
    ref_code = ref.dropna(subset=["IWU_HK_Geb"]).copy()

    if layer is None:
        layer = _pick_layer(ap2_gpkg_path, preferred=("buildings",))

    try:
        gdf = gpd.read_file(ap2_gpkg_path, layer=layer)
    except Exception:
        gdf = gpd.read_file(ap2_gpkg_path, layer=layer, engine="fiona")

    area_col = _find_area_col(gdf.columns)
    if area_col is None:
        raise KeyError(
            "Konnte keine Nutzflächen-Spalte finden (z.B. 'Final_ANutz'). "
            f"Vorhandene Spalten (Auszug): {list(gdf.columns)[:30]}"
        )

    is_nwg = pd.Series(True, index=gdf.index)
    if "WG_NWG" in gdf.columns:
        is_nwg = gdf["WG_NWG"].astype(str).str.upper().str.contains("NWG")

    typ_col = _find_typname_col(gdf.columns)
    hk_col = _find_hk_code_col(gdf.columns)

    if typ_col is None and hk_col is None:
        raise KeyError(
            "Konnte keine Typ-Spalte finden (Typname oder HK-Code). "
            f"Vorhandene Spalten (Auszug): {list(gdf.columns)[:30]}"
        )

    work = gdf.copy()
    work["__typ_norm"] = work[typ_col].map(_norm_text) if typ_col is not None else ""
    work["__hk"] = _extract_hk_code(work[hk_col]) if hk_col is not None else pd.NA

    merged = work.merge(
        ref[["__typ_norm", "qE_Kuehlung_kWh_m2a", "SEER_angenommen"]],
        on="__typ_norm",
        how="left",
    )

    need_fb = merged["qE_Kuehlung_kWh_m2a"].isna() & merged["__hk"].notna()
    if need_fb.any():
        fb = merged.loc[need_fb, ["__hk"]].merge(
            ref_code.rename(columns={"IWU_HK_Geb": "__hk"})[["__hk", "qE_Kuehlung_kWh_m2a", "SEER_angenommen"]],
            on="__hk",
            how="left",
        )
        merged.loc[need_fb, "qE_Kuehlung_kWh_m2a"] = fb["qE_Kuehlung_kWh_m2a"].values
        merged.loc[need_fb, "SEER_angenommen"] = fb["SEER_angenommen"].values

    area = pd.to_numeric(merged[area_col], errors="coerce").fillna(0.0)
    area_eff = area * ENERGY_REF_AREA_FACTOR
    spec_end = pd.to_numeric(merged["qE_Kuehlung_kWh_m2a"], errors="coerce").fillna(0.0)
    seer = pd.to_numeric(merged["SEER_angenommen"], errors="coerce").fillna(3.0)
    spec_ne = (spec_end * seer).fillna(0.0)

    out_cols = ["Q_RCool_NE_kWh_a", "Q_RCool_END_kWh_a", "spec_RCool_NE_kWh_m2a", "spec_RCool_END_kWh_m2a"]
    for c in out_cols:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    Q_end = (area_eff * spec_end).fillna(0.0)
    Q_ne = (area_eff * spec_ne).fillna(0.0)

    merged["Q_RCool_NE_kWh_a"] = 0.0
    merged["Q_RCool_END_kWh_a"] = 0.0
    merged["spec_RCool_NE_kWh_m2a"] = 0.0
    merged["spec_RCool_END_kWh_m2a"] = 0.0

    merged.loc[is_nwg, "Q_RCool_NE_kWh_a"] = Q_ne.loc[is_nwg].values
    merged.loc[is_nwg, "Q_RCool_END_kWh_a"] = Q_end.loc[is_nwg].values
    merged.loc[is_nwg, "spec_RCool_NE_kWh_m2a"] = spec_ne.loc[is_nwg].values
    merged.loc[is_nwg, "spec_RCool_END_kWh_m2a"] = spec_end.loc[is_nwg].values

    # remove helper/reference columns so ONLY the 4 new ones remain from this routine
    drop_cols = [c for c in ["qE_Kuehlung_kWh_m2a", "SEER_angenommen", "__typ_norm", "__hk"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    base_cols = [c for c in merged.columns if c not in out_cols]
    merged = merged[base_cols + out_cols]

    if verbose:
        n_nwg = int(is_nwg.sum())
        matched = int(merged.loc[is_nwg, "spec_RCool_END_kWh_m2a"].gt(0).sum())
        print(f"[cold-demand] Fläche: {area_col!r}, Typ: {typ_col!r}, HK: {hk_col!r}")
        print(f"[cold-demand] NWG: {n_nwg}/{len(merged)}; matched (spec_END>0): {matched}/{n_nwg}")

    # Rundung: Kühlenergiegrößen max. 2 Dezimalstellen (spezifisch und absolut)
    for c in out_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").round(2)

    # Jahresfelder als Integer sichern (GPKG-Datentypen)
    for col in ("DIVIS_year", "Final_Baujahr_Mitte", "forecast_year"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")

    out_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(out_gpkg_path, layer="buildings", driver="GPKG")
    return out_gpkg_path
