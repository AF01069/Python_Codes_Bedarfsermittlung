# -*- coding: utf-8 -*-
"""
kwp_bedarfskennwerte.methodology.heat_demand

Ziel:
- Aus einem getypten AP2-Gebäudelayer (GPKG) werden IWU-basierte Bedarfskennwerte
  für Raumwärme (RW) und Trinkwarmwasser (TWW) übernommen.
- Ausgabe: neue GPKG mit spezifischen Kennwerten [kWh/m²a] sowie
  gebäudebezogenen Bedarfen [kWh/a].

Wichtig:
- Die Kennwerte werden aus der kombinierten, "flachen" CSV übernommen
  (IWU_Bedarfskennwerte_combined_flat.csv). Diese enthält auch Fallback-Zeilen
  (leere Energieträger/Heizungsart), sodass jedes energierelevante Gebäude
  einen Kennwert erhalten soll.
- Zusätzlich werden QA-Dateien geschrieben, damit die Matching-Qualität nachvollziehbar ist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import geopandas as gpd



def _norm_token(v: object) -> str:
    """Normalize tokens used for matching.

    - Converts None/NaN/pandas NA to empty string.
    - Strips whitespace.
    - Converts common NA string representations ("nan", "none", "<na>") to empty string.
    """
    if v is None:
        return ""
    try:
        # pandas/ numpy NaN handling
        if isinstance(v, float) and np.isnan(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "<na>", "na"}:
        return ""
    return s


def _normalize_baujahr_label(raw: object, *, sector: str | None = None) -> str:
    """Normalizes Baujahresphasen strings so they match the IWU reference table.

    Handles variants like:
    - "1919 bis 1948" -> "1919-1948"
    - "1949 bis 1978" -> "1949-1978"
    - "bis 1978" / "vor 1978" -> "vor 1978" (NWG alt)
    - "ab 2010" -> "NEU" (NWG)
    The function is conservative: if nothing matches, returns the stripped input.
    """
    s = _norm_token(raw)
    if not s:
        return ""
    s2 = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    s2 = re.sub(r"\s+", " ", s2).strip()

    # unify 'bis' ranges
    m = re.match(r"^(\d{4})\s*(bis|-)\s*(\d{4})$", s2, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # 'bis YYYY' or 'vor YYYY'
    m = re.match(r"^(bis|vor)\s*(\d{4})$", s2, flags=re.IGNORECASE)
    if m:
        year = int(m.group(2))
        if sector == "NWG" and year in (1978, 1979):
            return "vor 1978"
        return f"bis {year}" if s2.lower().startswith("bis") else f"vor {year}"

    # 'ab YYYY'
    m = re.match(r"^ab\s*(\d{4})$", s2, flags=re.IGNORECASE)
    if m:
        year = int(m.group(1))
        if sector == "NWG" and year >= 2010:
            return "NEU"
        return f"ab {year}"

    # already canonical NWG labels
    if sector == "NWG":
        if s2.lower() in {"neu", "neubau", "neubauten"}:
            return "NEU"
        # common variant: "1978-2010" / "1978 bis 2010"
        if re.match(r"^1978\s*(bis|-)\s*2010$", s2):
            return "1978 bis 2010"

    return s2


def _normalize_entraeger(raw: object, *, sector: str) -> str:
    """Normalize Energieträger strings for matching.

    Hintergrund (NWG):
    Die bereitgestellte NWG-Referenztabelle differenziert häufig nur zwischen
    "EL" (elektrisch) und "SONST" (alle übrigen Energieträger).
    Daher werden für NWG alle nicht-elektrischen Energieträger zu "SONST"
    zusammengefasst.
    """
    e = _norm_token(raw).upper()
    if not e:
        return ""
    if sector == "NWG":
        if e in {"EL", "STROM", "ELEKTRIZITAET", "ELEKTRISCH"}:
            return "EL"
        # alles andere als SONST
        return "SONST"
    return e




# -----------------------------
# Konfiguration / Spaltennamen
# -----------------------------

# Spalten im AP2-Layer (Gebäude)
COL_SECTOR = "WG_NWG"  # "WG" oder "NWG"
COL_WG_TYP = "IWU_WG_Typ"
COL_NWG_TYP = "IWU_NWG_Typ"
COL_BAUJAHR = "IWU_Baujahresphase"
COL_CARRIER = "IWU_EnTraeger"
COL_HEATING = "IWU_Heizungsart"
COL_ANUTZ = "Final_ANutz"  # Nutzfläche (m²) für Multiplikation

# Kennwert-Output-Spalten (spezifisch)
COL_Q_RW_NE = "q_rw_ne_kwh_m2a"
COL_Q_TWW_NE = "q_tww_ne_kwh_m2a"
COL_Q_RW_END = "q_rw_end_kwh_m2a"
COL_Q_TWW_END = "q_tww_end_kwh_m2a"

# Gesamt-Endenergie (Summe aus Raumwärme + Trinkwarmwasser)
COL_Q_END_TOTAL = "q_end_total_kwh_m2a"

# Sanierungsgrad (implizit aus IWU-Referenz)
COL_SANI_SHARE_PCT = "sani_share_pct"

# Kennwert-Output-Spalten (Gebäude, absolut)
COL_ABS_RW_NE = "Q_RW_NE_kWh_a"
COL_ABS_TWW_NE = "Q_TWW_NE_kWh_a"
COL_ABS_RW_END = "Q_RW_END_kWh_a"
COL_ABS_TWW_END = "Q_TWW_END_kWh_a"
COL_ABS_END_TOTAL = "Q_END_TOTAL_kWh_a"

# Ausgabe-Spalten (spezifisch, redundant aber praktisch)
COL_SPEC_RW_NE = "spec_RW_NE_kWh_m2a"
COL_SPEC_TWW_NE = "spec_TWW_NE_kWh_m2a"
COL_SPEC_RW_END = "spec_RW_END_kWh_m2a"
COL_SPEC_TWW_END = "spec_TWW_END_kWh_m2a"
COL_SPEC_END_TOTAL = "spec_END_TOTAL_kWh_m2a"

# Kennwert-CSV (Flat) erwartete Spalten
REF_COL_SECTOR = "sector"       # WG / NWG
REF_COL_TYP = "typ_code"        # für WG: z.B. SEMI_DETACHED, für NWG: Langtext
REF_COL_BAUJAHR = "baujahr"     # String (muss zu IWU_Baujahresphase passen)
REF_COL_CARRIER = "entraeger"   # z.B. GAS, FW, ...
REF_COL_HEATING = "heizungsart" # z.B. Zentralheizung, Etagenheizung, Fernheizung
REF_COL_KENN = "kennwert"       # z.B. Q_RW_NE, Q_TWW_END
REF_COL_VALUE = "value"         # float
REF_COL_SANI_SHARE_PCT = "sani_share_pct"  # Sanierungsgrad [%] pro Typ (implizit)

# Sanierungsgrad-Spalte in der (flachen) IWU-Tabelle
REF_COL_SANI = "sani_share_pct"


# -----------------------------
# Hilfsfunktionen: Layer / IO
# -----------------------------


def _pick_gpkg_layer(
    path: Path,
    preferred: Sequence[str] = ("buildings", "ap2_buildings", "buildings_typed", "layer"),
) -> str:
    """Wählt einen Layernamen aus einer GPKG-Datei.

    Erst werden bevorzugte Layernamen gesucht, danach wird auf den ersten Layer
    der Datei zurückgegriffen.

    Hintergrund: In manchen GeoPandas-Setups ist `gpd.io.file.fiona` nicht verfügbar
    (None), obwohl `fiona` installiert ist oder `pyogrio` als Backend genutzt wird.
    """
    layers: List[str] = []
    # 1) Direkt via fiona
    try:
        import fiona  # type: ignore

        layers = list(fiona.listlayers(str(path)))
    except Exception:
        # 2) Fallback: pyogrio (wenn installiert)
        try:
            import pyogrio  # type: ignore

            layers = [name for (name, _geom_type) in pyogrio.list_layers(str(path))]
        except Exception:
            layers = []

    # 3) Wenn wir keine Layerliste bekommen: preferred Layer "probelesen"
    if not layers:
        for name in preferred:
            try:
                _ = gpd.read_file(str(path), layer=name, rows=1)
                return name
            except Exception:
                continue
        # 4) Letzter Versuch: GeoPandas Default (erster Layer)
        try:
            _ = gpd.read_file(str(path), rows=1)
            # Kann nicht sicher wissen, wie der Default-Layer heißt – wir geben den ersten
            # preferred zurück; `to_file(..., layer=layer_name)` nutzt später denselben Namen.
            return preferred[0]
        except Exception as e:
            raise RuntimeError(
                f"Konnte keinen Layer aus '{path}' bestimmen (weder fiona noch pyogrio verfügbar/lesbar). "
                "Ist der Pfad korrekt und ist es eine gültige GPKG-Datei->"
            ) from e

    for name in preferred:
        if name in layers:
            return name
    return layers[0]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_qa(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


# -----------------------------
# Referenztabelle laden/normalisieren
# -----------------------------

def load_iwu_reference_flat(iwu_base_dir: Path) -> pd.DataFrame:
    """
    Lädt die kombinierte IWU-Flattable. Erwarteter Dateiname:
    - IWU_Bedarfskennwerte_combined_flat.csv
    """
    csv_path = Path(iwu_base_dir) / "IWU_Bedarfskennwerte_combined_flat.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Referenzdatei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path, sep=';', dtype=str, keep_default_na=False, na_filter=False)

    for c in [REF_COL_SECTOR, REF_COL_TYP, REF_COL_BAUJAHR, REF_COL_CARRIER, REF_COL_HEATING, REF_COL_KENN]:
        df[c] = df[c].apply(_norm_token)

    # Sanierungsgrad optional
    if REF_COL_SANI_SHARE_PCT in df.columns:
        df[REF_COL_SANI_SHARE_PCT] = df[REF_COL_SANI_SHARE_PCT].apply(_norm_token)

    # Baujahresphasen normalisieren (v. a. NWG: vor 1978 / 1978 bis 2010 / NEU)
    df[REF_COL_BAUJAHR] = df.apply(lambda r: _normalize_baujahr_label(r[REF_COL_BAUJAHR], sector=r[REF_COL_SECTOR]), axis=1)
    df[REF_COL_CARRIER] = df.apply(lambda r: _normalize_entraeger(r[REF_COL_CARRIER], sector=r[REF_COL_SECTOR]), axis=1)

    df[REF_COL_VALUE] = pd.to_numeric(df[REF_COL_VALUE], errors="coerce")

    if REF_COL_SANI_SHARE_PCT in df.columns:
        df[REF_COL_SANI_SHARE_PCT] = pd.to_numeric(df[REF_COL_SANI_SHARE_PCT], errors="coerce")
    return df


def _pivot_reference_wide(df_flat: pd.DataFrame) -> pd.DataFrame:
    """
    Pivotiert die Flattable zu einer Wide-Form:
    Index: (sector, typ_code, baujahr, entraeger, heizungsart)
    Columns: Kennwerte (Q_RW_NE, Q_TWW_NE, Q_RW_END, ...)
    """
    key_cols = [REF_COL_SECTOR, REF_COL_TYP, REF_COL_BAUJAHR, REF_COL_CARRIER, REF_COL_HEATING]

    wide = (
        df_flat.pivot_table(
            index=key_cols,
            columns=REF_COL_KENN,
            values=REF_COL_VALUE,
            aggfunc="first",
        )
        .reset_index()
    )

    # Sanierungsgrad pro Schlüssel (nicht kennwertabhängig) anfügen, falls vorhanden.
    if REF_COL_SANI_SHARE_PCT in df_flat.columns:
        sani = (
            df_flat[key_cols + [REF_COL_SANI_SHARE_PCT]]
            .copy()
        )
        sani[REF_COL_SANI_SHARE_PCT] = pd.to_numeric(sani[REF_COL_SANI_SHARE_PCT], errors="coerce")
        sani = (
            sani.groupby(key_cols, dropna=False)[REF_COL_SANI_SHARE_PCT]
            .first()
            .reset_index()
        )
        wide = wide.merge(sani, on=key_cols, how="left")
    wide.columns = [c if isinstance(c, str) else str(c) for c in wide.columns]
    return wide


def _rename_ref_columns(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Benennt Kennwertspalten in die internen Spaltennamen um.
    """
    rename = {
        "Q_RW_NE": COL_Q_RW_NE,
        "Q_TWW_NE": COL_Q_TWW_NE,
        "Q_RW_END": COL_Q_RW_END,
        "Q_TWW_END": COL_Q_TWW_END,
        # Sanierungsgrad (falls in Wide vorhanden)
        REF_COL_SANI_SHARE_PCT: COL_SANI_SHARE_PCT,
    }
    return df_wide.rename(columns=rename).copy()


# -----------------------------
# Matching-Logik mit Fallbacks
# -----------------------------


# -----------------------------
# Normalisierung (Keys)
# -----------------------------

def _norm_str(x) -> str:
    """Robuste String-Normalisierung: NaN/None/'nan' -> '' und trim."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def _norm_baujahr_phase(raw: str) -> str:
    """
    Normalisiert IWU_Baujahresphase, so dass sie zur Referenz passt.
    Beispiele:
      '1949 – 1978' -> '1949-1978'
      '1949 - 1978' -> '1949-1978'
    Andere Klassen wie 'vor 1978', '1978 bis 2010', 'bis 1918' bleiben erhalten.
    """
    s = _norm_str(raw)
    s = s.replace("–", "-").replace("—", "-")
    # spaces around hyphen entfernen: '1949 - 1978' -> '1949-1978'
    s = re.sub(r"\s*-\s*", "-", s)
    # multiple spaces reduzieren
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_typ_code(sector: str, typ_code: str) -> str:
    """WG: SMALL_OUTBUILDING existiert nicht in der Referenz -> SFH_DETACHED."""
    t = _norm_str(typ_code)
    if _norm_str(sector) == "WG" and t == "SMALL_OUTBUILDING":
        return "SFH_DETACHED"
    return t

def _norm_entraeger(raw: str) -> str:
    """Energieträger in Referenz sind i.d.R. Großbuchstaben (GAS, FW, EL, ...)."""
    return _norm_str(raw).upper()

def _norm_heizungsart(entraeger: str, heizungsart: str) -> str:
    """
    Konsistenzregeln:
    - Fernwärme (FW) wird in der Referenz typischerweise mit Heizungsart 'Fernheizung' geführt.
    """
    e = _norm_entraeger(entraeger)
    h = _norm_str(heizungsart)
    if e == "FW":
        return "Fernheizung"
    return h

@dataclass(frozen=True)
class MatchKey:
    """
    Datenklasse f?r match key.
    """
    sector: str
    typ_code: str
    baujahr: str
    entraeger: str
    heizungsart: str


def _build_building_key_row(row: pd.Series) -> MatchKey:
    """
    Erstellt den Matching-Key eines Gebäudes und normalisiert ihn so, dass er
    möglichst robust gegen Darstellungsunterschiede (z.B. '1949 – 1978' vs '1949-1978')
    ist und zur IWU-Referenz passt.
    """
    sector = _norm_str(row.get(COL_SECTOR, ""))

    # Typcode je nach Sektor
    if sector == "WG":
        typ_raw = row.get(COL_WG_TYP, "")
    else:
        typ_raw = row.get(COL_NWG_TYP, "")
    typ_code = _norm_typ_code(sector, typ_raw)

    baujahr = _norm_baujahr_phase(row.get(COL_BAUJAHR, ""))
    entraeger = _norm_entraeger(row.get(COL_CARRIER, ""))
    heizungsart = _norm_heizungsart(entraeger, row.get(COL_HEATING, ""))

    return MatchKey(
        sector=sector,
        typ_code=typ_code,
        baujahr=baujahr,
        entraeger=entraeger,
        heizungsart=heizungsart,
    )


def _candidate_keys(key: MatchKey) -> List[MatchKey]:
    """Erzeugt Matching-Kandidaten (Priorität von spezifisch -> generisch).

    Hintergrund:
    - *WG*: IWU/TABULA differenziert Endenergie nach Energieträger/Heizungsart,
      aber Nutzenergie (RW_NE / TWW_NE) liegt typischerweise nur als "generische"
      Zeile ohne Energieträger/Heizungsart vor. Deshalb sind Fallbacks nötig.
    - *NWG*: In der IWU-NWG-Typologie wird i.d.R. nicht nach Heizungsart
      differenziert; Energieträger ist (sofern vorhanden) meist nur EL vs. SONST.
      Für robuste Abdeckung ignorieren wir Heizungsart immer und lassen
      Energieträger nur als optionalen ersten Kandidaten (EL/SONST) stehen.

    Rückgabe: Liste von Kandidaten-Keys in Prioritätsreihenfolge.
    """
    if key.sector == "NWG":
        # Heizungsart im NWG immer leer (Referenz i.d.R. ohne Heizungsart)
        k_nwg = MatchKey(key.sector, key.typ_code, key.baujahr, key.entraeger, "")
        candidates = [
            k_nwg,                                                # 1) NWG: (typ,baujahr,EL|SONST,"")
            MatchKey(key.sector, key.typ_code, key.baujahr, "", ""),   # 2) ohne Energieträger (falls Referenz so vorliegt)
            MatchKey(key.sector, key.typ_code, "", "", ""),            # 3) Typ-only
            MatchKey(key.sector, "", key.baujahr, "", ""),             # 4) Baujahr-only
            MatchKey(key.sector, "", "", "", ""),                      # 5) Full fallback
        ]
    else:
        # WG: klassisches Fallback-Schema
        candidates = [
            key,  # 1) exakt
            MatchKey(key.sector, key.typ_code, key.baujahr, "", key.heizungsart),   # 2) entraeger blank
            MatchKey(key.sector, key.typ_code, key.baujahr, key.entraeger, ""),    # 3) heizungsart blank
            MatchKey(key.sector, key.typ_code, key.baujahr, "", ""),               # 4) full fallback (hier stehen meist die NE-Werte!)
        ]

    # Duplikate entfernen (Reihenfolge beibehalten)
    seen = set()
    out: List[MatchKey] = []
    for k in candidates:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out




def _make_ref_lookup(df_ref_wide: pd.DataFrame) -> Dict[MatchKey, Dict[str, float]]:
    """
    Baut ein Lookup: MatchKey -> Kennwert-Dict (interne Spaltennamen -> float).
    """
    demand_cols = [
        COL_Q_RW_NE,
        COL_Q_TWW_NE,
        COL_Q_RW_END,
        COL_Q_TWW_END,
        COL_SANI_SHARE_PCT,
    ]
    lookup: Dict[MatchKey, Dict[str, float]] = {}

    for _, r in df_ref_wide.iterrows():
        mk = MatchKey(
            sector=_norm_token(r.get(REF_COL_SECTOR, "")),
            typ_code=_norm_token(r.get(REF_COL_TYP, "")),
            baujahr=_normalize_baujahr_label(r.get(REF_COL_BAUJAHR, ""), sector=_norm_token(r.get(REF_COL_SECTOR, ""))),
            entraeger=_normalize_entraeger(r.get(REF_COL_CARRIER, ""), sector=_norm_token(r.get(REF_COL_SECTOR, ""))),
            heizungsart=_norm_token(r.get(REF_COL_HEATING, "")),
        )
        vals = {}
        for c in demand_cols:
            if c in df_ref_wide.columns:
                vals[c] = float(r[c]) if pd.notna(r.get(c)) else np.nan
        lookup[mk] = vals

    return lookup


def _apply_reference_demands(
    gdf: gpd.GeoDataFrame,
    df_ref_wide: pd.DataFrame,
    *,
    qa_dir: Optional[Path] = None,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Weist jedem Gebäude spezifische Kennwerte zu (q_*_kwh_m2a).

    Wichtig: In der Referenz sind nicht alle Kennwerte in jeder Schlüsselzeile enthalten.
    Beispiel WG:
      - Q_RW_END / Q_TWW_END sind nach (entraeger, heizungsart) differenziert
      - Q_RW_NE / Q_TWW_NE liegen häufig nur als generische Zeile ohne entraeger/heizungsart vor
    Deshalb wird hier **spaltenweise** gematcht:
      - Für jede Zielspalte wird der erste nicht-leere Wert über Kandidaten-Keys gesucht.
    """
    df_ref_wide = _rename_ref_columns(df_ref_wide)
    ref_lookup = _make_ref_lookup(df_ref_wide)

    need_cols = [
        COL_Q_RW_NE,
        COL_Q_TWW_NE,
        COL_Q_RW_END,
        COL_Q_TWW_END,
        COL_SANI_SHARE_PCT,
    ]

    # Zielspalten initialisieren
    for c in need_cols:
        if c not in gdf.columns:
            gdf[c] = np.nan

    match_level: List[int] = []

    for idx, row in gdf.iterrows():
        mk = _build_building_key_row(row)
        cands = _candidate_keys(mk)

        # Match-Level: erster Kandidat, der in der Referenz existiert (unabhängig von NaNs)
        level = 0
        for i, cand in enumerate(cands, start=1):
            if cand in ref_lookup:
                level = i
                break
        match_level.append(level)

        # Spaltenweises Matching: je Kennwert den ersten nicht-NaN-Wert ziehen
        for col in need_cols:
            cur = gdf.at[idx, col]
            if pd.notna(cur):
                continue

            for cand in cands:
                vals = ref_lookup.get(cand)
                if not vals:
                    continue
                v = vals.get(col, np.nan)
                if pd.notna(v):
                    gdf.at[idx, col] = v
                    break

        # NOTE: Historisch gab es in einigen Tabellen Spalten wie "Q_*_END_TOTAL".
        # Diese werden bewusst **nicht** mehr genutzt, da der gewünschte Gesamt-
        # Endenergiekennwert als Summe aus Raumwärme- und TWW-Endenergie gebildet wird.

    gdf["_match_level"] = match_level

    # QA Report
    rep = (
        pd.Series(match_level, name="match_level")
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("match_level")
        .reset_index(name="n_buildings")
    )

    if qa_dir is not None:
        _ensure_dir(qa_dir)

        keys_df = pd.DataFrame(
            {
                "sector": gdf[COL_SECTOR].astype(str),
                "typ_code": np.where(gdf[COL_SECTOR] == "WG", gdf.get(COL_WG_TYP, ""), gdf.get(COL_NWG_TYP, "")),
                "baujahr": gdf.get(COL_BAUJAHR, ""),
                "entraeger": gdf.get(COL_CARRIER, ""),
                "heizungsart": gdf.get(COL_HEATING, ""),
                "match_level": match_level,
            }
        )
        _write_qa(keys_df, qa_dir / "qa_heat_demand_building_keys.csv")
        _write_qa(rep, qa_dir / "qa_heat_demand_match_report.csv")

        unmatched = gdf[gdf["_match_level"] == 0].copy()
        cols = ["LOD_UNITID", COL_SECTOR, COL_WG_TYP, COL_NWG_TYP, COL_BAUJAHR, COL_CARRIER, COL_HEATING, COL_ANUTZ]
        cols = [c for c in cols if c in unmatched.columns]
        _write_qa(pd.DataFrame(unmatched[cols]), qa_dir / "qa_heat_demand_unmatched.csv")

        lines = []
        lines.append("Heat-Demand Matching Summary")
        lines.append(f"n_buildings_total: {len(gdf)}")
        lines.append(f"n_matched: {(gdf['_match_level'] > 0).sum()}")
        lines.append(f"n_unmatched: {(gdf['_match_level'] == 0).sum()}")
        lines.append("")
        lines.append("match_level meaning:")
        lines.append("1 = exact (sector, typ, baujahr, entraeger, heizungsart)")
        lines.append("2 = entraeger blank fallback")
        lines.append("3 = heizungsart blank fallback")
        lines.append("4 = entraeger+heizungsart blank fallback")
        lines.append("5 = NWG: typ-only / baujahr-only / full fallback (optional)")
        _write_text("".join(lines), qa_dir / "qa_heat_demand_matching.txt")

    return gdf, rep


# -----------------------------
# Gebäudebedarfe berechnen
# -----------------------------

def _compute_building_demands(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Berechnet absolute Bedarfe [kWh/a] aus spezifischen Kennwerten [kWh/m²a]
    mittels Final_ANutz [m²].

    Schreibt:
    - spec_* (Kopie der q_* Spalten)
    - Q_*_kWh_a (spezifisch * ANutz)
    """
    if COL_ANUTZ not in gdf.columns:
        raise KeyError(f"Spalte '{COL_ANUTZ}' fehlt im Gebäudelayer (für absolute Bedarfe erforderlich).")

    # spezifische Spalten (Kopie)
    gdf[COL_SPEC_RW_NE] = gdf.get(COL_Q_RW_NE, np.nan)
    gdf[COL_SPEC_TWW_NE] = gdf.get(COL_Q_TWW_NE, np.nan)
    gdf[COL_SPEC_RW_END] = gdf.get(COL_Q_RW_END, np.nan)
    gdf[COL_SPEC_TWW_END] = gdf.get(COL_Q_TWW_END, np.nan)

    # Gesamt-Endenergie (spezifisch)
    gdf[COL_Q_END_TOTAL] = (
        pd.to_numeric(gdf.get(COL_Q_RW_END, np.nan), errors="coerce")
        + pd.to_numeric(gdf.get(COL_Q_TWW_END, np.nan), errors="coerce")
    )
    gdf[COL_SPEC_END_TOTAL] = gdf[COL_Q_END_TOTAL]

    # absolute Bedarfe
    anutz = pd.to_numeric(gdf[COL_ANUTZ], errors="coerce")
    gdf[COL_ABS_RW_NE] = pd.to_numeric(gdf.get(COL_Q_RW_NE, np.nan), errors="coerce") * anutz
    gdf[COL_ABS_TWW_NE] = pd.to_numeric(gdf.get(COL_Q_TWW_NE, np.nan), errors="coerce") * anutz
    gdf[COL_ABS_RW_END] = pd.to_numeric(gdf.get(COL_Q_RW_END, np.nan), errors="coerce") * anutz
    gdf[COL_ABS_TWW_END] = pd.to_numeric(gdf.get(COL_Q_TWW_END, np.nan), errors="coerce") * anutz
    gdf[COL_ABS_END_TOTAL] = pd.to_numeric(gdf.get(COL_Q_END_TOTAL, np.nan), errors="coerce") * anutz

    # Ausgabe runden (max. 2 Dezimalstellen)
    round_cols = [
        # spezifisch
        COL_Q_RW_NE, COL_Q_TWW_NE, COL_Q_RW_END, COL_Q_TWW_END, COL_Q_END_TOTAL,
        COL_SPEC_RW_NE, COL_SPEC_TWW_NE, COL_SPEC_RW_END, COL_SPEC_TWW_END, COL_SPEC_END_TOTAL,
        # absolut
        COL_ABS_RW_NE, COL_ABS_TWW_NE, COL_ABS_RW_END, COL_ABS_TWW_END, COL_ABS_END_TOTAL,
        # sani-share (Prozent)
        COL_SANI_SHARE_PCT,
    ]
    for _c in round_cols:
        if _c in gdf.columns:
            gdf[_c] = pd.to_numeric(gdf[_c], errors="coerce").round(2)

    return gdf


# -----------------------------
# Public API
# -----------------------------

def compute_heat_demand_for_ap2(
    *,
    ap2_gpkg_path: Path,
    iwu_base_dir: Path,
    out_gpkg_path: Path,
    out_dir: Optional[Path] = None,
    layer_name: Optional[str] = None,
) -> Path:
    """
    Hauptfunktion für AP2:
    - liest den getypten AP2-Layer
    - liest IWU-Referenztabelle (flat), pivottiert in wide
    - matcht Kennwerte inkl. Fallbacks
    - berechnet Gebäude-Bedarfe
    - schreibt Ergebnis-GPKG (mit demselben Layernamen wie Eingabe)
    - schreibt QA-Dateien (wenn out_dir gesetzt)

    Parameter:
    - ap2_gpkg_path: Pfad zur getypten AP2-GPKG (Eingabe)
    - iwu_base_dir : Basisverzeichnis data/Bedarfskennwerte_IWU (enthält IWU_Bedarfskennwerte_combined_flat.csv)
    - out_gpkg_path: Pfad zur Ergebnis-GPKG
    - out_dir      : Basis-Ausgabeverzeichnis (für QA -> out/ap2/qa/...), optional
    - layer_name   : Layername in ap2_gpkg_path, optional (sonst autodetect)
    """
    ap2_gpkg_path = Path(ap2_gpkg_path)
    iwu_base_dir = Path(iwu_base_dir)
    out_gpkg_path = Path(out_gpkg_path)

    if layer_name is None:
        layer_name = _pick_gpkg_layer(ap2_gpkg_path)

    gdf = gpd.read_file(ap2_gpkg_path, layer=layer_name)

    # Matching-relevante Spalten robust normalisieren (NaN/None -> "")
    for c in [COL_SECTOR, COL_WG_TYP, COL_NWG_TYP, COL_BAUJAHR, COL_CARRIER, COL_HEATING]:
        if c in gdf.columns:
            gdf[c] = gdf[c].apply(_norm_token)

    # Baujahresphasen + Energieträger + Heizungsart sektor-spezifisch normalisieren
    if COL_BAUJAHR in gdf.columns and COL_SECTOR in gdf.columns:
        gdf[COL_BAUJAHR] = gdf.apply(
            lambda r: _normalize_baujahr_label(r.get(COL_BAUJAHR, ""), sector=r.get(COL_SECTOR, "")),
            axis=1,
        )
    if COL_CARRIER in gdf.columns and COL_SECTOR in gdf.columns:
        gdf[COL_CARRIER] = gdf.apply(
            lambda r: _normalize_entraeger(r.get(COL_CARRIER, ""), sector=r.get(COL_SECTOR, "")),
            axis=1,
        )
    if COL_HEATING in gdf.columns and COL_SECTOR in gdf.columns:
        # Für NWG wird Heizungsart in der Referenztabelle i.d.R. nicht differenziert
        gdf[COL_HEATING] = gdf.apply(
            lambda r: ("" if r.get(COL_SECTOR, "") == "NWG" else _norm_token(r.get(COL_HEATING, ""))),
            axis=1,
        )

    # Referenz laden/pivotieren
    df_flat = load_iwu_reference_flat(iwu_base_dir)
    df_wide = _pivot_reference_wide(df_flat)

    qa_dir = None
    if out_dir is not None:
        qa_dir = Path(out_dir) / "ap2" / "qa"
        _write_qa(df_flat, qa_dir / "qa_iwu_reference_long_norm.csv")
        _write_qa(df_wide, qa_dir / "qa_iwu_reference_wide.csv")

    # Kennwerte anwenden + QA
    gdf, _ = _apply_reference_demands(gdf, df_wide, qa_dir=qa_dir)

    # Bedarfe berechnen
    gdf = _compute_building_demands(gdf)

    # Historische/unerwünschte Felder entfernen (falls vorhanden)
    drop_cols = [
        "q_rw_end_total_kwh_m2a",
        "q_tww_end_total_kwh_m2a",
    ]
    for c in drop_cols:
        if c in gdf.columns:
            gdf = gdf.drop(columns=[c])

    # Ergebnis schreiben
    _ensure_dir(out_gpkg_path.parent)
    gdf.to_file(out_gpkg_path, layer=layer_name, driver="GPKG")

    return out_gpkg_path
