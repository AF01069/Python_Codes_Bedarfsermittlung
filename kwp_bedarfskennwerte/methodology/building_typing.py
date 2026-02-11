"""
building_typing.py

AP2: Typisierung von Gebäuden (Wohn- und Nichtwohngebäude) und
     Ableitung von IWU-kompatiblen Klassen.

Zentrale Funktionen
-------------------
- load_and_merge(...)          : führt AP1-Gebäudelayer mit Zensus-CSV und Geometrie-CSV zusammen
- assign_wg_nwg(...)           : WG/NWG-Klassifikation je Gebäude
- assign_iwu_residential_type  : IWU-Typ für Wohngebäude (WG)
- assign_iwu_nwg_type          : IWU-Typ für Nichtwohngebäude (NWG, 11er-Typologie)
- assign_iwu_baujahresphase    : IWU-Baujahresphase (WG: 9 Klassen, NWG: 3 Klassen)
- flag_outbuildings            : markiert SMALL_OUTBUILDING und Energie-Relevanz
- run_building_typing(...)     : High-Level-Pipeline für AP2 (GPKG-Output)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")  # nicht-interaktives Backend für Skriptbetrieb
import matplotlib.pyplot as plt


PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Konstanten für Wohngebäude-Typen (IWU) und Mapping
# ---------------------------------------------------------------------------

IWU_TYPE_CODES = [
    "MFH_LARGE_GT_20",
    "MFH_MEDIUM_UP_TO_20",
    "MFH_SMALL_UP_TO_6",
    "ROW_HOUSE",
    "SEMI_DETACHED",
    "SFH_DETACHED",
    "SMALL_OUTBUILDING",
]

# Mapping aus der finalen Typ-Spezifikation (Zensus-Ergebnis / GPKG)
FINAL_LABEL_TO_IWU = {
    "EZFH freistehend": "SFH_DETACHED",
    "EZFH Doppel-haushälfte": "SEMI_DETACHED",
    "EZFH Doppelhaushälfte": "SEMI_DETACHED",
    "EZFH Reihenhaus (und sonstige)": "ROW_HOUSE",
    "EZFH Reihenhaus": "ROW_HOUSE",
    "MFH mit 3 bis 6 Wohnungen": "MFH_SMALL_UP_TO_6",
    "MFH mit 7 bis 20 Wohnungen": "MFH_MEDIUM_UP_TO_20",
    "MFH mit 21 oder mehr Wohnungen": "MFH_LARGE_GT_20",
}

# ---------------------------------------------------------------------------
# Baujahresphasen – Wohngebäude (IWU)
# ---------------------------------------------------------------------------

WG_IWU_BJ_PHASES = [
    "bis 1918",
    "1919 – 1948",
    "1919-1948",
    "1949 – 1978",
    "1949-1978",
    "1979 – 1994",
    "1979-1994",
    "1995 – 2001",
    "1995-2001",
    "2002 – 2009",
    "2002-2009",
    "2010 – 2015",
    "2010-2015",
    "2016 – 2020",
    "2016-2020",
    "2021 – 2025",
    "2021-2025",
]

# Normalisierte Schreibweise für WG-Phasen
WG_IWU_BJ_NORMALIZED = {
    "bis 1918": "bis 1918",
    "1919-1948": "1919 – 1948",
    "1919 – 1948": "1919 – 1948",
    "1949-1978": "1949 – 1978",
    "1949 – 1978": "1949 – 1978",
    "1979-1994": "1979 – 1994",
    "1979 – 1994": "1979 – 1994",
    "1995-2001": "1995 – 2001",
    "1995 – 2001": "1995 – 2001",
    "2002-2009": "2002 – 2009",
    "2002 – 2009": "2002 – 2009",
    "2010-2015": "2010 – 2015",
    "2010 – 2015": "2010 – 2015",
    "2016-2020": "2016 – 2020",
    "2016 – 2020": "2016 – 2020",
    "2021-2025": "2021 – 2025",
    "2021 – 2025": "2021 – 2025",
}

# Später nützlich für Sanierungsstatus:
WG_IWU_BJ_RECENT = [
    "1979 – 1994",
    "1995 – 2001",
    "2002 – 2009",
    "2010 – 2015",
    "2016 – 2020",
    "2021 – 2025",
]

# ---------------------------------------------------------------------------
# Baujahresphasen – Nichtwohngebäude (IWU-Logik)
# ---------------------------------------------------------------------------

NWG_IWU_BJ_PHASES = {
    "vor 1978": "vor 1978",
    "bis 1978": "vor 1978",
    "1978 bis 2010": "1978 bis 2010",
    "1978-2010": "1978 bis 2010",
    "ab 2010": "ab 2010",
    ">=2010": "ab 2010",
}

# ---------------------------------------------------------------------------
# Nichtwohngebäude – IWU-Gebäudetypen (11er-Typologie)
# ---------------------------------------------------------------------------

NWG_IWU_TYPE_BY_ID = {
    1: "Büro-, Verwaltungs- oder Amtsgebäude",
    2: "Gebäude für Forschung und Hochschullehre",
    3: "Gebäude für Gesundheit und Pflege",
    4: "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
    5: "Gebäude für Kultur und Freizeit",
    6: "Sportgebäude",
    7: "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
    8: "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
    9: "Handelsgebäude",
    10: "Technikgebäude (Ver- und Entsorgung)",
    11: "Verkehrsgebäude",
}

# Reverse Mapping für den Fall, dass im Datensatz die Texte stehen
NWG_IWU_TYPE_BY_NAME = {v: k for k, v in NWG_IWU_TYPE_BY_ID.items()}


# ---------------------------------------------------------------------------
# Daten laden & zusammenführen
# ---------------------------------------------------------------------------

import geopandas as gpd
import pandas as pd


import geopandas as gpd
import pandas as pd


def load_and_merge(ap1_gpkg: str, zensus_csv: str, geom_csv: str) -> gpd.GeoDataFrame:
    """
    Lädt und vereint die Eingangsdatensätze für AP2.

    Annahme:
    - ap1_gpkg (z.B. out/zensus/ap1_buildings_enriched_zensus.gpkg)
      enthält bereits alle ZENSUS-Attribute, Final_Baujahrklasse etc.
      (Ergebnis von ap1_enrich).

    - geom_csv ist die Geometrie-Analyse-CSV aus AP1
      (z.B. out/ap1/geometry_analysis/<LOD2>_geomstats_attributes.csv)
      mit u.a.:
        * LOD_UNITID
        * height_mean_m
        * footprint_area_m2
        * storeys_est
        * dwelling_est
        * n_adjacent_buildings
        * share_wall_to_neighbours
        * type_indicator
      usw.

    Das Zensus-CSV wird hier NICHT mehr aktiv verwendet, weil die
    Informationen bereits im GPKG enthalten sind. Der Parameter
    bleibt nur aus Kompatibilitätsgründen in der Signatur.
    """

    # 1) Gebäudelayer inkl. Zensus aus dem GPKG lesen
    gdf = gpd.read_file(ap1_gpkg)

    # 2) Geometrie-Attribute aus der LOD2-Geometrieanalyse lesen
    df_geom = pd.read_csv(geom_csv)

    # 3) Join-Schlüssel bestimmen: bevorzugt LOD_UNITID
    if "LOD_UNITID" in gdf.columns and "LOD_UNITID" in df_geom.columns:
        key_geom = "LOD_UNITID"
    elif "building_id" in gdf.columns and "building_id" in df_geom.columns:
        key_geom = "building_id"
    else:
        raise KeyError(
            "Für den Join zwischen GPKG und Geometrie-CSV wird entweder "
            "'LOD_UNITID' oder 'building_id' in beiden Tabellen benötigt.\n"
            f"GPKG-Spalten:  {list(gdf.columns)}\n"
            f"Geom-CSV-Spalten: {list(df_geom.columns)}"
        )

    # 4) Merge: Geometrie-Attribute an den bestehenden Gebäudelayer anhängen
    gdf_merged = gdf.merge(
        df_geom,
        on=key_geom,
        how="left",
        suffixes=("", "_geom"),
    )

    return gdf_merged





# ---------------------------------------------------------------------------
# WG / NWG – Grundtypisierung
# ---------------------------------------------------------------------------

def assign_wg_nwg(row: pd.Series) -> str:
    """
    Weist Gebäude als Wohngebäude (WG) oder Nichtwohngebäude (NWG) aus.

    Regel:
    1) Wenn 'Final_NWGoderWG' (Zensus) vorhanden ist → Wert übernehmen.
    2) Wenn nicht vorhanden → Standard: 'WG'.

    Keine Fallbacks über Nutzung oder Geometrie.
    """

    zensus_flag = (row.get("Final_NWGoderWG") or "").strip().upper()

    if zensus_flag in ("WG", "NWG"):
        return zensus_flag

    # Default, wenn keine verlässliche Zuweisung existiert:
    return "WG"



# ---------------------------------------------------------------------------
# Wohngebäude – IWU-Typisierung
# ---------------------------------------------------------------------------

def assign_iwu_residential_type(row: pd.Series) -> str:
    """
    Liefert einen der IWU_TYPE_CODES für Wohngebäude (WG).

    Annahme:
    - Diese Funktion wird nur für Zeilen mit WG_NWG == 'WG' aufgerufen.

    Priorität:
      1) 'type_indicator' aus der Geometrie-CSV (wenn vorhanden und gültig)
      2) Heuristische Ableitung aus Geometriekennwerten
         (footprint_area_m2, storeys_est, dwelling_est,
          n_adjacent_buildings, share_wall_to_neighbours, ...)

    Rückgabe ist NIE leer; jedes Wohngebäude erhält einen Typ.
    """

    # 1) type_indicator aus der Geometrieanalyse (CSV)
    ti_geom = (row.get("type_indicator") or "").strip()
    if ti_geom in IWU_TYPE_CODES:
        return ti_geom

    # 2) Heuristische Typisierung aus Geometrie
    return _heuristic_iwu_residential_type(row)



def _heuristic_iwu_residential_type(row: pd.Series) -> str:
    """
    Heuristische Typisierung für Wohngebäude (WG),
    wenn kein gültiger 'type_indicator' vorliegt.

    Verwendete Kennwerte aus der Geometrie-CSV:
      - footprint_area_m2
      - storeys_est  (einzige zuverlässige Geschossquelle)
      - durchschnFlaechejeWohn (Zensus)
      - n_adjacent_buildings
      - share_wall_to_neighbours

    Grundlogik:
      1) Sehr kleine Gebäude => SMALL_OUTBUILDING
      2) Kleine Wohngebäude (<= 2 Wohneinheiten & <= 220 m²)
         → SFH / SEMI_DETACHED / ROW_HOUSE
      3) Mehrfamilienhäuser → kleine / mittlere / große MFH
         über Wohnungszahl (dwell)
    """

    # --- Grundfläche (m²) ---
    area = (
        row.get("footprint_area_m2")
        or row.get("LOD_Grundflaeche_m2")
        or 0.0
    )

    # --- Geschosszahl: ausschließlich aus storeys_est ---
    storeys = row.get("storeys_est")
    if storeys is None or pd.isna(storeys):
        storeys = 1
    storeys = int(round(storeys))
    if storeys < 1:
        storeys = 1

    # --- Nachbarschaftsindikatoren ---
    neighbours = int(row.get("n_adjacent_buildings") or 0)

    share_wall = row.get("share_wall_to_neighbours")
    if share_wall is None or pd.isna(share_wall):
        share_wall = 0.0
    share_wall = float(share_wall)

    # --- geschätzte nutzbare Fläche ---
    total_floor_area = area * storeys

    # --- Kleine Nebengebäude erkennen ---
    if total_floor_area < 65:
        return "SMALL_OUTBUILDING"

    # --- Wohnungszahl: zuerst aus Zensus "durchschnFlaechejeWohn" ---
    avg_wfl = row.get("durchschnFlaechejeWohn")
    dwell = None

    if avg_wfl is not None and not pd.isna(avg_wfl):
        try:
            avg_wfl = float(avg_wfl)
            if avg_wfl > 10:  # realistische Mindestgröße für eine Wohnung
                dwell = total_floor_area / avg_wfl
        except Exception:
            dwell = None

    # --- Fallback: heuristische Annahme 80 m² pro Wohnung ---
    if dwell is None:
        dwell = total_floor_area / 80.0

    dwell = int(round(dwell))
    if dwell < 1:
        dwell = 1

    # --- 2) Kleine Wohngebäude (< =2 Wohnungen, <= 220 m² Gesamtfläche) ---
    if dwell <= 2 and total_floor_area <= 220:

        # Reihenhaus:
        # - mindestens zwei Nachbarn oder sehr hoher Anteil gemeinsamer Wand
        if neighbours >= 2 or share_wall >= 0.6:
            return "ROW_HOUSE"

        # Doppelhaushälfte:
        # - ein Nachbar oder mittlere gemeinsame Wandanteile
        if neighbours == 1 or (0.2 < share_wall < 0.6):
            return "SEMI_DETACHED"

        # ansonsten freistehend:
        return "SFH_DETACHED"

    # --- 3) Mehrfamilienhäuser: Klassifikation über Wohnungszahl ---
    if dwell <= 6:
        return "MFH_SMALL_UP_TO_6"
    if dwell <= 20:
        return "MFH_MEDIUM_UP_TO_20"
    return "MFH_LARGE_GT_20"





def flag_outbuildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Markiert kleine Nebengebäude (SMALL_OUTBUILDING) innerhalb der Wohngebäude.

    Erwartung:
    - Spalte 'IWU_WG_Type' ist schon gesetzt (nur für WG relevant).
    """

    if "IWU_WG_Typ" not in gdf.columns:
        # Sicherheitsnetz: falls noch nicht berechnet, jetzt nur für WG berechnen
        mask_wg = gdf["WG_NWG"] == "WG"
        gdf.loc[mask_wg, "IWU_WG_Typ"] = gdf[mask_wg].apply(
            assign_iwu_residential_type, axis=1
        )

    gdf["is_small_outbuilding"] = gdf["IWU_WG_Typ"] == "SMALL_OUTBUILDING"
    gdf["is_energy_relevant"] = ~gdf["is_small_outbuilding"]

    return gdf


# ---------------------------------------------------------------------------
# Nichtwohngebäude – IWU-Typisierung (11er-Typologie)
# ---------------------------------------------------------------------------

def assign_iwu_nwg_type(row: pd.Series) -> str:
    """
    Weist für ein Nichtwohngebäude den IWU-NWG-Typ (1..11) zu.

    Quelle:
      1) Spalte 'Final_NWG_Typ' (Zahl 1..11 oder Text laut Liste)
      2) Fallback Mapping aus 'Final_Nutzung_vereinheitlicht'

    Rückgabe: Textbezeichnung wie in NWG_IWU_TYPE_BY_ID.
    """

    if (row.get("WG_NWG") or "").strip() != "NWG":
        return ""  # nur für NWG sinnvoll

    # 1) Direkte Typ-Spalte (ID oder Name)
    raw_type = row.get("Final_NWG_Typ")  # Spaltenname ggf. anpassen!
    if raw_type is not None and not (isinstance(raw_type, float) and pd.isna(raw_type)):

        # Fall: numerische ID
        try:
            t_id = int(raw_type)
            if t_id in NWG_IWU_TYPE_BY_ID:
                return NWG_IWU_TYPE_BY_ID[t_id]
        except (ValueError, TypeError):
            pass

        # Fall: Textbezeichnung
        label = str(raw_type).strip()
        if label in NWG_IWU_TYPE_BY_NAME:
            return label

    # 2) Fallback: Final_Nutzung_vereinheitlicht grob mappen
    usage = (row.get("Final_Nutzung_vereinheitlicht") or "").lower()

    if any(s in usage for s in ["büro", "verwaltung", "amt"]):
        return NWG_IWU_TYPE_BY_ID[1]
    if any(s in usage for s in ["forschung", "hochschule", "uni"]):
        return NWG_IWU_TYPE_BY_ID[2]
    if any(s in usage for s in ["krankenhaus", "klinik", "pflege", "gesundheit", "arzt"]):
        return NWG_IWU_TYPE_BY_ID[3]
    if any(s in usage for s in ["schule", "kita", "kindertag", "betreuung"]):
        return NWG_IWU_TYPE_BY_ID[4]
    if any(s in usage for s in ["kultur", "theater", "museum", "freizeit"]):
        return NWG_IWU_TYPE_BY_ID[5]
    if any(s in usage for s in ["sport", "halle", "stadion"]):
        return NWG_IWU_TYPE_BY_ID[6]
    if any(s in usage for s in ["hotel", "pension", "beherberg", "gast", "restaurant", "kantine"]):
        return NWG_IWU_TYPE_BY_ID[7]
    if any(s in usage for s in ["produktion", "werkstatt", "lager", "halle", "betrieb"]):
        return NWG_IWU_TYPE_BY_ID[8]
    if any(s in usage for s in ["handel", "einkauf", "kaufhaus", "supermarkt", "laden"]):
        return NWG_IWU_TYPE_BY_ID[9]
    if any(s in usage for s in ["technik", "versorgung", "entsorgung", "heizwerk", "umspann"]):
        return NWG_IWU_TYPE_BY_ID[10]
    if any(s in usage for s in ["bahnhof", "flughafen", "terminal", "verkehr"]):
        return NWG_IWU_TYPE_BY_ID[11]

    # Fallback: "Büro/Verwaltung" als relativ neutrale Default-Kategorie
    return NWG_IWU_TYPE_BY_ID[1]


# ---------------------------------------------------------------------------
# Baujahresphasen (WG & NWG) – einheitliche Funktion
# ---------------------------------------------------------------------------

def assign_iwu_baujahresphase(row: pd.Series) -> str:
    """
    Weist für jedes Gebäude eine IWU-Baujahresphase zu.

    - Für WG: 9-teilige IWU-WG-Phasen
    - Für NWG: 3-teilige IWU-NWG-Phasen (vor 1978, 1978 bis 2010, ab 2010)

    Quelle: Spalte 'Final_Baujahrklasse' (aus AP1/AP1-Enrichment).
    """

    bj_raw = (row.get("Final_Baujahrklasse") or "").strip()
    if not bj_raw:
        return ""

    wg_nwg = (row.get("WG_NWG") or "").strip()

    # Wohngebäude: 1:1 in IWU-Phasen überführen
    if wg_nwg == "WG":
        # Schreibweise vereinheitlichen (Minus / Gedankenstrich etc.)
        key = bj_raw.replace("–", "-")
        if key in WG_IWU_BJ_NORMALIZED:
            return WG_IWU_BJ_NORMALIZED[key]

        # Fallback: wenn exakt so drin steht, akzeptieren
        if bj_raw in WG_IWU_BJ_PHASES:
            return WG_IWU_BJ_NORMALIZED.get(bj_raw, bj_raw)

        # sonst: heuristische Zuordnung über Zahlen
        # Erwartete Form: "YYYY-YYYY" oder "bis YYYY"
        if "bis" in bj_raw:
            # alles "bis XXXX" -> ersten Bereich WG-Logik
            return "bis 1918"
        # Einfacher Fallback auf "1949 – 1978"
        return "1949 – 1978"

    # Nichtwohngebäude: drei IWU-NWG-Phasen
    if wg_nwg == "NWG":
        key = bj_raw.replace("–", "-")
        if key in NWG_IWU_BJ_PHASES:
            return NWG_IWU_BJ_PHASES[key]
        if bj_raw in NWG_IWU_BJ_PHASES:
            return NWG_IWU_BJ_PHASES[bj_raw]

        # ganz grober Fallback: Jahreszahlen parsen
        years = list(map(int, re.findall(r"\d{4}", bj_raw)))
        if years:
            y = min(years)
            if y < 1978:
                return "vor 1978"
            if y < 2011:
                return "1978 bis 2010"
            return "ab 2010"

        # wenn gar nichts passt, konservativ "vor 1978"
        return "vor 1978"

    # falls WG_NWG leer/unklar: einfach Rohwert zurückgeben
    return bj_raw


# ---------------------------------------------------------------------------
# Zensus-Heizsystem (dominanter Energieträger) – optionaler Baustein
# ---------------------------------------------------------------------------

def dominant_zensus(row: pd.Series) -> str:
    """
    Liefert den dominanten Energieträger aus Zensus-Spalten.

    Erwartete Spalten (Anteile oder absoluten Werte):
    - GAS, OEL, FERN, STROM, BIOM, SONST
    """

    cols = ["GAS", "OEL", "FERN", "STROM", "BIOM", "SONST"]
    vals = {c: row.get(c) for c in cols if c in row.index}

    if not vals:
        return ""

    s = pd.Series(vals)
    if s.isna().all():
        return ""

    return s.idxmax()



def normalize_energietraeger(raw: object) -> str:
    """Normalisiert Energieträger-Codes auf einen stabilen, IWU-nahen Satz.

    Ziel ist, dass Downstream (heat_demand) robuste und eindeutige Codes erhält.
    Rückgabecodes:
      - FW   (Fern-/Nahwärme)
      - GAS
      - OEL
      - BIOM (Biomasse / Holz / Pellet)
      - KOHLE
      - EL   (Strom / Direkt)
      - WP   (Wärmepumpe)
      - SONST
      - ''   (leer / unbekannt)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip().upper()

    # häufige Varianten / Tippfehler
    s = s.replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE")

    if s in {"FERN", "FW", "FERNWAERME", "FERNWÄRME", "NAHWAERME", "NAHWÄRME"}:
        return "FW"
    if s in {"GAS", "ERDGAS"}:
        return "GAS"
    if s in {"OEL", "OE L", "HEIZOEL", "HEIZÖL", "OELK", "OELKessel".upper()}:
        return "OEL"
    if s in {"BIOM", "BIOMASSE", "HOLZ", "PELLET", "HACKSCHNITZEL", "SCHEITHOLZ"}:
        return "BIOM"
    if s in {"KOHLE", "KOKS"}:
        return "KOHLE"
    if s in {"STROM", "EL", "ELEKTRISCH", "ELEKTRO"}:
        return "EL"
    if s in {"WP", "WAERMEPUMPE", "WÄRMEPUMPE", "WAERMEPUMPE".upper()}:
        return "WP"
    if s in {"SONST", "SONSTIGES", "OTHER"}:
        return "SONST"

    # Wenn schon ein normierter Code, durchreichen
    if s in {"FW", "GAS", "OEL", "BIOM", "KOHLE", "EL", "WP"}:
        return s

    return "SONST"

# ---------------------------------------------------------------------------
# Sanierungsstatus – heuristisches Modell (noch nicht in Pipeline verdrahtet)
# ---------------------------------------------------------------------------

def assign_sanierungsstatus(row: pd.Series) -> str:
    """
    Heuristische Abschätzung des Sanierungszustands.

    Erwartete Spalten:
    - eigentumsquote      (0..1)
    - leerstandsquote     (0..1)
    - is_denkmalschutz    (bool)
    - IWU_Baujahresphase  (Text wie in WG_IWU_BJ_NORMALIZED oder NWG-Phasen)

    Kategorien:
    - 'modernisiert'
    - 'teilmodernisiert'
    - 'unsaniert'
    """

    eigentum = row.get("eigentumsquote") or 0.0
    leerstand = row.get("leerstandsquote") or 0.0
    is_denkmal = bool(row.get("is_denkmalschutz"))
    bj_phase = (row.get("IWU_Baujahresphase") or "").strip()

    idx = (
        0.4 * eigentum -
        0.3 * leerstand +
        0.4 * (0 if is_denkmal else 1) +
        0.2 * (1 if bj_phase in WG_IWU_BJ_RECENT else 0)
    )

    if idx > 0.5:
        return "modernisiert"
    if idx > 0.2:
        return "teilmodernisiert"
    return "unsaniert"


# ----------------------------------------------------------------
# Hilfsfunktion für die Ausgabe: Plots Gebäudetypisierung
# ----------------------------------------------------------------
def _plot_categorical_distribution(series: pd.Series, title: str, out_png: Path) -> None:
    """
    Erzeugt ein einfaches Balkendiagramm der Häufigkeitsverteilung
    einer kategorialen Variable und speichert es als PNG.
    """
    if series is None or series.empty:
        return

    s = (
        series
        .astype("string")
        .fillna("<<MISSING>>")
        .replace("", "<<EMPTY>>")
    )
    vc = s.value_counts().sort_values(ascending=False)

    if vc.empty:
        return

    plt.figure(figsize=(max(6, 0.6 * len(vc)), 4))
    vc.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Anzahl Gebäude")
    plt.xticks(rotation=45, ha="right")
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*")
        plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ----------------------------------------------------------------
# Hilfsfunktion für die Ausgabe: Statistik Gebäudetypisierung
# ----------------------------------------------------------------
def _write_typing_stats(gdf: gpd.GeoDataFrame, out_gpkg: PathLike) -> None:
    """
    Schreibt eine Textdatei mit Häufigkeitsverteilungen und Missing-Anteilen
    sowie einige einfache Diagramme.

    Es werden jeweils Auswertungen erstellt für:
    - alle Gebäude
    - nur energie-relevante Gebäude (is_energy_relevant == True)
    """

    out_path = Path(out_gpkg)
    stats_txt = out_path.with_name(out_path.stem + "_stats.txt")

    # Kern-Kennwerte, die wir betrachten wollen
    key_cols = [
        "WG_NWG",
        "Final_Nutzung_vereinheitlicht",
        "IWU_WG_Typ",
        "IWU_NWG_Typ",
        "IWU_Baujahresphase",
        "Final_Energietraeger",
        "Final_Heizungsart",
        "Final_Nutzflaeche_m2",
    ]

    # Subsets: alle Gebäude / energie-relevant
    mask_all = pd.Series(True, index=gdf.index)
    if "is_energy_relevant" in gdf.columns:
        mask_energy = gdf["is_energy_relevant"].fillna(False)
    else:
        mask_energy = pd.Series(False, index=gdf.index)

    subsets = {
        "ALLE_GEBÄUDE": mask_all,
        "ENERGIE_RELEVANT": mask_energy,
    }

    with stats_txt.open("w", encoding="utf-8") as f:
        f.write("AP2 – Statistik Gebäudetypisierung\n")
        f.write(f"GPKG-Datei: {out_path}\n")
        f.write(f"Anzahl Gebäude gesamt: {len(gdf)}\n\n")

        for label, mask in subsets.items():
            sub = gdf[mask].copy()
            f.write("=" * 72 + "\n")
            f.write(f"STATISTIK FÜR: {label}\n")
            f.write(f"Anzahl Gebäude in diesem Subset: {len(sub)}\n")
            f.write("=" * 72 + "\n\n")

            # Missing-Gebäude über alle Kernfelder (mindestens ein Feld leer)
            missing_any = pd.Series(False, index=sub.index)

            for col in key_cols:
                if col not in sub.columns:
                    continue

                s = sub[col]
                # Missing-Definition: NaN oder leerer String
                if s.dtype == "float" or s.dtype == "int":
                    miss_mask = s.isna()
                else:
                    s_str = s.astype("string")
                    miss_mask = s_str.isna() | (s_str.str.strip() == "")

                missing_count = int(miss_mask.sum())
                total = len(sub)
                missing_pct = 100.0 * missing_count / total if total > 0 else 0.0

                missing_any = missing_any | miss_mask

                f.write(f"--- Feld: {col}\n")
                f.write(f"  Anzahl gültige Werte: {total - missing_count}\n")
                f.write(f"  Anzahl fehlende Werte: {missing_count} ({missing_pct:.1f} %)\n")

                # Häufigkeitsverteilung (inkl. Missing/Empty markiert)
                if col != "Final_Nutzflaeche_m2":
                    vc = (
                        s.astype("string")
                        .fillna("<<MISSING>>")
                        .replace("", "<<EMPTY>>")
                        .value_counts()
                        .sort_values(ascending=False)
                    )
                    f.write("  Häufigkeitsverteilung:\n")
                    for val, cnt in vc.items():
                        pct = 100.0 * cnt / total if total > 0 else 0.0
                        f.write(f"    {val}: {cnt} ({pct:.1f} %)\n")
                else:
                    # numerische Basisstatistik für Final_Nutzflaeche_m2
                    s_num = pd.to_numeric(s, errors="coerce")
                    s_num = s_num.dropna()
                    if not s_num.empty:
                        f.write("  Numerische Kenngrößen (Final_Nutzflaeche_m2):\n")
                        f.write(f"    min:   {s_num.min():.1f} m²\n")
                        f.write(f"    max:   {s_num.max():.1f} m²\n")
                        f.write(f"    mean:  {s_num.mean():.1f} m²\n")
                        f.write(f"    p25:   {s_num.quantile(0.25):.1f} m²\n")
                        f.write(f"    p50:   {s_num.quantile(0.50):.1f} m²\n")
                        f.write(f"    p75:   {s_num.quantile(0.75):.1f} m²\n")
                    else:
                        f.write("  Numerische Kenngrößen (Final_Nutzflaeche_m2): keine gültigen Werte.\n")

                f.write("\n")

            # Gebäude mit fehlenden Werten in mindestens einem Kernfeld
            miss_any_count = int(missing_any.sum())
            total = len(sub)
            miss_any_pct = 100.0 * miss_any_count / total if total > 0 else 0.0
            f.write(">>> Gebäude mit fehlenden Werten in mind. einem Kernfeld:\n")
            f.write(f"    Anzahl: {miss_any_count} ({miss_any_pct:.1f} %)\n\n\n")

    # -------------------------------------------------
    # Plots für einige zentrale kategoriale Variablen
    # -------------------------------------------------
    plot_cols = {
        "WG_NWG": "Verteilung WG/NWG",
        "Final_Nutzung_vereinheitlicht": "Verteilung Final_Nutzung_vereinheitlicht",
        "IWU_Baujahresphase": "Verteilung IWU_Baujahresphase",
        "Final_Energietraeger": "Verteilung Final_Energietraeger",
    }

    for col, title in plot_cols.items():
        if col not in gdf.columns:
            continue

        # Alle Gebäude
        png_all = out_path.with_name(out_path.stem + f"_{col}_ALL.png")
        _plot_categorical_distribution(gdf[col], f"{title} – alle Gebäude", png_all)

        # Energie-relevante Gebäude
        if "is_energy_relevant" in gdf.columns:
            gdf_energy = gdf[gdf["is_energy_relevant"].fillna(False)]
            if not gdf_energy.empty:
                png_energy = out_path.with_name(out_path.stem + f"_{col}_ENERGY.png")
                _plot_categorical_distribution(
                    gdf_energy[col],
                    f"{title} – energie-relevante Gebäude",
                    png_energy,
                )

# ---------------------------------------------------------------------------
# High-Level-Pipeline für AP2
# ---------------------------------------------------------------------------

def run_building_typing(ap1_gpkg: PathLike,
                        zensus_csv: PathLike,
                        geom_csv: PathLike,
                        out_gpkg: PathLike) -> None:
    """
    Führt die komplette Typisierung durch und schreibt
    ein GPKG mit typisierten Gebäuden.

    Output-Spalten (Auszug):
    - WG_NWG
    - IWU_WG_Typ        (für WG)
    - IWU_NWG_Typ        (für NWG, Text der 11er-Typologie)
    - IWU_Baujahresphase (WG: 9 Klassen, NWG: 3 Klassen)
    - is_small_outbuilding, is_energy_relevant
    """

    gdf = load_and_merge(ap1_gpkg, zensus_csv, geom_csv)

    # 1) WG / NWG
    gdf["WG_NWG"] = gdf.apply(assign_wg_nwg, axis=1)

    # 2) Wohngebäude-Typen (IWU / Outbuilding-Flag)
    mask_wg = gdf["WG_NWG"] == "WG"
    gdf.loc[mask_wg, "IWU_WG_Typ"] = gdf[mask_wg].apply(
        assign_iwu_residential_type, axis=1
    )
    gdf = flag_outbuildings(gdf)

    # 3) Nichtwohngebäude-Typen (IWU-NWG 1..11)
    mask_nwg = gdf["WG_NWG"] == "NWG"
    gdf.loc[mask_nwg, "IWU_NWG_Typ"] = gdf[mask_nwg].apply(
        assign_iwu_nwg_type, axis=1
    )

    # -------------------------------------------------------------
    # 4) IWU-Baujahresphase – immer gesetzt
    # -------------------------------------------------------------
    gdf["IWU_Baujahresphase"] = gdf.apply(assign_iwu_baujahresphase, axis=1)

    # Fallback: fehlende Baujahresphase → häufigste Klasse aus Zensus
    mask_missing_bj = gdf["IWU_Baujahresphase"].isna() | (gdf["IWU_Baujahresphase"] == "")
    if mask_missing_bj.any():
        # häufigste (modus) Final_Baujahrklasse aus AP1/Zensus
        if "Final_Baujahrklasse" in gdf.columns:
            bj_mode = (
                gdf["Final_Baujahrklasse"]
                .dropna()
                .replace("", pd.NA)
                .mode()
            )
            bj_mode = bj_mode.iloc[0] if len(bj_mode) > 0 else "1949 – 1978"
        else:
            bj_mode = "1949 – 1978"

        gdf.loc[mask_missing_bj, "IWU_Baujahresphase"] = bj_mode

    # -------------------------------------------------------------
    # 5) Energieträger – stabil übernehmen (nicht überschreiben!)
    # -------------------------------------------------------------
    # In AP1/AP1-Enrichment existiert i.d.R. bereits ein (dominanter) Energieträger.
    # Der bisherige Code hat ihn überschrieben -> führt zu falschen Heat-Demand-Matches.
    if "Final_Energietraeger" not in gdf.columns:
        gdf["Final_Energietraeger"] = ""

    # vorhandene Werte normalisieren
    gdf["Final_Energietraeger"] = gdf["Final_Energietraeger"].apply(normalize_energietraeger)

    # Fehlende Werte: aus Zensus-Spalten ableiten (dominant_zensus liefert z.B. GAS/OEL/FERN/...)
    mask_missing_et = gdf["Final_Energietraeger"].isna() | (gdf["Final_Energietraeger"] == "")
    if mask_missing_et.any():
        derived = gdf.loc[mask_missing_et].apply(dominant_zensus, axis=1)
        derived = derived.apply(normalize_energietraeger)
        gdf.loc[mask_missing_et, "Final_Energietraeger"] = derived

    # Fallback: wenn immer noch leer -> häufigster Energieträger im Datensatz
    mask_missing_et2 = gdf["Final_Energietraeger"].isna() | (gdf["Final_Energietraeger"] == "")
    if mask_missing_et2.any():
        et_mode = (
            gdf["Final_Energietraeger"]
            .replace("", pd.NA)
            .dropna()
            .mode()
        )
        et_mode = et_mode.iloc[0] if len(et_mode) > 0 else "GAS"
        gdf.loc[mask_missing_et2, "Final_Energietraeger"] = et_mode

    # zusätzliche, für Downstream hilfreiche Systemgruppe (identisch zu Final_Energietraeger)
    gdf["IWU_Systemgruppe"] = gdf["Final_Energietraeger"]

# -------------------------------------------------------------
    # 6) Heizungsart – passend zum Energieträger (Fallback-Regeln)
    # -------------------------------------------------------------
    def infer_heizungsart(row):
        """
        F?hrt infer_heizungsart aus.
        
        Args:
            row: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        val = row.get("Final_Heizungsart")

        # bereits vorhandene Angabe verwenden, wenn nicht leer / NA
        if val is not None and not pd.isna(val):
            s = str(val).strip()
            if s != "":
                return s

        # Kategorische Heizungsarten (robust für Heat-Demand):
        #   - Fernheizung (FW)
        #   - Zentralheizung (GAS/OEL/BIOM/KOHLE/WP)
        #   - Etagenheizung (EL bzw. typisch dezentral)
        et = normalize_energietraeger(row.get("Final_Energietraeger") or row.get("IWU_Systemgruppe"))

        if et == "FW":
            return "Fernheizung"
        if et in {"GAS", "OEL", "BIOM", "HOLZ", "PELLET", "KOHLE", "WP"}:
            return "Zentralheizung"
        if et == "EL":
            # ohne weitere Info: in Bestandsdaten häufig dezentrale Direktheizung
            return "Etagenheizung"

        # generischer Fallback
        return "Zentralheizung"

    gdf["Final_Heizungsart"] = gdf.apply(infer_heizungsart, axis=1)

    # -------------------------------------------------------------
    # 7) Final_Nutzflaeche_m2 – gewährleisten, dass immer ein Wert existiert
    # -------------------------------------------------------------
    def infer_nutzflaeche(row):
        """
        F?hrt infer_nutzflaeche aus.
        
        Args:
            row: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        nf = row.get("Final_Nutzflaeche_m2")

        # robust in float umwandeln
        try:
            nf_val = float(nf)
        except (TypeError, ValueError):
            nf_val = None

        if nf_val is not None and nf_val > 0:
            return nf_val

        # fallback: footprint_area × storeys_est
        area = (
            row.get("footprint_area_m2")
            or row.get("LOD_Grundflaeche_m2")
            or 0
        )

        storeys = (
            row.get("storeys_est")
            or row.get("LOD_Stockwerke")
            or 1
        )
        try:
            storeys = int(round(float(storeys)))
        except Exception:
            storeys = 1

        return max(area * storeys, 20.0)  # Mindestwert, um Division/0 zu verhindern

    gdf["Final_Nutzflaeche_m2"] = gdf.apply(infer_nutzflaeche, axis=1)


    # -------------------------------------------------------------
    # 7b) Zusätzliche Regeln: Kleingebäude & bestimmte Nutzungen
    # -------------------------------------------------------------

    # Sicherstellen, dass die Flags existieren (falls später einmal geändert wird)
    if "is_small_outbuilding" not in gdf.columns:
        gdf["is_small_outbuilding"] = False
    if "is_energy_relevant" not in gdf.columns:
        gdf["is_energy_relevant"] = True

    # 1) Gebäude mit Nutzfläche < 60 m² → nicht energierelevant & Kleingebäude
    nf = pd.to_numeric(gdf["Final_Nutzflaeche_m2"], errors="coerce")
    mask_small_area = nf < 60
    gdf.loc[mask_small_area, "is_energy_relevant"] = False
    gdf.loc[mask_small_area, "is_small_outbuilding"] = True

    # 2) Bestimmte Nutzungen (Kleingarten, Wochenendhaus, Kapelle) → nicht energierelevant
    if "Final_Nutzung_original" in gdf.columns:
        usage_raw = (
            gdf["Final_Nutzung_original"]
            .astype("string")
            .fillna("")
            .str.strip()
        )
        usage_lower = usage_raw.str.lower()

        # Wir erlauben sowohl exakte Namen als auch Teilstrings
        NON_RELEVANT_EXACT = {
            "wochenend- und ferienhausfläche",
            "kleingarten",
            "kapelle",
        }
        NON_RELEVANT_KEYWORDS = [
            "kleingarten",      # fängt auch "kleingartenfläche" usw. ab
            "wochenend",        # "wochenend-" / "wochenendhaus"
            "ferienhaus",
            "kapelle",
        ]

        mask_exact = usage_lower.isin(NON_RELEVANT_EXACT)
        # Teilstring-Suche (OR-Verknüpfung über Keywords)
        pattern = "|".join(NON_RELEVANT_KEYWORDS)
        mask_substr = usage_lower.str.contains(pattern, regex=True)

        mask_usage = mask_exact | mask_substr

        gdf.loc[mask_usage, "is_energy_relevant"] = False
        gdf.loc[mask_usage, "is_small_outbuilding"] = True

    # Kommentar-Hinweis zur Erweiterbarkeit:
    # TODO: Diese Liste sollte für andere Regionen/Gebiete noch erweitert werden.

    # -------------------------------------------------------------
    # 8) Sicherstellen: WG/NWG-Typen sind vollständig befüllt
    # -------------------------------------------------------------
    # WG
    mask_wg = gdf["WG_NWG"] == "WG"
    mask_wg_missing = gdf["IWU_WG_Typ"].isna() | (gdf["IWU_WG_Typ"] == "")
    if (mask_wg & mask_wg_missing).any():
        gdf.loc[mask_wg & mask_wg_missing, "IWU_WG_Typ"] = gdf[mask_wg & mask_wg_missing].apply(
            assign_iwu_residential_type, axis=1
        )


    # NWG
    mask_nwg = gdf["WG_NWG"] == "NWG"
    mask_nwg_missing = gdf["IWU_NWG_Typ"].isna() | (gdf["IWU_NWG_Typ"] == "")
    if (mask_nwg & mask_nwg_missing).any():
        gdf.loc[mask_nwg & mask_nwg_missing, "IWU_NWG_Typ"] = gdf[mask_nwg & mask_nwg_missing].apply(
            assign_iwu_nwg_type, axis=1
        )

    # -------------------------------------------------------------
    # 9) Statistik-Textdatei + Diagramme (auf vollem Datensatz)
    # -------------------------------------------------------------
    _write_typing_stats(gdf, out_gpkg)

    # -------------------------------------------------------------
    # 10) Finale Spaltenauswahl + Umbenennung für AP2-Typing-GPKG
    # -------------------------------------------------------------
    # wir behalten alle Spalten, die für Heat-Demand benötigt werden
    # und bereinigen nur den Output nach außen
    geom_col = gdf.geometry.name  # meist "geometry"

    desired_cols = [
        "LOD_UNITID",
        "Final_Nutzung_original",
        "Final_Nutzung_Quelle",
        "DIVIS_flag",
        "DIVIS_ext_kurzcharakteristik",
        "DIVIS_Baujahr_Extrakt",
        "GITTER_ID_100m",
        "AnteilUeber65",
        "Eigentuemerquote",
        "Leerstandsquote",
        "Final_Baujahrklasse",
        "Final_Stockwerke_schaetzung",
        "Final_Nutzflaeche_m2",
        "height_mean_m",
        "footprint_area_m2",
        "volume_est_m3",
        "n_adjacent_buildings",
        "WG_NWG",
        "IWU_WG_Typ",
        "is_small_outbuilding",
        "IWU_NWG_Typ",
        "IWU_Baujahresphase",
        "Final_Heizungsart",
        "Final_Energietraeger",
        # HK-DE Adresse (für Klimakorrektur via PLZ):
        "HK_postplz",
        "HK_match",
        # für interne Logik/Heat-Demand sinnvoll, daher mitnehmen:
        "is_energy_relevant",
        "Final_Nutzung_vereinheitlicht",
        # Geometrie unbedingt behalten:
        geom_col,
    ]

    # nur Spalten verwenden, die es tatsächlich gibt
    keep_cols_existing = [c for c in desired_cols if c in gdf.columns]

    missing_cols = [
        c for c in desired_cols
        if c not in gdf.columns and c != geom_col
    ]
    if missing_cols:
        print(
            "[ap2][WARN] Folgende erwartete Spalten fehlen im Typing-Ergebnis "
            "und können nicht in die finale Datei übernommen werden:",
            missing_cols,
        )

    gdf_out = gdf[keep_cols_existing].copy()

    # Umbenennungen
    rename_map = {
        "DIVIS_ext_kurzcharakteristik": "DIVIS_info",
        "DIVIS_Baujahr_Extrakt": "DIVIS_year",
        "GITTER_ID_100m": "ZENSUS_GitterID",
        "AnteilUeber65": "ZENSUS_Anteil65Plus",
        "Eigentuemerquote": "ZENSUS_EigQuote",
        "Leerstandsquote": "ZENSUS_LeerQuote",
        "Final_Stockwerke_schaetzung": "Final_Etagen",
        "Final_Nutzflaeche_m2": "Final_ANutz",
        "height_mean_m": "Final_Hoehe",
        "footprint_area_m2": "Final_AGrund",
        "volume_est_m3": "Final_Vges",
        "n_adjacent_buildings": "Final_AngrGeb",
        "is_small_outbuilding": "Final_kleinGeb",
        "Final_Heizungsart": "IWU_Heizungsart",
        "Final_Energietraeger": "IWU_EnTraeger",
        # die restlichen behalten ihren Namen:
        # LOD_UNITID, Final_Nutzung_original, Final_Nutzung_Quelle,
        # DIVIS_flag, Final_Baujahrklasse, WG_NWG, IWU_WG_Typ,
        # IWU_NWG_Typ, IWU_Baujahresphase, is_energy_relevant,
        # Final_Nutzung_vereinheitlicht, ZENSUS_* usw.
    }

    gdf_out = gdf_out.rename(columns=rename_map)

    # Jahresfelder als Integer sichern (GPKG-Datentypen)
    for col in ("DIVIS_year", "Final_Baujahr_Mitte"):
        if col in gdf_out.columns:
            gdf_out[col] = pd.to_numeric(gdf_out[col], errors="coerce").astype("Int64")

    # sicherstellen, dass die Geometriespalte korrekt gesetzt bleibt
    gdf_out.set_geometry(geom_col, inplace=True)

    # -------------------------------------------------------------
    # 11) Finale Typing-GPKG schreiben
    # -------------------------------------------------------------
    gdf_out.to_file(out_gpkg, layer="buildings_typed", driver="GPKG")

