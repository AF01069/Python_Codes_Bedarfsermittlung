"""
AP1-Enrichment: Zensus, GHS-OBAT, DIVIS und Adressen.

Erg?nzt den AP1-Geb?udelayer um:
- Zensus-100m-Gitterattribute (Baujahrklassen, Heiztr?ger)
- GHS-OBAT (Baualtersattribute, Formfaktoren)
- DIVIS-Denkmalliste (WMS-Abfrage)
"""
# workflows/ap1_enrich.py
from __future__ import annotations
from pathlib import Path

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlencode

from ..data_catalog.sources import DivisWmsSource
from ..config.runtime import PipelineContext
from ..data_catalog.analyzer import AP1EnrichCSVStatistics

# --- HK-DE Adressanreicherung (PLZ etc.) ---
try:
    # Empfohlener Ort: kwp_bedarfskennwerte/utils/addresses_hk.py
    from ..utils.addresses_hk import enrich_buildings_with_hk_addresses, HKAddressColumns
except Exception:  # pragma: no cover
    # Fallback: falls das Modul (noch) nicht ins Paket verschoben wurde
    from ..utils.addresses_hk import enrich_buildings_with_hk_addresses, HKAddressColumns



def _gpkg_feature_count(path: Path, layer: str) -> int | None:
    """Return feature count for a GPKG layer without loading geometries (fast).

    Returns None if file/layer cannot be read.
    """
    try:
        import fiona  # type: ignore
        with fiona.open(path, layer=layer) as src:
            return len(src)
    except Exception:
        return None


def _try_load_cached_layer(
    path: Path,
    layer: str,
    *,
    expected_len: int | None = None,
    verbose: bool = True,
    label: str = "cached",
):
    """Load a cached GPKG layer only if it looks compatible with current run.

    Compatibility check is based on feature count (expected_len). If expected_len
    is provided and does not match, the cache is ignored to avoid accidentally
    reusing huge outputs from earlier runs (which can explode DIVIS/WMS calls).
    """
    if not path.exists():
        return None
    if expected_len is not None:
        cnt = _gpkg_feature_count(path, layer)
        if cnt is None:
            if verbose:
                print(f"[ap1_enrich] WARN: Konnte Feature-Count von {label} nicht lesen: {path}")
            return None
        if cnt != expected_len:
            if verbose:
                print(
                    f"[ap1_enrich] WARN: {label} hat {cnt} Features, erwartet {expected_len}. "
                    "Ignoriere vorhandene Datei."
                )
            return None

    try:
        return gpd.read_file(path, layer=layer)
    except Exception as exc:
        if verbose:
            print(f"[ap1_enrich] WARN: Konnte {label} nicht laden: {path} ({exc})")
        return None


def _get_settings(ctx: PipelineContext):
    """
    Erlaubt sowohl ctx.settings.* als auch ctx.*.
    Damit funktioniert das Enrichment unabhängig davon,
    ob der Context ein 'settings'-Attribut hat oder nicht.
    """
    return getattr(ctx, "settings", ctx)


def _get_data_cfg(settings):
    """Hilfsfunktion: Daten-Unterkonfiguration (settings.data) ermitteln."""
    return getattr(settings, "data", settings)


def _extract_year_from_text(text):
    """
    Extrahiert ein Baujahr aus einem Textfeld.

    Logik:
    1) Wenn eine vierstellige Jahreszahl vorkommt (z. B. '1875', 'ca. 1920-1930'),
       wird die erste solche Zahl verwendet.
    2) Falls keine vierstellige Jahreszahl vorkommt, wird nach Jahrhundert-
       Angaben gesucht, z. B. '19. Jh.', '19. Jh', '19. Jahrhundert'.
       In diesem Fall wird das Baujahr als (Jahrhundert * 100) gesetzt,
       also z. B.:
         - '19. Jh.'  -> 1900
         - '18. Jh.'  -> 1800
         - '20. Jh.'  -> 2000
    """
    if not isinstance(text, str):
        return pd.NA

    # 1) Vierstellige Jahreszahl suchen
    m_year = re.search(r"\b(\d{4})\b", text)
    if m_year:
        year = int(m_year.group(1))
        if 1200 <= year <= 2100:
            return year

    # 2) Jahrhundertangaben wie '19. Jh.', '19. Jh', '19. Jahrhundert'
    m_century = re.search(
        r"(\d{1,2})\s*\.->\s*(->:Jh\.->|Jahrhundert)",
        text,
        flags=re.IGNORECASE,
    )
    if m_century:
        century = int(m_century.group(1))
        # z. B. 19 -> 1900, 18 -> 1800
        year = century * 100
        if 1200 <= year <= 2100:
            return year

    return pd.NA


def _extract_year_from_text(text) -> pd.Series:
    """
    Extrahiert die erste vierstellige Jahreszahl aus einem Textfeld.
    - Erwartet z. B. Werte wie 'um 1880', '1875/1900', 'ca. 1920-1930'.
    - Gibt ein Integer-Baujahr zurück (z. B. 1880) oder <NA>, wenn nichts Plausibles gefunden wird.
    """
    if not isinstance(text, str):
        return pd.NA
    m = re.search(r"\d{4}", text)
    if not m:
        return pd.NA
    year = int(m.group(0))
    if year < 1200 or year > 2100:
        return pd.NA
    return year

def _get_settings(ctx: PipelineContext):
    """
    Hilfsfunktion: erlaubt sowohl ctx.settings.* als auch ctx.* als Zugriff.
    Damit kann das Enrichment sowohl mit einem echten PipelineContext als auch
    mit einem einfachen SimpleNamespace aus der CLI verwendet werden.
    """
    return getattr(ctx, "settings", ctx)

def _get_data_cfg(settings):
    """Hilfsfunktion: Daten-Unterkonfiguration (settings.data) ermitteln."""
    return getattr(settings, "data", settings)

def _load_ap1_buildings(ctx: PipelineContext) -> gpd.GeoDataFrame:
    """
    Lädt den Gebäudelayer aus dem AP1-Ergebnis (GPKG oder CSV).
    Annahme:
    - AP1 schreibt ein GPKG mit einem Layer 'buildings' ODER
    - eine CSV mit WKT-Geometrie.
    """
    settings = _get_settings(ctx)
    out_dir = Path(getattr(settings, "out_dir", "out"))
    ap1_out_dir = out_dir / "ap1" / "compare"

    # Default-GPKG: out/ap1/compare/ nutzungsspezifikation_vereinheitlicht.gpkg
    gpkg_path = ap1_out_dir / "nutzungsspezifikation_vereinheitlicht.gpkg"

    if gpkg_path.exists():
        return gpd.read_file(gpkg_path, layer="buildings")

    # Fallback: CSV-Variante
    csv_path = ap1_out_dir / "ap1_buildings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"AP1-Ergebnis nicht gefunden (weder GPKG noch CSV): {gpkg_path} / {csv_path}"
        )

    # TODO[AP1]: Falls deine CSV Geometrien in WKT enthält, hier WKT->Geo umsetzen.
    # Beispiel:
    # from shapely import wkt
    # df = pd.read_csv(csv_path)
    # df["geometry"] = df["wkt"].apply(wkt.loads)
    # gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25833")
    # return gdf
    df = pd.read_csv(csv_path)
    raise NotImplementedError("TODO: CSV->GeoDataFrame-Konvertierung implementieren")


def _extract_year_from_text(text) -> pd.Series:
    """
    Extrahiert die erste vierstellige Jahreszahl aus einem Textfeld.
    Erwartet z. B. 'um 1880', '1875/1900', 'ca. 1920-1930'.
    """
    if not isinstance(text, str):
        return pd.NA
    m = re.search(r"\d{4}", text)
    if not m:
        return pd.NA
    year = int(m.group(0))
    if year < 1200 or year > 2100:
        return pd.NA
    return year


def _get_settings(ctx: PipelineContext):
    return getattr(ctx, "settings", ctx)


def _get_data_cfg(settings):
    return getattr(settings, "data", settings)

def _fetch_divis_wms_for_area(
    bbox_25833: tuple[float, float, float, float],
    target_epsg: int,
    out_divis_dir: Path,
) -> gpd.GeoDataFrame:
    """
    Holt DIVIS-Daten via WMS (L1,L2,L3) für eine BBOX in EPSG:25833
    und speichert sie vor dem Mapping in out_divis_dir.

    WICHTIG:
    Der Dienst liefert die Geometrien effektiv in Gauss-Krüger Zone 5
    (EPSG:31469), auch wenn EPSG:25833 als CRS angefragt wird.
    Deshalb:
    - wir interpretieren die WMS-Geometrien als EPSG:31469
    - und transformieren sie danach nach target_epsg (i.d.R. 25833).
    """
    BASE_URL = "https://cardomap3.idu.de/lfds/public/ogc.ashx"
    COMMON_PARAMS = {
        "PKGID": 5,
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "LAYERS": "L1,L2,L3",        # Flächen, Linien, Punkte
        "QUERY_LAYERS": "L1,L2,L3",
        "STYLES": "",
        # Der Dienst verwendet faktisch 31469-Koordinaten,
        # auch wenn wir 25833 angeben – wir korrigieren das später explizit.
        "CRS": "EPSG:25833",
        "INFO_FORMAT": "application/geo+json",
    }

    # HTTP-Session mit Retries
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))

    # Höhere Auflösung + dichteres Sampling:
    #  - 2048x2048 Bild
    #  - step_px=2 → ca. 1024*1024 = 1 Mio. Abtastpunkte
    # Für dein relativ kleines Untersuchungsgebiet sollte das noch vertretbar sein.
    width = height = 2048

    def sample_pixels(step_px: int = 2):
        """
        Erzeugt eine Liste von Pixelpositionen (I,J) innerhalb des Bildes.
        Mit width=height=2048 und step_px=2 ergibt das 1024*1024 Abtastpunkte.
        """
        is_ = list(range(step_px // 2, width, step_px))
        js_ = list(range(step_px // 2, height, step_px))
        return [(i, j) for i in is_ for j in js_]


    def getfeatureinfo_geojson(i: int, j: int):
        """
        F?hrt getfeatureinfo_geojson aus.
        
        Args:
            i: Beschreibung.
            j: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        params = COMMON_PARAMS.copy()
        params.update({
            "REQUEST": "GetFeatureInfo",
            "BBOX": ",".join(map(str, bbox_25833)),
            "WIDTH": width,
            "HEIGHT": height,
            "I": i,
            "J": j,
            "FEATURE_COUNT": 1000,
            "feature_count": 1000,
        })
        url = f"{BASE_URL}->{urlencode(params)}"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if not isinstance(data, dict) or "features" not in data:
            return None
        if not data.get("features"):
            return None
        return data

    print(f"[DIVIS-WMS] Verwende BBOX (EPSG:25833): {bbox_25833}")
    all_features: list[dict] = []
    for (i, j) in sample_pixels(step_px=64):
        gj = getfeatureinfo_geojson(i, j)
        if gj is None:
            continue
        all_features.extend(gj.get("features", []))

    if not all_features:
        print("[DIVIS-WMS] Keine Features aus WMS erhalten.")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

    # *** WICHTIGER PART: CRS = 31469 (Gauss-Krüger Zone 5) ***
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:31469")

    # Duplikate nach ID-Feld herausfiltern, falls vorhanden
    id_candidates = ["id", "OB_ID", "OBJEKTID", "LFDNR"]
    id_col = next((c for c in id_candidates if c in gdf.columns), None)
    if id_col:
        before = len(gdf)
        gdf = gdf.drop_duplicates(subset=[id_col])
        after = len(gdf)
        print(f"[DIVIS-WMS] Drop duplicates nach {id_col}: {before} -> {after}")

    # Jetzt explizit nach target_epsg (meist 25833) transformieren
    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)

    tb = gdf.total_bounds
    print(f"[DIVIS-WMS] Ergebnis-BBOX in EPSG:{target_epsg}: {tb}")
    print(f"[DIVIS-WMS] Anzahl Features: {len(gdf)}")

    # --- Vor dem Mapping speichern (für Debug) ---
    out_divis_dir.mkdir(parents=True, exist_ok=True)
    full_gpkg = out_divis_dir / "divis_wms_raw.gpkg"
    head_csv = out_divis_dir / "divis_wms_raw_head.csv"

    try:
        gdf.to_file(full_gpkg, layer="divis_wms", driver="GPKG")
        print(f"[DIVIS-WMS] Roh-GPKG nach {full_gpkg} geschrieben.")
    except Exception as exc:
        print(f"[DIVIS-WMS] WARN: Konnte Roh-GPKG {full_gpkg} nicht schreiben: {exc}")

    try:
        gdf.drop(columns=["geometry"], errors="ignore").head(200).to_csv(
            head_csv, index=False, encoding="utf-8-sig"
        )
        print(f"[DIVIS-WMS] Head-CSV nach {head_csv} geschrieben.")
    except Exception as exc:
        print(f"[DIVIS-WMS] WARN: Konnte Head-CSV {head_csv} nicht schreiben: {exc}")

    return gdf

def _load_divis_for_area(ctx: PipelineContext, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Lädt DIVIS-Daten für das Gebiet der Gebäude.

    Reihenfolge:
    1) Webservice (WMS, L1/L2/L3) → bevorzugte Quelle
    2) Lokale Datei, falls WMS leer oder fehlgeschlagen:
       - settings.data.divis_path
       - out/ap1/divis_raw.gpkg
       - data/divis/divis_sn.gpkg

    Vor dem räumlichen Mapping werden die geladenen DIVIS-Features
    in 'out/divis' als GPKG/CSV gespeichert (für Debug/QGIS).
    """
    settings = _get_settings(ctx)
    data_cfg = _get_data_cfg(settings)

    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    target_epsg = int(getattr(settings, "target_epsg", getattr(ctx, "target_epsg", 25833)))
    divis_out_dir = out_base / "divis"

    if buildings.empty:
        print("[DIVIS] Gebäude-Layer ist leer – keine DIVIS-Abfrage.")
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

    # BBOX aus Gebäudegeometrien
    minx, miny, maxx, maxy = buildings.total_bounds
    bbox_25833 = (float(minx), float(miny), float(maxx), float(maxy))

    # 1) WMS-Standard
    divis_gdf = _fetch_divis_wms_for_area(bbox_25833, target_epsg, divis_out_dir)
    if divis_gdf is not None and not divis_gdf.empty:
        return divis_gdf

    print("[DIVIS] WMS lieferte keine Daten – wechsle auf lokale DIVIS-Datei (Fallback).")

    # 2) Lokale Dateien (Fallback-Reihenfolge)
    # a) settings.data.divis_path
    local_candidates = []

    divis_path_cfg = getattr(data_cfg, "divis_path", None)
    if divis_path_cfg:
        local_candidates.append(Path(divis_path_cfg))

    # b) Cache unter out/ap1/divis_raw.gpkg
    ap1_out_dir = out_base / "ap1"
    local_candidates.append(ap1_out_dir / "divis_raw.gpkg")

    # c) Standardpfad unter data/divis
    local_candidates.append(Path("data/divis/divis_sn.gpkg"))

    for cand in local_candidates:
        if cand and cand.exists():
            print(f"[DIVIS] Lade lokale DIVIS-Datei: {cand}")
            try:
                gdf = gpd.read_file(cand)
                if gdf.crs is None:
                    gdf.set_crs(epsg=target_epsg, inplace=True)
                elif gdf.crs.to_epsg() != target_epsg:
                    gdf = gdf.to_crs(epsg=target_epsg)

                # vor Mapping speichern
                divis_out_dir.mkdir(parents=True, exist_ok=True)
                local_gpkg = divis_out_dir / "divis_local_raw.gpkg"
                try:
                    gdf.to_file(local_gpkg, layer="divis_local", driver="GPKG")
                    print(f"[DIVIS] Lokale DIVIS-Rohdaten nach {local_gpkg} geschrieben.")
                except Exception as exc:
                    print(f"[DIVIS] WARN: Konnte {local_gpkg} nicht schreiben: {exc}")

                return gdf
            except Exception as exc:
                print(f"[DIVIS] WARN: Konnte {cand} nicht lesen: {exc}")

    print("[DIVIS] Keine lokale DIVIS-Datei gefunden – gebe leeres GeoDataFrame zurück.")
    return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

def _extract_year_from_datierung(text):
    """
    Extrahiert ein Baujahr aus einer Datierungsbeschreibung.

    Unterstützt:
    1) vierstellige Jahreszahlen   → '1875', 'ca. 1920-1930'
    2) Jahrhundertangaben          → '19. Jh.', '19. Jahrhundert' → 1900
    3) 'er Jahre'-Angaben          → '1960er Jahre', '1840er', '1970er' → 1960
    """
    if not isinstance(text, str):
        return pd.NA

    # ----------------------------------------------
    # 1) Vierstellige Jahreszahl
    # ----------------------------------------------
    m_year = re.search(r"\b(\d{4})\b", text)
    if m_year:
        year = int(m_year.group(1))
        if 1200 <= year <= 2100:
            return year

    # ----------------------------------------------
    # 2) 'er Jahre'-Angaben wie '1960er Jahre', '1840er'
    #    RegEx fängt 3- oder 4-stellige Jahrzehntangaben ab
    # ----------------------------------------------
    m_decade = re.search(
        r"\b(\d{3,4})\s*er(->:\s+Jahre)->\b",
        text,
        flags=re.IGNORECASE,
    )
    if m_decade:
        decade = int(m_decade.group(1))
        # Jahrzehnt prüfen
        if 1200 <= decade <= 2100:
            return decade

    # ----------------------------------------------
    # 3) Jahrhundertangaben: '19. Jh.', '19. Jahrhundert'
    # ----------------------------------------------
    m_century = re.search(
        r"(\d{1,2})\s*\.->\s*(->:Jh\.->|Jahrhundert)",
        text,
        flags=re.IGNORECASE,
    )
    if m_century:
        century = int(m_century.group(1))
        year = century * 100
        if 1200 <= year <= 2100:
            return year

    return pd.NA


def _enrich_with_divis(ctx: PipelineContext, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ergänzt den Gebäudelayer um DIVIS-Informationen.

    NEUE VARIANTE (Option B):
    --------------------------------
    Statt einen grob gesampelten DIVIS-Layer via WMS-BBOX zu laden und dann
    einen sjoin zu machen, werden jetzt für JEDES Gebäude einzeln
    GetFeatureInfo-Anfragen abgesetzt (gebäudegenauer "Klick" wie in QGIS).

    Vorgehen:
    - Schwerpunkt (representative_point) des Gebäudes bestimmen (EPSG:25833).
    - Kleine BBOX um diesen Punkt legen (z. B. +- 10 m).
    - GetFeatureInfo mit INFO_FORMAT=application/geo+json aufrufen.
    - Erstes sinnvolles Denkmal-Feature auswerten (DENKMALART / DENKMALSTATUS / OBJEKTART).
    - Attribute direkt ins Gebäude schreiben:

        DIVIS_flag
        DIVIS_Denkmalstatus
        DIVIS_Baujahr
        DIVIS_ext_kurzcharakteristik
        DIVIS_ext_datierung
        DIVIS_Baujahr_Extrakt

    Zusätzlich wird das angereicherte Ergebnis wie bisher in
    out/divis/divis_buildings_enriched.gpkg und .csv abgelegt.
    """
    settings = _get_settings(ctx)
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    divis_out_dir = out_base / "divis"
    divis_out_dir.mkdir(parents=True, exist_ok=True)

    if buildings.empty:
        print("[DIVIS] Gebäude-Layer ist leer – setze leere DIVIS-Spalten.")
        for col in [
            "DIVIS_flag",
            "DIVIS_Denkmalstatus",
            "DIVIS_Baujahr",
            "DIVIS_ext_kurzcharakteristik",
            "DIVIS_ext_datierung",
            "DIVIS_Baujahr_Extrakt",
        ]:
            buildings[col] = pd.NA if col != "DIVIS_flag" else False
        return buildings

    # ------------------------------------------------------------------
    # WMS-Setup (pro Gebäude ein GetFeatureInfo)
    # ------------------------------------------------------------------
    BASE_URL = "https://cardomap3.idu.de/lfds/public/ogc.ashx"
    COMMON_PARAMS = {
        "PKGID": 5,
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        # L1 = Flächen, L2 = Linien, L3 = Punkte
        "LAYERS": "L1,L2,L3",
        "QUERY_LAYERS": "L1,L2,L3",
        "STYLES": "",
        "CRS": "EPSG:25833",  # wir arbeiten im Gebäude-CRS
        "INFO_FORMAT": "application/geo+json",
    }

    # HTTP-Session mit Retries (analog DIVIS-WMS-Source)
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.mount("https://", HTTPAdapter(max_retries=retry))

    # Bildgröße und Pixelposition (Bildmitte)
    width = height = 101
    center_i = center_j = width // 2

    # BBOX-Halbausdehnung um den Gebäude-Schwerpunkt (in m)
    # -> relativ klein, damit wirklich das Gebäude "angeklickt" wird
    dx = dy = 10.0

    # ------------------------------------------------------------------
    # Ergebnis-Container initialisieren
    # ------------------------------------------------------------------
    idx = buildings.index
    divis_flag = pd.Series(False, index=idx, dtype="boolean")
    divis_denkmalstatus = pd.Series(pd.NA, index=idx, dtype="object")
    divis_baujahr = pd.Series(pd.NA, index=idx, dtype="object")
    divis_ext_kurz = pd.Series(pd.NA, index=idx, dtype="object")
    divis_ext_datierung = pd.Series(pd.NA, index=idx, dtype="object")
    divis_baujahr_extrakt = pd.Series(pd.NA, index=idx, dtype="Int64")


    # Optional: für QA eine Liste der "Roh-Treffer" mit building_id
    hits_records: list[dict] = []

    # Hilfsfunktion: bestes Denkmal-Feature aus einer Feature-Liste finden
    def _choose_best_feature(features: list[dict]) -> dict | None:
        if not features:
            return None
        # bevorzugt ein Feature mit DENKMALART / DENKMALSTATUS / OBJEKTART
        for f in features:
            props = f.get("properties") or {}
            for key in ("DENKMALART", "DENKMALSTATUS", "OBJEKTART"):
                val = props.get(key)
                if val not in (None, "", " "):
                    return f
        # sonst: erstes Feature nehmen
        return features[0]

    # ------------------------------------------------------------------
    # Hauptschleife: pro Gebäude ein GetFeatureInfo
    # ------------------------------------------------------------------
    print(f"[DIVIS] Starte gebäudegenaues WMS-GetFeatureInfo für {len(buildings)} Gebäude ...")

    # Sicherstellen, dass Gebäude im erwarteten CRS liegen
    # (wir gehen davon aus, dass AP1 bereits nach target_epsg gebracht hat)
    # -> falls nicht, hier konvertieren
    # target_epsg = int(getattr(settings, "target_epsg", getattr(ctx, "target_epsg", 25833)))
    # if buildings.crs is not None and buildings.crs.to_epsg() != target_epsg:
    #     buildings = buildings.to_crs(epsg=target_epsg)

    for idx_row, geom in buildings.geometry.items():
        if geom is None or geom.is_empty:
            continue

        try:
            pt = geom.representative_point()
        except Exception:
            continue

        x, y = pt.x, pt.y
        bbox = (x - dx, y - dy, x + dx, y + dy)

        params = COMMON_PARAMS.copy()
        params.update({
            "REQUEST": "GetFeatureInfo",
            "BBOX": ",".join(map(str, bbox)),
            "WIDTH": width,
            "HEIGHT": height,
            "I": center_i,
            "J": center_j,
            "FEATURE_COUNT": 10,
            "feature_count": 10,
        })
        url = f"{BASE_URL}->{urlencode(params)}"

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        features = (data.get("features") or []) if isinstance(data, dict) else []
        if not features:
            continue

        best = _choose_best_feature(features)
        if best is None:
            continue

        props = best.get("properties") or {}
        # Keys robust in Großschreibung behandeln
        props_up = {str(k).upper(): v for k, v in props.items()}

        # Flag
        divis_flag[idx_row] = True

        # Denkmalstatus
        status = (
            props_up.get("DENKMALART")
            or props_up.get("DENKMALSTATUS")
            or props_up.get("OBJEKTART")
        )
        if status in (None, ""):
            status = pd.NA
        divis_denkmalstatus[idx_row] = status

        # Baujahr-Feld direkt aus DIVIS (falls vorhanden)
        bj = props_up.get("BAUJAHR") or props_up.get("BJ_VON") or props_up.get("BAUJAHR_VON")
        divis_baujahr[idx_row] = bj if bj not in (None, "") else pd.NA

        # Kurzcharakteristik & Datierung
        ext_kurz = props.get("ext_kurzcharakteristik") or props.get("EXT_KURZCHARAKTERISTIK")
        ext_dat = props.get("ext_datierung") or props.get("EXT_DATIERUNG")

        divis_ext_kurz[idx_row] = ext_kurz if ext_kurz not in (None, "") else pd.NA
        divis_ext_datierung[idx_row] = ext_dat if ext_dat not in (None, "") else pd.NA

        # Baujahr-Extrakt aus Datierungstext
        year_ex = _extract_year_from_datierung(ext_dat) if isinstance(ext_dat, str) else pd.NA
        if pd.isna(year_ex) and isinstance(bj, (str, int)):
            # Fallback: evtl. vierstellige Zahl aus BAUJAHR-Feld ziehen
            year_ex = _extract_year_from_datierung(str(bj))
        divis_baujahr_extrakt[idx_row] = year_ex

        # QA-Record
        hit_rec = {
            "building_index": idx_row,
            "building_id": buildings.loc[idx_row].get("building_id", None),
            "x": x,
            "y": y,
            "DIVIS_Denkmalstatus": status,
            "DIVIS_Baujahr": bj,
            "DIVIS_ext_kurzcharakteristik": ext_kurz,
            "DIVIS_ext_datierung": ext_dat,
            "DIVIS_Baujahr_Extrakt": year_ex,
        }
        hits_records.append(hit_rec)

    # ------------------------------------------------------------------
    # Ergebnisse in Gebäude-DataFrame schreiben
    # ------------------------------------------------------------------
    buildings["DIVIS_flag"] = divis_flag
    buildings["DIVIS_Denkmalstatus"] = divis_denkmalstatus
    buildings["DIVIS_Baujahr"] = divis_baujahr
    buildings["DIVIS_ext_kurzcharakteristik"] = divis_ext_kurz
    buildings["DIVIS_ext_datierung"] = divis_ext_datierung
    buildings["DIVIS_Baujahr_Extrakt"] = divis_baujahr_extrakt

    print(
        f"[DIVIS] Gebäude mit DIVIS-Treffer: "
        f"{int(divis_flag.sum())} von {len(buildings)}"
    )

    # ------------------------------------------------------------------
    # QA: Gebäude+DIVIS als GPKG/CSV ablegen (wie bisher)
    # ------------------------------------------------------------------
    buildings_divis_gpkg = divis_out_dir / "divis_buildings_enriched.gpkg"
    buildings_divis_csv = divis_out_dir / "divis_buildings_enriched.csv"
    try:
        buildings.to_file(buildings_divis_gpkg, layer="buildings_divis", driver="GPKG")
        print(f"[DIVIS] Gebäude+DIVIS als GPKG nach {buildings_divis_gpkg} geschrieben.")
    except Exception as exc:
        print(f"[DIVIS] WARN: Konnte {buildings_divis_gpkg} nicht schreiben: {exc}")

    try:
        buildings.drop(columns="geometry", errors="ignore").to_csv(
            buildings_divis_csv, index=False, encoding="utf-8-sig"
        )
        print(f"[DIVIS] Gebäude+DIVIS als CSV nach {buildings_divis_csv} geschrieben.")
    except Exception as exc:
        print(f"[DIVIS] WARN: Konnte {buildings_divis_csv} nicht schreiben: {exc}")

    # Optional: zusätzliche QA-Datei nur mit den Treffern
    if hits_records:
        hits_df = pd.DataFrame.from_records(hits_records)
        hits_csv = divis_out_dir / "divis_building_hits.csv"
        try:
            hits_df.to_csv(hits_csv, index=False, encoding="utf-8-sig")
            print(f"[DIVIS] QA-Tabelle der DIVIS-Treffer nach {hits_csv} geschrieben.")
        except Exception as exc:
            print(f"[DIVIS] WARN: Konnte {hits_csv} nicht schreiben: {exc}")

    return buildings

def _load_or_create_zensus_dataset(ctx: PipelineContext) -> gpd.GeoDataFrame:
    """
    Stellt sicher, dass ein Zensus-100m-Datensatz existiert.
    - Wenn bereits erzeugt → laden
    - Wenn nicht → via ZensusGridSource erzeugen

    Standard-Ablage:
    - <workdir>/data/Zensus2022/zensus_100m.gpkg

    Optional kann in settings.data.zensus_path ein expliziter Pfad gesetzt werden.
    """

    settings = _get_settings(ctx)
    data_cfg = _get_data_cfg(settings)

    from kwp_bedarfskennwerte.config.paths import rel_zensus2022_dir
    base = Path(getattr(ctx, "workdir", ".")) if getattr(ctx, "workdir", None) is not None else Path(".")
    zensus_out_dir = base / rel_zensus2022_dir()
    zensus_out_dir.mkdir(parents=True, exist_ok=True)

    zensus_path_cfg = getattr(data_cfg, "zensus_path", None)
    if zensus_path_cfg:
        zensus_path = Path(zensus_path_cfg)
    else:
        zensus_path = zensus_out_dir / "zensus_100m.gpkg"

    # Falls die Datei existiert → direkt laden
    if zensus_path.exists():
        return gpd.read_file(zensus_path)

    # Falls nicht: ZensusGridSource verwenden
    from kwp_bedarfskennwerte.data_catalog.sources import ZensusGridSource

    zensus_source = ZensusGridSource()
    zensus_gdf = zensus_source.load(ctx)

    # Abspeichern im (ggf. neuen) Standardpfad
    try:
        zensus_path.parent.mkdir(parents=True, exist_ok=True)
        zensus_gdf.to_file(zensus_path, driver="GPKG")
        print(f"[ZENSUS] 100m-Grid nach {zensus_path} geschrieben.")
    except Exception as exc:
        print(f"[ZENSUS] WARN: Konnte Zensus-Grid nicht nach {zensus_path} schreiben: {exc}")

    return zensus_gdf

def _year_to_zensus_class(year) -> pd.Series:
    """
    Mappt ein numerisches Baujahr auf die Zensus-Baujahresklassen:
      - vor_1919
      - 1919-1948
      - 1949-1978
      - 1979-1990
      - 1991-2000
      - 2001-2010
      - 2011-2019
      - 2020+
    """
    if pd.isna(year):
        return pd.NA
    try:
        y = int(year)
    except Exception:
        return pd.NA

    if y < 1919:
        return "vor_1919"
    if y <= 1948:
        return "1919-1948"
    if y <= 1978:
        return "1949-1978"
    if y <= 1990:
        return "1979-1990"
    if y <= 2000:
        return "1991-2000"
    if y <= 2010:
        return "2001-2010"
    if y <= 2019:
        return "2011-2019"
    return "2020+"


def _zensus_class_to_midyear(label) -> pd.Series:
    """
    Approximiert ein 'Stichjahr' zu einer Zensus-Baujahresklasse.
    Dient nur als Hilfswert, z. B. für spätere Kennwertableitungen.
    """
    mapping = {
        "vor_1919": 1910,
        "1919-1948": 1935,
        "1949-1978": 1960,
        "1979-1990": 1985,
        "1991-2000": 1995,
        "2001-2010": 2005,
        "2011-2019": 2015,
        "2020+": 2022,
    }
    if pd.isna(label):
        return pd.NA
    return mapping.get(label, pd.NA)


def _infer_residential_flag(df: pd.DataFrame) -> pd.Series:
    """
    Leitet ein Wohngebäude-Flag pro Gebäude ab.

    Priorität:
    1) OBAT_is_residential (falls vorhanden)
    2) Final_Nutzung_vereinheitlicht (Text enthält 'wohn')
    3) LOD/BMAP/OSM_NWGoderWG == 'WG'
    """
    is_res = pd.Series(False, index=df.index, dtype="boolean")

    # 1) OBAT-Flag
    if "OBAT_is_residential" in df.columns:
        is_res = is_res | df["OBAT_is_residential"].fillna(False)

    # 2) Finale Nutzung als Text
    if "Final_Nutzung_vereinheitlicht" in df.columns:
        mask = df["Final_Nutzung_vereinheitlicht"].astype(str).str.contains(
            "wohn", case=False, na=False
        )
        is_res = is_res | mask

    # 3) WG/NWG-Flags
    for col in ("LOD_NWGoderWG", "BMAP_NWGoderWG", "OSM_NWGoderWG"):
        if col in df.columns:
            mask = df[col].astype(str).str.upper().eq("WG")
            is_res = is_res | mask

    return is_res

def _estimate_storeys_and_floor_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schätzt Geschosszahl und Nutzfläche pro Gebäude.

    - bevorzugt vorhandene LOD_Stockwerke
    - sonst Gebäudehöhe / "beste" Geschosshöhe:
        -> wir suchen n in [1..20], so dass Höhe/n in [2.0, 4.0] liegt
           und möglichst nah an 3.0 m ist.
        -> wenn nichts passt, fallback: round(H/3.0)
    - Grundfläche:
        -> LOD_Grundflaeche_m2 > BMAP_area_m2 > OSM_area_m2 > geometry.area
    - Nutzfläche = Grundfläche * Geschosszahl

    Ergebnis:
      - Final_Stockwerke_schaetzung
      - Final_Nutzflaeche_m2
    """
    # WICHTIG: Index eindeutig machen, damit combine_first/reindex nicht kollidiert
    df = df.copy().reset_index(drop=True)
    idx = df.index

    # Grundfläche bestimmen
    footprint = pd.Series(np.nan, index=idx, dtype="float64")
    for col in ("LOD_Grundflaeche_m2", "BMAP_area_m2", "OSM_area_m2"):
        if col in df.columns:
            footprint = footprint.combine_first(pd.to_numeric(df[col], errors="coerce"))

    # Fallback: Geometriefläche (falls GeoDataFrame mit Polygonen)
    if "geometry" in df.columns:
        try:
            geom_area = df.geometry.area.astype("float64")
            footprint = footprint.combine_first(geom_area)
        except Exception:
            pass

    # Gebäudehöhe bestimmen
    height = pd.Series(np.nan, index=idx, dtype="float64")
    for col in ("LOD_GebHoehe", "BMAP_height_m", "OSM_height_m"):
        if col in df.columns:
            height = height.combine_first(pd.to_numeric(df[col], errors="coerce"))

    # vorhandene Geschosszahlen bevorzugen auskommentiert, da oft 0 als Flag eingetragen.
    storeys = pd.Series(np.nan, index=idx, dtype="float64")
    #if "LOD_Stockwerke" in df.columns:
    #   storeys = storeys.combine_first(pd.to_numeric(df["LOD_Stockwerke"], errors="coerce"))

    # Berechnung der Geschosszahl aus Höhe (falls nötig)
    def best_storey_count(h: float) -> int:
        """
        F?hrt best_storey_count aus.
        
        Args:
            h: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        if not np.isfinite(h) or h <= 0:
            return 1
        best_n = None
        best_diff = 1e9
        for n in range(1, 21):
            storey_h = h / n
            if storey_h < 2.0 or storey_h > 4.0:
                continue
            diff = abs(storey_h - 3.0)
            if diff < best_diff:
                best_diff = diff
                best_n = n
        if best_n is not None:
            return best_n
        # Fallback, wenn nichts im 2–4m-Bereich passt
        return max(1, int(round(h / 3.0)))

    missing_storeys_mask = storeys.isna() & height.notna()
    storeys.loc[missing_storeys_mask] = height.loc[missing_storeys_mask].apply(best_storey_count)

    # Fallback: immer mind. 1 Geschoss
    storeys = storeys.fillna(1.0)
    storeys = storeys.clip(lower=1)

    # Nutzfläche = Grundfläche * Geschosszahl
    floor_area = footprint * storeys

    df["Final_Stockwerke_schaetzung"] = storeys.astype("float64")
    df["Final_Nutzflaeche_m2"] = floor_area.astype("float64")

    return df


def _assign_final_heating_from_zensus(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Weist für jedes Gebäude einen wahrscheinlichen Energieträger und eine
    Heizungsart zu, basierend auf:

      - Zensus-Anteilen in der 100m-Zelle
      - Baulichem Alter (Final_Baujahr_Mitte / -klasse)
      - Nutzfläche (Final_Nutzflaeche_m2)

    Heuristik:
      - ältere Gebäude → eher fossile Brennstoffe (Gas, Öl, Kohle, Holz)
      - neuere Gebäude → eher Wärmepumpe / erneuerbare (Solar/Geothermie/WP, Strom)
      - große Gebäude → eher Fernwärme + Zentral-/Blockheizung
      - kleine Gebäude → eher Etagenheizung / Einzelöfen
    """
    df = gdf.copy()

    # Sicherstellen, dass die Zensus-Anteile und Zell-IDs vorhanden sind
    carrier_cols = [
        ("Gas", "Gas"),
        ("Heizoel", "Heizöl"),
        ("Holz_Holzpellets", "Holz/Holzpellets"),
        ("Biomasse_Biogas", "Biomasse/Biogas"),
        ("Solar_Geothermie_Waermepumpen", "Solar/Geothermie/Wärmepumpen"),
        ("Strom", "Strom"),
        ("Kohle", "Kohle"),
        ("Fernwaerme", "Fernwärme"),
        ("kein_Energietraeger", "kein Energieträger"),
    ]
    heating_cols = [
        ("Fernheizung", "Fernheizung"),
        ("Etagenheizung", "Etagenheizung"),
        ("Blockheizung", "Blockheizung"),
        ("Zentralheizung", "Zentralheizung"),
        ("Einzel_Mehrraumoefen", "Einzel-/Mehrraumoefen"),
        ("keine_Heizung", "keine Heizung"),
    ]

    if "GITTER_ID_100m" not in df.columns:
        # Keine Zellinformation → Fallback: nur dominante Zensus-Klasse übernehmen
        if "ZENSUS_Energietraeger_Heizung" in df.columns:
            df["Final_Energietraeger"] = df["ZENSUS_Energietraeger_Heizung"]
        else:
            df["Final_Energietraeger"] = pd.NA
        if "ZENSUS_Heizungsart" in df.columns:
            df["Final_Heizungsart"] = df["ZENSUS_Heizungsart"]
        else:
            df["Final_Heizungsart"] = pd.NA
        df["Final_Heizsystem_quelle"] = "Zensus_Mode"
        return df

    # Zielspalten initialisieren
    for col in ("Final_Energietraeger", "Final_Heizungsart", "Final_Heizsystem_quelle"):
        if col not in df.columns:
            df[col] = pd.NA

    # Nutzfläche schätzen (falls noch nicht vorhanden)
    if "Final_Nutzflaeche_m2" not in df.columns or df["Final_Nutzflaeche_m2"].isna().all():
        df = _estimate_storeys_and_floor_area(df)

    floor_area = pd.to_numeric(df["Final_Nutzflaeche_m2"], errors="coerce").fillna(0.0)

    # Größenkategorien
    #   klein  : < 150 m2
    #   mittel : 150–500 m2
    #   groß   : 500–1500 m2
    #   sehr groß: >1500 m2 (z. B. größere Blöcke, MFH, Nichtwohngebäude)
    size_small = floor_area < 150.0
    size_medium = (floor_area >= 150.0) & (floor_area <= 500.0)
    size_large = (floor_area > 500.0) & (floor_area <= 1500.0)
    size_very_large = floor_area > 1500.0

    # Baualter (Stichjahr) bestimmen
    if "Final_Baujahr_Mitte" in df.columns:
        bj_mid = pd.to_numeric(df["Final_Baujahr_Mitte"], errors="coerce")
    else:
        bj_mid = pd.Series(np.nan, index=df.index, dtype="float64")
        if "Final_Baujahrklasse" in df.columns:
            bj_mid = df["Final_Baujahrklasse"].map(_zensus_class_to_midyear)

    rng = np.random.default_rng(2025)

    # Gruppierung nach Zensuszelle
    groups = df.groupby("GITTER_ID_100m").groups

    for grid_id, idxs in groups.items():
        idxs = list(idxs)
        if not idxs:
            continue

        group = df.loc[idxs]
        row0 = group.iloc[0]

        # Basis-Anteile Energieträger
        base_carrier = np.array(
            [float(row0.get(col, 0.0) or 0.0) for col, _ in carrier_cols],
            dtype=float,
        )
        if not np.isfinite(base_carrier).any() or base_carrier.sum() <= 0:
            # Fallback: nur dominante Klasse verwenden
            if "ZENSUS_Energietraeger_Heizung" in df.columns:
                df.loc[idxs, "Final_Energietraeger"] = df.loc[
                    idxs, "ZENSUS_Energietraeger_Heizung"
                ]
                df.loc[idxs, "Final_Heizsystem_quelle"] = "Zensus_Mode"
            continue

        # Basis-Anteile Heizungsart
        base_heating = np.array(
            [float(row0.get(col, 0.0) or 0.0) for col, _ in heating_cols],
            dtype=float,
        )
        if not np.isfinite(base_heating).any() or base_heating.sum() <= 0:
            base_heating = np.ones(len(heating_cols), dtype=float)

        for idx in idxs:
            age = bj_mid.loc[idx]
            fa = floor_area.loc[idx]

            # -------------------------------
            # Energieträger-Gewichte ableiten
            # -------------------------------
            weights_c = base_carrier.copy()

            # Altersheuristik:
            if np.isfinite(age):
                if age < 1949:
                    # sehr alt → stark fossil geprägt, auch Kohle/Holz
                    for k, (col, _) in enumerate(carrier_cols):
                        if col in ("Gas", "Heizoel", "Kohle", "Holz_Holzpellets"):
                            weights_c[k] *= 1.8
                        if col in ("Solar_Geothermie_Waermepumpen", "Strom"):
                            weights_c[k] *= 0.4
                elif age < 1979:
                    # 1950–1978 → fossil dominiert, aber schon Gas/Fernwärme
                    for k, (col, _) in enumerate(carrier_cols):
                        if col in ("Gas", "Heizoel", "Fernwaerme"):
                            weights_c[k] *= 1.4
                        if col in ("Solar_Geothermie_Waermepumpen", "Strom"):
                            weights_c[k] *= 0.6
                elif age < 2001:
                    # 1979–2000 → deutlicher Gas/Fernwärme-Anteil, Öl nimmt ab
                    for k, (col, _) in enumerate(carrier_cols):
                        if col in ("Gas", "Fernwaerme"):
                            weights_c[k] *= 1.5
                        if col == "Heizoel":
                            weights_c[k] *= 0.7
                        if col in ("Solar_Geothermie_Waermepumpen", "Strom"):
                            weights_c[k] *= 0.8
                else:
                    # ab 2001 → deutlich mehr WP/erneuerbare
                    for k, (col, _) in enumerate(carrier_cols):
                        if col in ("Solar_Geothermie_Waermepumpen", "Strom"):
                            weights_c[k] *= 2.0
                        if col in ("Heizoel", "Kohle"):
                            weights_c[k] *= 0.3

            # Größenheuristik:
            if size_very_large.loc[idx] or size_large.loc[idx]:
                # große Gebäude → mehr Fernwärme, weniger Einzel-/kein ET
                for k, (col, _) in enumerate(carrier_cols):
                    if col == "Fernwaerme":
                        weights_c[k] *= 1.6
                    if col == "kein_Energietraeger":
                        weights_c[k] *= 0.3
            elif size_small.loc[idx]:
                # kleine Gebäude → stärker individuelle Heizungen
                for k, (col, _) in enumerate(carrier_cols):
                    if col in ("Gas", "Heizoel", "Holz_Holzpellets"):
                        weights_c[k] *= 1.3

            # Sicherstellen, dass die Summe > 0 ist
            weights_c = np.maximum(weights_c, 0.0)
            if weights_c.sum() <= 0:
                weights_c = np.ones_like(weights_c)

            probs_c = weights_c / weights_c.sum()
            k_sel = rng.choice(len(carrier_cols), p=probs_c)
            carrier_label = carrier_cols[k_sel][1]
            df.at[idx, "Final_Energietraeger"] = carrier_label

            # -------------------------------
            # Heizungsart-Gewichte ableiten
            # -------------------------------
            weights_h = base_heating.copy()

            # Konsistenz mit Energieträger:
            if carrier_label == "Fernwärme":
                # Fernwärme → Fernheizung / Block / Zentral
                for k, (_, lbl) in enumerate(heating_cols):
                    if lbl in ("Fernheizung", "Blockheizung", "Zentralheizung"):
                        weights_h[k] *= 1.8
                    if lbl in ("Einzel-/Mehrraumoefen", "keine Heizung"):
                        weights_h[k] *= 0.4
            elif carrier_label in ("Holz/Holzpellets", "Biomasse/Biogas", "Kohle"):
                # Feste Brennstoffe → eher Einzel-/Mehrraumoefen oder kleinere Systeme
                for k, (_, lbl) in enumerate(heating_cols):
                    if lbl in ("Einzel-/Mehrraumoefen", "Etagenheizung"):
                        weights_h[k] *= 1.6
            elif carrier_label in ("Solar/Geothermie/Wärmepumpen", "Strom"):
                # WP/Strom → oft Zentral- oder Etagenheizung
                for k, (_, lbl) in enumerate(heating_cols):
                    if lbl in ("Zentralheizung", "Etagenheizung"):
                        weights_h[k] *= 1.5

            # Größenheuristik:
            if size_very_large.loc[idx] or size_large.loc[idx]:
                for k, (_, lbl) in enumerate(heating_cols):
                    if lbl in ("Zentralheizung", "Fernheizung", "Blockheizung"):
                        weights_h[k] *= 1.7
                    if lbl in ("Etagenheizung", "Einzel-/Mehrraumoefen"):
                        weights_h[k] *= 0.5
            elif size_small.loc[idx]:
                for k, (_, lbl) in enumerate(heating_cols):
                    if lbl in ("Etagenheizung", "Einzel-/Mehrraumoefen"):
                        weights_h[k] *= 1.5

            weights_h = np.maximum(weights_h, 0.0)
            if weights_h.sum() <= 0:
                weights_h = np.ones_like(weights_h)

            probs_h = weights_h / weights_h.sum()
            h_sel = rng.choice(len(heating_cols), p=probs_h)
            heating_label = heating_cols[h_sel][1]

            df.at[idx, "Final_Heizungsart"] = heating_label
            df.at[idx, "Final_Heizsystem_quelle"] = "Zensus_Heuristik"

    # Globaler Fallback, falls irgendwo noch NA ist
    if "ZENSUS_Energietraeger_Heizung" in df.columns:
        mask = df["Final_Energietraeger"].isna() & df["ZENSUS_Energietraeger_Heizung"].notna()
        df.loc[mask, "Final_Energietraeger"] = df.loc[mask, "ZENSUS_Energietraeger_Heizung"]
        df.loc[mask, "Final_Heizsystem_quelle"] = "Zensus_Mode"

    if "ZENSUS_Heizungsart" in df.columns:
        mask = df["Final_Heizungsart"].isna() & df["ZENSUS_Heizungsart"].notna()
        df.loc[mask, "Final_Heizungsart"] = df.loc[mask, "ZENSUS_Heizungsart"]
        df.loc[mask, "Final_Heizsystem_quelle"] = "Zensus_Mode"

    return df


def _assign_final_baujahr_from_sources_and_zensus(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Erzeugt finale Baujahresphasen pro Gebäude mit Priorität:

        1) DIVIS (DIVIS_Baujahr_Extrakt / DIVIS_Baujahr)
        2) OBAT  (OBAT_Baujahr_Mitte)
        3) Zensus-Verteilung in der 100m-Zelle (nur Wohngebäude; Zufallszuordnung)
        4) ZENSUS_Baujahr_Klasse (Mehrheitsklasse der Zelle) als Fallback,
           mindestens für Nichtwohngebäude.

    Neue Spalten:
        - Final_Baujahrklasse
        - Final_Baujahr_quelle   ('DIVIS', 'OBAT', 'Zensus_Random', 'Zensus_Mode')
        - Final_Baujahr_Mitte    (approx. Jahr, z. B. 1960)
    """
    df = gdf.copy()

    # --------------------------------------------------------------
    # 0) Zielspalten initialisieren
    # --------------------------------------------------------------
    for col in ("Final_Baujahrklasse", "Final_Baujahr_quelle", "Final_Baujahr_Mitte"):
        if col not in df.columns:
            df[col] = pd.NA

    # --------------------------------------------------------------
    # 1) Jahre aus DIVIS / OBAT bestimmen
    # --------------------------------------------------------------
    # DIVIS: bevorzugt DIVIS_Baujahr_Extrakt, sonst DIVIS_Baujahr (falls numerisch)
    if "DIVIS_Baujahr_Extrakt" in df.columns:
        year_divis_ex = pd.to_numeric(df["DIVIS_Baujahr_Extrakt"], errors="coerce")
    else:
        year_divis_ex = pd.Series(np.nan, index=df.index)

    if "DIVIS_Baujahr" in df.columns:
        year_divis_raw = pd.to_numeric(df["DIVIS_Baujahr"], errors="coerce")
    else:
        year_divis_raw = pd.Series(np.nan, index=df.index)

    year_divis = year_divis_ex.where(~year_divis_ex.isna(), year_divis_raw)

    # OBAT: mittleres Baujahr
    if "OBAT_Baujahr_Mitte" in df.columns:
        year_obat = pd.to_numeric(df["OBAT_Baujahr_Mitte"], errors="coerce")
    else:
        year_obat = pd.Series(np.nan, index=df.index)

    # 1a) DIVIS priorisiert
    mask_divis = year_divis.notna()
    df.loc[mask_divis, "Final_Baujahr_Mitte"] = year_divis.loc[mask_divis]
    df.loc[mask_divis, "Final_Baujahrklasse"] = year_divis.loc[mask_divis].map(
        _year_to_zensus_class
    )
    df.loc[mask_divis, "Final_Baujahr_quelle"] = "DIVIS"

    # 1b) OBAT nur dort, wo noch nichts gesetzt wurde
    mask_obat = df["Final_Baujahrklasse"].isna() & year_obat.notna()
    df.loc[mask_obat, "Final_Baujahr_Mitte"] = year_obat.loc[mask_obat]
    df.loc[mask_obat, "Final_Baujahrklasse"] = year_obat.loc[mask_obat].map(
        _year_to_zensus_class
    )
    df.loc[mask_obat, "Final_Baujahr_quelle"] = "OBAT"

    # --------------------------------------------------------------
    # 2) Zensus-Verteilung je 100m-Zelle für Wohngebäude nutzen
    # --------------------------------------------------------------
    age_cols = [
        ("Vor1919", "vor_1919"),
        ("a1919bis1948", "1919-1948"),
        ("a1949bis1978", "1949-1978"),
        ("a1979bis1990", "1979-1990"),
        ("a1991bis2000", "1991-2000"),
        ("a2001bis2010", "2001-2010"),
        ("a2011bis2019", "2011-2019"),
        ("a2020undspaeter", "2020+"),
    ]
    if not all(c in df.columns for c, _ in age_cols) or "GITTER_ID_100m" not in df.columns:
        # Keine Verteilungsinformation → später nur Fallback auf ZENSUS_Baujahr_Klasse
        if "ZENSUS_Baujahr_Klasse" in df.columns:
            mask = df["Final_Baujahrklasse"].isna() & df["ZENSUS_Baujahr_Klasse"].notna()
            df.loc[mask, "Final_Baujahrklasse"] = df.loc[mask, "ZENSUS_Baujahr_Klasse"]
            df.loc[mask, "Final_Baujahr_quelle"] = "Zensus_Mode"
            df.loc[mask, "Final_Baujahr_Mitte"] = df.loc[mask, "ZENSUS_Baujahr_Klasse"].map(
                _zensus_class_to_midyear
            )
        return df

    # Wohngebäude-Flag
    is_res = _infer_residential_flag(df)

    # Random-Generator (fixer Seed für reproduzierbare Ergebnisse)
    rng = np.random.default_rng(12345)

    # Mapping von Klassenlabel → Index im Alters-Array
    label_to_idx = {label: i for i, (_, label) in enumerate(age_cols)}

    # Spaltenpositionen für iloc-Schreibzugriffe
    col_idx_class = df.columns.get_loc("Final_Baujahrklasse")
    col_idx_src = df.columns.get_loc("Final_Baujahr_quelle")
    col_idx_mid = df.columns.get_loc("Final_Baujahr_Mitte")

    # Gruppierung nach Zensuszelle
    groups = df.groupby("GITTER_ID_100m").groups
    for grid_id, idxs in groups.items():
        idxs = list(idxs)
        if not idxs:
            continue

        group = df.loc[idxs]

        # Zensus-Verteilung aus der ersten Zeile der Zelle
        row0 = group.iloc[0]
        weights = np.array(
            [max(float(row0[col]), 0.0) for col, _ in age_cols],
            dtype=float,
        )

        if not np.isfinite(weights).any() or weights.sum() <= 0.0:
            # Keine verwertbare Verteilung
            continue

        # 2a) bekannte Baujahre (DIVIS/OBAT) aus der Verteilung "herausrechnen"
        mask_known = (
            group["Final_Baujahrklasse"].notna()
            & group["Final_Baujahr_quelle"].isin(["DIVIS", "OBAT"])
        )
        if mask_known.any():
            counts = group.loc[mask_known, "Final_Baujahrklasse"].value_counts()
            for cls_label, cnt in counts.items():
                j = label_to_idx.get(cls_label)
                if j is None:
                    continue
                weights[j] = max(0.0, weights[j] - float(cnt))

        # 2b) verbleibende Wohngebäude ohne Baujahr zufallsverteilt zuordnen
        #     gemäß der Restverteilung in der Zelle
        unknown_mask = (
            df["Final_Baujahrklasse"].isna()
            & is_res
            & df["GITTER_ID_100m"].eq(grid_id)
        )
        pos_unknown = np.nonzero(unknown_mask.to_numpy())[0]
        if len(pos_unknown) == 0:
            continue

        for pos in pos_unknown:
            if weights.sum() <= 0:
                break  # nichts mehr zum Verteilen

            w = np.maximum(weights, 0.0)
            if w.sum() <= 0:
                break
            probs = w / w.sum()
            k = rng.choice(len(age_cols), p=probs)

            label = age_cols[k][1]
            df.iloc[pos, col_idx_class] = label
            df.iloc[pos, col_idx_src] = "Zensus_Random"
            df.iloc[pos, col_idx_mid] = _zensus_class_to_midyear(label)

            # eine "Einheit" aus der Klasse abziehen
            if weights[k] > 0:
                weights[k] = max(0.0, weights[k] - 1.0)

    # --------------------------------------------------------------
    # 3) Fallback: übrige Gebäude (mind. Nichtwohngebäude) mit
    #    ZENSUS_Baujahr_Klasse befüllen
    # --------------------------------------------------------------
    if "ZENSUS_Baujahr_Klasse" in df.columns:
        mask_remaining = df["Final_Baujahrklasse"].isna() & df["ZENSUS_Baujahr_Klasse"].notna()
        df.loc[mask_remaining, "Final_Baujahrklasse"] = df.loc[
            mask_remaining, "ZENSUS_Baujahr_Klasse"
        ]
        df.loc[mask_remaining, "Final_Baujahr_quelle"] = "Zensus_Mode"
        df.loc[mask_remaining, "Final_Baujahr_Mitte"] = df.loc[
            mask_remaining, "ZENSUS_Baujahr_Klasse"
        ].map(_zensus_class_to_midyear)

    return df




def _enrich_with_zensus(
    ctx: PipelineContext,
    buildings: gpd.GeoDataFrame,
    zensus_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Ergänzt den Gebäudelayer um Basis-Informationen aus dem Zensus-100m-Raster.

    Aktueller Umfang (nur BASISDATEN, kein Baujahr-/Heizungs-Mapping):
    - ZENSUS_Durchschnittsalter
    - ZENSUS_Anteil_65plus
    - ZENSUS_Leerstandsquote
    - ZENSUS_Eigentumsquote
    - ZENSUS_Gebaeudetyp

    Wichtige Eigenschaften:
    - buildings enthält i.d.R. bereits die DIVIS-Spalten (wenn zuvor _enrich_with_divis
      ausgeführt wurde).
    - Alle vorhandenen DIVIS-Spalten bleiben unverändert erhalten.
    - Ergebnis wird im Unterordner out/zensus abgelegt:
        * out/zensus/ap1_buildings_enriched_zensus.gpkg
        * out/zensus/ap1_buildings_enriched_zensus.csv
    """

    settings = _get_settings(ctx)
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    zensus_out_dir = out_base / "zensus"
    zensus_out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = zensus_out_dir / "ap1_buildings_enriched_zensus.gpkg"
    csv_path = zensus_out_dir / "ap1_buildings_enriched_zensus.csv"

    # ------------------------------------------------------------------ #
    # Sonderfall: kein Zensus-Grid vorhanden → nur leere ZENSUS-Spalten
    # ------------------------------------------------------------------ #
    if zensus_gdf is None or zensus_gdf.empty:
        for col in [
            "ZENSUS_Gebaeudetyp",
            "ZENSUS_Anteil_65plus",
            "ZENSUS_Leerstandsquote",
            "ZENSUS_Eigentumsquote",
            "ZENSUS_Durchschnittsalter",
            # Platzhalter für spätere Schritte (Baujahr/Heizung),
            # die aktuell NOCH NICHT befüllt werden:
            "ZENSUS_Baujahr_Klasse",
            "ZENSUS_Energietraeger_Heizung",
            "ZENSUS_Heizungsart",
        ]:
            if col not in buildings.columns:
                buildings[col] = pd.NA

        # trotzdem als GPKG/CSV ablegen
        try:
            buildings.to_file(gpkg_path, layer="buildings_zensus", driver="GPKG")
            print(f"[ZENSUS] Gebäude+Zensus (leer) als GPKG nach {gpkg_path} geschrieben.")
        except Exception as exc:
            print(f"[ZENSUS] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")

        try:
            buildings.drop(columns="geometry", errors="ignore").to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
            print(f"[ZENSUS] Gebäude+Zensus (leer) als CSV nach {csv_path} geschrieben.")
        except Exception as exc:
            print(f"[ZENSUS] WARN: Konnte {csv_path} nicht schreiben: {exc}")

        return buildings

    # ------------------------------------------------------------------ #
    # Normalfall: Zensus-Grid vorhanden → spatial join
    # ------------------------------------------------------------------ #
    zensus = zensus_gdf
    if zensus.crs != buildings.crs:
        zensus = zensus.to_crs(buildings.crs)

    # Räumlicher Join: Gebäude -> Zensuszelle (100m)
    g_joined = gpd.sjoin(
        buildings,
        zensus,
        how="left",
        predicate="intersects",  # ggf. "centroid.within"
    )

    # ------------------------------------------------------------------ #
    # 1) Direkt übernommene Kennzahlen (Skalare pro Zelle)
    # ------------------------------------------------------------------ #
    direct_mapping = {
        # Bevölkerung / Demografie
        "ZENSUS_Einwohner": ["Einwohner"],
        "ZENSUS_Anteil_Auslaender": ["AnteilAuslaender"],
        "ZENSUS_Durchschnittsalter": [
            "Durchschnittsalter",
            "durchschnittsalter",
        ],
        "ZENSUS_Anteil_Unter18": ["AnteilUnter18"],
        "ZENSUS_Anteil_65plus": [
            "AnteilUeber65",
            "Anteil_65plus",
            "PERS_65PANTEIL",
        ],
        "ZENSUS_DurchschnHHGroesse": ["DurchschnHHGroesse"],
        "ZENSUS_Miete_Qm": ["durchschnMieteQM"],
        "ZENSUS_Flaeche_je_Wohnung": ["durchschnFlaechejeWohn"],
        "ZENSUS_Flaeche_je_Bewohner": ["durchschnFlaechejeBew"],

        # Wohnungsmarkt
        "ZENSUS_Eigentumsquote": [
            "Eigentuemerquote",
            "W_EIGQ_4000W_000",
            "W_EIGENT_QUOTE",
        ],
        "ZENSUS_Leerstandsquote": [
            "Leerstandsquote",
            "W_LEERSTQ_4000W_000",
            "W_LEERST_QUOTE",
        ],
        "ZENSUS_MALeerstQuote": ["MALeerstQuote"],

        # Summen/Anzahlen Gebäude / Heizsystem / Energieträger
        "ZENSUS_Anzahl_Gebaeude": ["Insgesamt_Gebaeude"],
        "ZENSUS_Anzahl_Heizungsart": ["Insgesamt_Heizungsart"],
        "ZENSUS_Anzahl_Energietraeger": ["Insgesamt_Energietraeger"],

        # Energieträger-Anteile (relativ je Zelle)
        "ZENSUS_Anteil_Gas": ["Gas"],
        "ZENSUS_Anteil_Heizoel": ["Heizoel"],
        "ZENSUS_Anteil_Holz_Holzpellets": ["Holz_Holzpellets"],
        "ZENSUS_Anteil_Biomasse_Biogas": ["Biomasse_Biogas"],
        "ZENSUS_Anteil_Solar_Geothermie_Waermepumpen": ["Solar_Geothermie_Waermepumpen"],
        "ZENSUS_Anteil_Strom": ["Strom"],
        "ZENSUS_Anteil_Kohle": ["Kohle"],
        "ZENSUS_Anteil_Fernwaerme": ["Fernwaerme"],
        "ZENSUS_Anteil_kein_Energietraeger": ["kein_Energietraeger"],

        # Heizungsarten-Anteile (relativ je Zelle)
        "ZENSUS_Anteil_Fernheizung": ["Fernheizung"],
        "ZENSUS_Anteil_Etagenheizung": ["Etagenheizung"],
        "ZENSUS_Anteil_Blockheizung": ["Blockheizung"],
        "ZENSUS_Anteil_Zentralheizung": ["Zentralheizung"],
        "ZENSUS_Anteil_Einzel_Mehrraumoefen": ["Einzel_Mehrraumoefen"],
        "ZENSUS_Anteil_keine_Heizung": ["keine_Heizung"],
    }

    for target_col, candidates in direct_mapping.items():
        col_found = None
        for col in candidates:
            if col in g_joined.columns:
                col_found = col
                break
        if col_found is not None:
            g_joined[target_col] = g_joined[col_found]
        else:
            if target_col not in g_joined.columns:
                g_joined[target_col] = pd.NA

    # ------------------------------------------------------------------ #
    # 2) Abgeleitete Kategorien (dominierende Klassen je 100m-Zelle)
    # ------------------------------------------------------------------ #
    # a) Dominierende Baualtersklasse
    age_cols = [
        ("Vor1919", "vor_1919"),
        ("a1919bis1948", "1919-1948"),
        ("a1949bis1978", "1949-1978"),
        ("a1979bis1990", "1979-1990"),
        ("a1991bis2000", "1991-2000"),
        ("a2001bis2010", "2001-2010"),
        ("a2011bis2019", "2011-2019"),
        ("a2020undspaeter", "2020+"),
    ]
    if all(col in g_joined.columns for col, _ in age_cols):
        age_df = g_joined[[col for col, _ in age_cols]].astype("float64")
        # idxmax liefert den Spaltennamen mit dem hoechsten Anteil je Zeile
        # Bei all-NA Zeilen vorher maskieren, um FutureWarning zu vermeiden
        age_mask = age_df.notna().any(axis=1)
        age_idxmax = pd.Series(pd.NA, index=age_df.index, dtype="object")
        if age_mask.any():
            age_idxmax.loc[age_mask] = age_df.loc[age_mask].idxmax(axis=1, skipna=True)
        age_label_map = {col: label for col, label in age_cols}
        g_joined["ZENSUS_Baujahr_Klasse"] = age_idxmax.map(age_label_map)
    else:
        if "ZENSUS_Baujahr_Klasse" not in g_joined.columns:
            g_joined["ZENSUS_Baujahr_Klasse"] = pd.NA

    # b) Dominierender Energieträger
    carrier_cols = [
        ("Gas", "Gas"),
        ("Heizoel", "Heizöl"),
        ("Holz_Holzpellets", "Holz/Holzpellets"),
        ("Biomasse_Biogas", "Biomasse/Biogas"),
        ("Solar_Geothermie_Waermepumpen", "Solar/Geothermie/Wärmepumpen"),
        ("Strom", "Strom"),
        ("Kohle", "Kohle"),
        ("Fernwaerme", "Fernwärme"),
        ("kein_Energietraeger", "kein Energieträger"),
    ]
    if all(col in g_joined.columns for col, _ in carrier_cols):
        carrier_df = g_joined[[col for col, _ in carrier_cols]].astype("float64")
        carrier_mask = carrier_df.notna().any(axis=1)
        carrier_idxmax = pd.Series(pd.NA, index=carrier_df.index, dtype="object")
        if carrier_mask.any():
            carrier_idxmax.loc[carrier_mask] = carrier_df.loc[carrier_mask].idxmax(axis=1, skipna=True)
        carrier_label_map = {col: label for col, label in carrier_cols}
        g_joined["ZENSUS_Energietraeger_Heizung"] = carrier_idxmax.map(carrier_label_map)
    else:
        if "ZENSUS_Energietraeger_Heizung" not in g_joined.columns:
            g_joined["ZENSUS_Energietraeger_Heizung"] = pd.NA

    # c) Dominierende Heizungsart
    heating_cols = [
        ("Fernheizung", "Fernheizung"),
        ("Etagenheizung", "Etagenheizung"),
        ("Blockheizung", "Blockheizung"),
        ("Zentralheizung", "Zentralheizung"),
        ("Einzel_Mehrraumoefen", "Einzel-/Mehrraumoefen"),
        ("keine_Heizung", "keine Heizung"),
    ]
    if all(col in g_joined.columns for col, _ in heating_cols):
        heating_df = g_joined[[col for col, _ in heating_cols]].astype("float64")
        heating_mask = heating_df.notna().any(axis=1)
        heating_idxmax = pd.Series(pd.NA, index=heating_df.index, dtype="object")
        if heating_mask.any():
            heating_idxmax.loc[heating_mask] = heating_df.loc[heating_mask].idxmax(axis=1, skipna=True)
        heating_label_map = {col: label for col, label in heating_cols}
        g_joined["ZENSUS_Heizungsart"] = heating_idxmax.map(heating_label_map)
    else:
        if "ZENSUS_Heizungsart" not in g_joined.columns:
            g_joined["ZENSUS_Heizungsart"] = pd.NA

    # ------------------------------------------------------------------ #
    # 2b) Finale Baujahrsphasen je Gebäude
    # ------------------------------------------------------------------ #
    g_joined = _assign_final_baujahr_from_sources_and_zensus(g_joined)

    # ------------------------------------------------------------------ #
    # 2c) Finale Heizungsart & Energieträger je Gebäude aus Zensus
    # ------------------------------------------------------------------ #
    g_joined = _assign_final_heating_from_zensus(g_joined)

    # ------------------------------------------------------------------ #
    # Aufräumen: Hilfsspalten aus dem spatial join entfernen
    # ------------------------------------------------------------------ #
    g_joined = g_joined.drop(
        columns=[c for c in g_joined.columns if c.startswith("index_")],
        errors="ignore",
    )




    # ------------------------------------------------------------------ #
    # Ergebnis als GPKG + CSV im ZENSUS-Unterordner ablegen
    # (inkl. aller vorhandenen DIVIS-Spalten)
    # ------------------------------------------------------------------ #
    try:
        g_joined.to_file(gpkg_path, layer="buildings_zensus", driver="GPKG")
        print(f"[ZENSUS] Gebäude+Zensus als GPKG nach {gpkg_path} geschrieben.")
    except Exception as exc:
        print(f"[ZENSUS] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")

    try:
        g_joined.drop(columns="geometry", errors="ignore").to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        print(f"[ZENSUS] Gebäude+Zensus als CSV nach {csv_path} geschrieben.")
    except Exception as exc:
        print(f"[ZENSUS] WARN: Konnte {csv_path} nicht schreiben: {exc}")

    return g_joined



def _map_obat_epoch_to_label(epoch: pd.Series) -> pd.Series:
    """
    Mappt die OBAT-Epochencodes (0–5) auf gut lesbare Baujahresklassen.

    OBAT-Epochen laut Spezifikation:
        0 = außerhalb des built-up domain
        1 = vor 1980
        2 = 1980–1990
        3 = 1990–2000
        4 = 2000–2010
        5 = 2010–2020
    """
    mapping = {
        0: "außerhalb built-up",
        1: "vor 1980",
        2: "1980–1990",
        3: "1990–2000",
        4: "2000–2010",
        5: "2010–2020",
    }
    return epoch.map(lambda v: mapping.get(int(v), pd.NA) if pd.notna(v) else pd.NA)


def _map_obat_epoch_to_midyear(epoch: pd.Series) -> pd.Series:
    """
    Approximiert ein 'Stichjahr' je OBAT-Epoche (Mittel der Zeitspanne).
    Diese Werte kannst Du später bei Bedarf an Deine IWU-Klassen anpassen.
    """
    midyears = {
        1: 1975,  # vor 1980 → ungefähr Mitte der 1970er
        2: 1985,  # 1980–1990
        3: 1995,  # 1990–2000
        4: 2005,  # 2000–2010
        5: 2015,  # 2010–2020
    }
    return epoch.map(lambda v: midyears.get(int(v), pd.NA) if pd.notna(v) else pd.NA)


def _map_obat_use(use_code: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Mappt OBAT-use (0–2) auf Textlabel + WG/NWG-Flags.

        0 = außerhalb built-up domain
        1 = residential
        2 = non-residential
    """
    def label(v):
        """
        F?hrt label aus.
        
        Args:
            v: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        if pd.isna(v):
            return pd.NA
        v = int(v)
        if v == 1:
            return "Wohngebäude (residential)"
        if v == 2:
            return "Nichtwohngebäude (non-residential)"
        if v == 0:
            return "außerhalb built-up domain"
        return pd.NA

    lbl = use_code.map(label)
    is_res = use_code.map(lambda v: (int(v) == 1) if pd.notna(v) else False)
    is_nres = use_code.map(lambda v: (int(v) == 2) if pd.notna(v) else False)
    return lbl, is_res.astype("boolean"), is_nres.astype("boolean")

def _add_empty_obat_columns(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ergänzt leere OBAT-Spalten, falls kein Datensatz / kein Join möglich ist.
    """
    default_cols = {
        "OBAT_id": pd.NA,
        "OBAT_country": pd.NA,
        "OBAT_adm1": pd.NA,
        "OBAT_height_m": pd.NA,
        "OBAT_shapefactor": pd.NA,
        "OBAT_area_m2": pd.NA,
        "OBAT_perimeter_m": pd.NA,
        "OBAT_use_code": pd.NA,
        "OBAT_use_label": pd.NA,
        "OBAT_is_residential": False,
        "OBAT_is_nonresidential": False,
        "OBAT_epoch_code": pd.NA,
        "OBAT_Baujahrklasse_OBAT": pd.NA,
        "OBAT_Baujahr_Mitte": pd.NA,
        "OBAT_dist_m": pd.NA,
    }
    for col, val in default_cols.items():
        if col not in buildings.columns:
            if isinstance(val, bool):
                buildings[col] = pd.Series(False, index=buildings.index, dtype="boolean")
            else:
                buildings[col] = pd.NA
    return buildings


def _enrich_with_ghs_obat(
    ctx: PipelineContext,
    buildings: gpd.GeoDataFrame,
    ghs_obat_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Ergänzt den Gebäudelayer um GHS-OBAT-Informationen (Baualtersklasse, Nutzung etc.).

    Vorgehen:
    - GHS-OBAT-Datensatz über _load_or_create_ghs_obat_dataset bereitstellen.
    - CRS mit Gebäude-Layer harmonisieren.
    - OBAT-Attribute vor dem Join in eindeutige OBAT_* Spalten umbenennen.
    - sjoin_nearest (Gebäude → nächster OBAT-Zentroid) mit Distanzspalte.
    - Mapping:
        * epoch (0–5) → OBAT_Baujahrklasse_OBAT (Text) + OBAT_Baujahr_Mitte (Stichjahr)
        * use (0–2)   → OBAT_use_label + WG/NWG-Flags
    - Ergebnis in out/ghs_obat:
        * ap1_buildings_enriched_ghs_obat.gpkg
        * ap1_buildings_enriched_ghs_obat.csv
    """

    settings = _get_settings(ctx)
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    obat_out_dir = out_base / "ghs_obat"
    obat_out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = obat_out_dir / "ap1_buildings_enriched_ghs_obat.gpkg"
    csv_path = obat_out_dir / "ap1_buildings_enriched_ghs_obat.csv"

    # ------------------------------------------------------------------ #
    # Sonderfälle: leere Gebäude- oder OBAT-Datensätze
    # ------------------------------------------------------------------ #

    if buildings.empty:
        print("[GHS-OBAT] Gebäude-Layer ist leer – setze nur leere OBAT-Spalten.")
        buildings = _add_empty_obat_columns(buildings)
        try:
            buildings.to_file(gpkg_path, layer="buildings_ghs_obat", driver="GPKG")
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")
        try:
            buildings.drop(columns="geometry", errors="ignore").to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte {csv_path} nicht schreiben: {exc}")
        return buildings

    if ghs_obat_gdf is None or ghs_obat_gdf.empty:
        print("[GHS-OBAT] Kein GHS-OBAT-Datensatz vorhanden – OBAT-Spalten leer.")
        buildings = _add_empty_obat_columns(buildings)
        try:
            buildings.to_file(gpkg_path, layer="buildings_ghs_obat", driver="GPKG")
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")
        try:
            buildings.drop(columns="geometry", errors="ignore").to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte {csv_path} nicht schreiben: {exc}")
        return buildings

    # ------------------------------------------------------------------ #
    # CRS harmonisieren
    # ------------------------------------------------------------------ #

    if ghs_obat_gdf.crs != buildings.crs:
        try:
            ghs_obat_gdf = ghs_obat_gdf.to_crs(buildings.crs)
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: CRS-Transformation OBAT->Buildings fehlgeschlagen: {exc}")
            buildings = _add_empty_obat_columns(buildings)
            return buildings

    # ------------------------------------------------------------------ #
    # Relevante OBAT-Spalten finden (Original-Namen im OBAT-Datensatz)
    # ------------------------------------------------------------------ #

    col_id = next((c for c in ("id", "Id", "ID") if c in ghs_obat_gdf.columns), None)
    col_country = next((c for c in ("country", "Country") if c in ghs_obat_gdf.columns), None)
    col_adm1 = next((c for c in ("adm1", "Adm1") if c in ghs_obat_gdf.columns), None)
    col_height = next((c for c in ("height", "Height") if c in ghs_obat_gdf.columns), None)
    col_shapef = next(
        (c for c in ("shapefactor", "ShapeFactor", "Shape_Factor") if c in ghs_obat_gdf.columns),
        None,
    )
    col_area = next((c for c in ("area", "Area") if c in ghs_obat_gdf.columns), None)
    col_perim = next((c for c in ("perimeter", "Perimeter") if c in ghs_obat_gdf.columns), None)
    col_use = next((c for c in ("use", "Use") if c in ghs_obat_gdf.columns), None)
    col_epoch = next((c for c in ("epoch", "Epoch") if c in ghs_obat_gdf.columns), None)

    # Nur wirklich benötigte Spalten + Geometrie in eine Kopie übernehmen
    keep_cols = ["geometry"]
    for c in (col_id, col_country, col_adm1, col_height, col_shapef, col_area, col_perim, col_use, col_epoch):
        if c:
            keep_cols.append(c)

    ghs_src = ghs_obat_gdf[keep_cols].copy()

    # OBAT-Spalten in eindeutige, „fixe“ Namen umbenennen
    rename_map = {}
    if col_id:
        rename_map[col_id] = "OBAT_id"
    if col_country:
        rename_map[col_country] = "OBAT_country"
    if col_adm1:
        rename_map[col_adm1] = "OBAT_adm1"
    if col_height:
        rename_map[col_height] = "OBAT_height_raw"
    if col_shapef:
        rename_map[col_shapef] = "OBAT_shapefactor_raw"
    if col_area:
        rename_map[col_area] = "OBAT_area_raw"
    if col_perim:
        rename_map[col_perim] = "OBAT_perimeter_raw"
    if col_use:
        rename_map[col_use] = "OBAT_use_code_raw"
    if col_epoch:
        rename_map[col_epoch] = "OBAT_epoch_code_raw"

    ghs_src = ghs_src.rename(columns=rename_map)

    # Auf Gebäudeseite evtl. alte OBAT-Spalten entfernen, um Kollisionen zu vermeiden
    buildings_clean = buildings.drop(
        columns=[c for c in buildings.columns if c.startswith("OBAT_")],
        errors="ignore",
    )

    # ------------------------------------------------------------------ #
    # sjoin_nearest: Gebäude → nächster OBAT-Punkt
    # ------------------------------------------------------------------ #

    try:
        joined = gpd.sjoin_nearest(
            buildings_clean,
            ghs_src,
            how="left",
            distance_col="OBAT_dist_m",
        )
    except Exception as exc:
        print(f"[GHS-OBAT] WARN: sjoin_nearest fehlgeschlagen ({exc}) – OBAT-Spalten bleiben leer.")
        buildings = _add_empty_obat_columns(buildings)
        return buildings

    # ------------------------------------------------------------------ #
    # OBAT-Spalten endgültig typisieren / aufräumen
    # ------------------------------------------------------------------ #

    # ID
    joined["OBAT_id"] = joined.get("OBAT_id", pd.NA)

    # Country / Adm1
    joined["OBAT_country"] = joined.get("OBAT_country", pd.NA)
    joined["OBAT_adm1"] = joined.get("OBAT_adm1", pd.NA)

    # Höhe, Shape, Fläche, Umfang (numerisch)
    joined["OBAT_height_m"] = (
        pd.to_numeric(joined["OBAT_height_raw"], errors="coerce")
        if "OBAT_height_raw" in joined.columns
        else pd.NA
    )
    joined["OBAT_shapefactor"] = (
        pd.to_numeric(joined["OBAT_shapefactor_raw"], errors="coerce")
        if "OBAT_shapefactor_raw" in joined.columns
        else pd.NA
    )
    joined["OBAT_area_m2"] = (
        pd.to_numeric(joined["OBAT_area_raw"], errors="coerce")
        if "OBAT_area_raw" in joined.columns
        else pd.NA
    )
    joined["OBAT_perimeter_m"] = (
        pd.to_numeric(joined["OBAT_perimeter_raw"], errors="coerce")
        if "OBAT_perimeter_raw" in joined.columns
        else pd.NA
    )

    # Nutzung & Epochencode (numerisch, Int64)
    if "OBAT_use_code_raw" in joined.columns:
        joined["OBAT_use_code"] = pd.to_numeric(
            joined["OBAT_use_code_raw"], errors="coerce"
        ).astype("Int64")
    else:
        joined["OBAT_use_code"] = pd.NA

    if "OBAT_epoch_code_raw" in joined.columns:
        joined["OBAT_epoch_code"] = pd.to_numeric(
            joined["OBAT_epoch_code_raw"], errors="coerce"
        ).astype("Int64")
    else:
        joined["OBAT_epoch_code"] = pd.NA

    # Textlabel + Flags für Nutzung
    if "OBAT_use_code" in joined.columns:
        lbl, is_res, is_nres = _map_obat_use(joined["OBAT_use_code"])
        joined["OBAT_use_label"] = lbl
        joined["OBAT_is_residential"] = is_res
        joined["OBAT_is_nonresidential"] = is_nres
    else:
        joined["OBAT_use_label"] = pd.NA
        joined["OBAT_is_residential"] = pd.Series(False, index=joined.index, dtype="boolean")
        joined["OBAT_is_nonresidential"] = pd.Series(False, index=joined.index, dtype="boolean")

    # Baualtersklassen + Stichjahr
    if "OBAT_epoch_code" in joined.columns:
        joined["OBAT_Baujahrklasse_OBAT"] = _map_obat_epoch_to_label(
            joined["OBAT_epoch_code"]
        )
        joined["OBAT_Baujahr_Mitte"] = _map_obat_epoch_to_midyear(
            joined["OBAT_epoch_code"]
        )
    else:
        joined["OBAT_Baujahrklasse_OBAT"] = pd.NA
        joined["OBAT_Baujahr_Mitte"] = pd.NA

    # Aufräumen: Join-Hilfsfelder und *_raw entfernen
    joined = joined.drop(
        columns=[c for c in joined.columns if c.startswith("index_")],
        errors="ignore",
    )
    joined = joined.drop(
        columns=[c for c in joined.columns if c.endswith("_raw")],
        errors="ignore",
    )

    # ------------------------------------------------------------------ #
    # Ergebnis schreiben
    # ------------------------------------------------------------------ #

    try:
        joined.to_file(gpkg_path, layer="buildings_ghs_obat", driver="GPKG")
        print(f"[GHS-OBAT] Gebäude+OBAT als GPKG nach {gpkg_path} geschrieben.")
    except Exception as exc:
        print(f"[GHS-OBAT] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")

    try:
        joined.drop(columns="geometry", errors="ignore").to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        print(f"[GHS-OBAT] Gebäude+OBAT als CSV nach {csv_path} geschrieben.")
    except Exception as exc:
        print(f"[GHS-OBAT] WARN: Konnte {csv_path} nicht schreiben: {exc}")

    return joined


def _load_or_create_ghs_obat_dataset(ctx: PipelineContext) -> gpd.GeoDataFrame:
    """
    Stellt sicher, dass ein GHS-OBAT-Datensatz für das Projektgebiet existiert.

    Logik:
    - Wenn in settings.data.ghs_obat_path ein expliziter Pfad zu einer bereits
      zugeschnittenen Datei gesetzt ist und diese existiert → direkt laden.
    - Sonst wird ein Standard-Clip unter out/ghs_obat/ghs_obat_clip.gpkg erwartet
      bzw. erzeugt:
        * existiert er bereits → laden
        * existiert er nicht → GhsObatSource().load(ctx) aufrufen, der die
          Rohdatei aus data/ lädt, auf target_epsg bringt, auf die BBOX clippt
          und den Clip zurückgibt; dieser wird dann als ghs_obat_clip.gpkg
          gespeichert.

    Falls GhsObatSource nicht importierbar ist, wird ein leerer GeoDataFrame
    mit der Ziel-CRS (target_epsg) zurückgegeben.
    """

    settings = _get_settings(ctx)
    data_cfg = _get_data_cfg(settings)

    # Basis-Ausgabeverzeichnis wie bei AP1 / Zensus / DIVIS
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    ghs_out_dir = out_base / "ghs_obat"
    ghs_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) expliziter Pfad aus Settings (falls gesetzt)
    ghs_path_cfg = getattr(data_cfg, "ghs_obat_path", None)
    if ghs_path_cfg:
        ghs_path = Path(ghs_path_cfg)
        if ghs_path.exists():
            try:
                print(f"[GHS-OBAT] Lade GHS-OBAT-Datensatz aus settings.data.ghs_obat_path: {ghs_path}")
                return gpd.read_file(ghs_path)
            except Exception as exc:
                print(f"[GHS-OBAT] WARN: Konnte Datei aus settings.data.ghs_obat_path ({ghs_path}) nicht lesen: {exc}")
        else:
            print(f"[GHS-OBAT] WARN: Pfad aus settings.data.ghs_obat_path existiert nicht: {ghs_path}")

    # 2) Standard-Clip-Datei im out-Verzeichnis
    clip_path = ghs_out_dir / "ghs_obat_clip.gpkg"

    # 2a) vorhandenen Clip laden, falls möglich
    if clip_path.exists():
        try:
            print(f"[GHS-OBAT] Verwende bestehenden GHS-OBAT-Clip: {clip_path}")
            return gpd.read_file(clip_path)
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte bestehenden Clip {clip_path} nicht lesen: {exc}")

    # 3) Sonst: über Source erzeugen
    try:
        from kwp_bedarfskennwerte.data_catalog.sources import GhsObatSource
    except ImportError as exc:
        print(f"[GHS-OBAT] WARN: GhsObatSource nicht importierbar: {exc}")
        target_epsg = int(getattr(settings, "target_epsg", getattr(ctx, "target_epsg", 25833)))
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

    # Wenn ein expliziter Rohpfad konfiguriert ist, an die Source übergeben,
    # sonst übernimmt GhsObatSource intern die Suche im data/-Ordner.
    src = GhsObatSource(path=ghs_path_cfg) if ghs_path_cfg else GhsObatSource()
    gdf = src.load(ctx)

    if gdf is None:
        gdf = gpd.GeoDataFrame(geometry=[])
    if gdf.empty:
        print("[GHS-OBAT] WARN: GhsObatSource.load(ctx) hat einen leeren Datensatz geliefert.")
        return gdf

    # 4) Clip zusätzlich unter clip_path persistieren, falls noch nicht vorhanden
    if not clip_path.exists():
        try:
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(clip_path, layer="ghs_obat", driver="GPKG")
            print(f"[GHS-OBAT] Clip nach {clip_path} geschrieben.")
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte Clip {clip_path} nicht schreiben: {exc}")

    return gdf


def _enrich_with_hk(ctx: PipelineContext, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reichert Gebäude um HK-DE Adressattribute an (insb. PLZ).

    Motivation:
      - Klimakorrektur der Heizbedarfskennwerte benötigt eine regionale Zuordnung.
      - PLZ ist ein robuster Schlüssel und sollte früh im Workflow verfügbar sein.

    Steuerung über Settings/CLI (Defaults sind bewusst tolerant):
      - settings.enrich_with_hk_addresses (bool, default True)
      - settings.hk_path (optional, str/Path)
      - settings.hk_place_filter (optional, list[str])  # z.B. ['Chemnitz'] für Tests
      - settings.hk_building_id_col (optional, default 'LOD_UNITID')

    Verhalten:
      - Wenn HK-Datei fehlt oder Join-Key nicht existiert -> Warnung und unveränderte Gebäude zurück.
      - Join-Key ist standardmäßig 'LOD_UNITID' (muss HK-DE `oid` entsprechen).
    """
    settings = _get_settings(ctx)

    # Option: DIVIS überspringen (CLI: --skip-divis / ENV: KWP_SKIP_DIVIS=1)
    skip_divis = bool(getattr(settings, 'skip_divis', False) or getattr(settings, 'enrich_skip_divis', False))
    if not skip_divis:
        env_flag = os.environ.get('KWP_SKIP_DIVIS', '0').strip().lower() in ('1','true','yes','y')
        skip_divis = env_flag
    if skip_divis and verbose:
        print('[ap1_enrich] --skip-divis aktiv: DIVIS-Enrichment wird übersprungen.')

    if not bool(getattr(settings, "enrich_with_hk_addresses", True)):
        return buildings

    building_id_col = getattr(settings, "hk_building_id_col", "LOD_UNITID")

    if building_id_col not in buildings.columns:
        print(
            f"[HK] WARN: Join-Key '{building_id_col}' fehlt im Gebäudelayer – "
            "überspringe HK-Adressanreicherung."
        )
        return buildings

    hk_path = getattr(settings, "hk_path", None)
    place_filter = getattr(settings, "hk_place_filter", None)

    # base_dir = Projektwurzel (wie in addresses_hk erwartet)
    base_dir = Path(getattr(settings, "project_root", getattr(ctx, "project_root", Path.cwd())))

    # --------------------------------------------------------------
    # Cache/Idempotenz:
    # ap1-enrich wird im Standard-Workflow mehrfach gestartet
    # (only-zensus / only-obat / full). Damit die HK-Anreicherung nicht
    # jedes Mal erneut läuft, nutzen wir einen vorhandenen Cache nur dann,
    # wenn die Feature-Anzahl zur aktuellen Basis passt.
    # --------------------------------------------------------------
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    hk_out_dir = out_base / "hk_addresses"
    gpkg_cache = hk_out_dir / "ap1_buildings_enriched_hk.gpkg"

    cached = _try_load_cached_layer(
        gpkg_cache,
        "buildings_hk",
        expected_len=len(buildings),
        verbose=True,
        label="HK-Cache",
    )
    if cached is not None and "HK_match" in cached.columns:
        # WICHTIG: Cache niemals als komplettes DF zurückgeben, sonst können
        # bereits vorhandene Anreicherungen (DIVIS/OBAT/Zensus etc.) verloren gehen,
        # wenn der Cache aus einem früheren Schritt stammt.
        # Stattdessen: nur HK-Spalten positionsbasiert in den aktuellen Layer kopieren.
        b = buildings.reset_index(drop=True).copy()
        c = cached.reset_index(drop=True)

        hk_cols = [col for col in c.columns if col == "HK_match" or col.startswith("HK_")]
        for col in hk_cols:
            b[col] = c[col].values

        return b



    try:
        enriched = enrich_buildings_with_hk_addresses(
            buildings,
            base_dir=base_dir,
            building_id_col=building_id_col,
            hk_path=hk_path,
            place_filter=place_filter,
            cols=HKAddressColumns(),
        )
    except FileNotFoundError as exc:
        print(f"[HK] WARN: {exc} – überspringe HK-Adressanreicherung.")
        return buildings
    except Exception as exc:
        print(f"[HK] WARN: HK-Adressanreicherung fehlgeschlagen ({exc}) – überspringe.")
        return buildings

    # Debug-Output (QA/QGIS)
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    hk_out_dir = out_base / "hk_addresses"
    hk_out_dir.mkdir(parents=True, exist_ok=True)

    gpkg_path = hk_out_dir / "ap1_buildings_enriched_hk.gpkg"
    csv_path = hk_out_dir / "ap1_buildings_enriched_hk.csv"

    try:
        enriched.to_file(gpkg_path, layer="buildings_hk", driver="GPKG")
        print(f"[HK] Gebäude+HK als GPKG nach {gpkg_path} geschrieben.")
    except Exception as exc:
        print(f"[HK] WARN: Konnte {gpkg_path} nicht schreiben: {exc}")

    try:
        enriched.drop(columns="geometry", errors="ignore").to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        print(f"[HK] Gebäude+HK als CSV nach {csv_path} geschrieben.")
    except Exception as exc:
        print(f"[HK] WARN: Konnte {csv_path} nicht schreiben: {exc}")

    return enriched



def run_enrichment(
    ctx: PipelineContext,
    verbose: bool = True,
    skip_divis: bool = False,
) -> Path:
    """
    Führt die Anreicherungs-Schritte auf dem AP1-Ergebnis aus:

    - DIVIS (Denkmalstatus, Baujahr, Erläuterung; inkl. einfacher Denkmal-Flag)
      → Ergebnis: out/divis/divis_buildings_enriched.gpkg/.csv
    - Zensus (Gebäudetyp/Bauweise, Baujahresklassen, Energieträger, Heizungsart,
              demographischer Indikator 65+, Leerstands- und Eigentumsquote)
      → Ergebnis: out/zensus/ap1_buildings_enriched_zensus.gpkg/.csv

    VARIANTE B (neu):
    - OBAT wird im Standardlauf automatisch erzeugt, falls kein kompatibler OBAT-Cache existiert.
      → Ergebnis: out/ghs_obat/ap1_buildings_enriched_ghs_obat.gpkg/.csv

    Steuerung über Settings/CLI:
    - settings.enrich_only_divis  (Flag: --only-divis)
    - settings.enrich_only_zensus (Flag: --only-zensus)

    Rückgabe:
    - Pfad zur wichtigsten erzeugten GPKG-Datei:
      * nur DIVIS  → out/divis/divis_buildings_enriched.gpkg
      * nur Zensus → out/zensus/ap1_buildings_enriched_zensus.gpkg
      * beide      → out/zensus/ap1_buildings_enriched_zensus.gpkg
    """
    # NOTE: skip_divis kommt aus der CLI (--skip-divis) und muss hier als
    # Parameter existieren. (Sonst: TypeError/NameError, wenn die CLI das
    # Keyword übergibt bzw. der Code es referenziert.)
    skip_divis = bool(skip_divis)

    settings = _get_settings(ctx)

    only_divis = bool(getattr(settings, "enrich_only_divis", False))
    only_zensus = bool(getattr(settings, "enrich_only_zensus", False))

    zensus_ran_early = False  # Zensus im Standardlauf früh ausführen (auch wenn DIVIS später abgebrochen wird)

    # Schutz: beides gleichzeitig ist nicht sinnvoll
    if only_divis and only_zensus:
        raise ValueError(
            "Konflikt: enrich_only_divis und enrich_only_zensus wurden "
            "gleichzeitig gesetzt. Bitte nur eines von beiden verwenden."
        )

    # Basis-Verzeichnisse / Pfade
    out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
    divis_out_dir = out_base / "divis"
    zensus_out_dir = out_base / "zensus"
    ghs_out_dir = out_base / "ghs_obat"
    divis_out_dir.mkdir(parents=True, exist_ok=True)
    zensus_out_dir.mkdir(parents=True, exist_ok=True)
    ghs_out_dir.mkdir(parents=True, exist_ok=True)

    divis_gpkg = divis_out_dir / "divis_buildings_enriched.gpkg"
    zensus_gpkg = zensus_out_dir / "ap1_buildings_enriched_zensus.gpkg"
    obat_gpkg = ghs_out_dir / "ap1_buildings_enriched_ghs_obat.gpkg"

    # ------------------------------------------------------------------
    # 1) AP1-Basis laden
    # ------------------------------------------------------------------
    if verbose:
        print("[ap1_enrich] Lade AP1-Gebäudelayer...")

    base_buildings = _load_ap1_buildings(ctx)

    # ------------------------------------------------------------------
    # 1b) OBAT als Basis: Cache nur nutzen, wenn Featurecount kompatibel
    # ------------------------------------------------------------------
    cached_obat = _try_load_cached_layer(
        obat_gpkg,
        "buildings_ghs_obat",
        expected_len=len(base_buildings),
        verbose=verbose,
        label="GHS-OBAT-Basis",
    )

    if cached_obat is not None:
        if verbose:
            print(f"[ap1_enrich] Verwende vorhandenen GHS-OBAT-Datensatz als Basis: {obat_gpkg}")
        base_buildings = cached_obat

        # HK-Spalten sicherstellen (ohne Überschreiben anderer Enrichments)
        base_buildings = _enrich_with_hk(ctx, base_buildings)

    else:
        # ==============================================================
        # VARIANTE B: OBAT im Standardlauf automatisch erzeugen
        # ==============================================================
        # 1) HK früh ergänzen (PLZ etc.) - idempotent, cache-sicher
        base_buildings = _enrich_with_hk(ctx, base_buildings)

        # 2) OBAT-Datensatz laden/erzeugen (Clip)
        if verbose:
            print("[ap1_enrich] (auto-obat) Lade/erstelle GHS-OBAT-Datensatz ...")
        ghs_obat_gdf = _load_or_create_ghs_obat_dataset(ctx)

        # 3) OBAT-Join durchführen + Ergebnis schreiben (out/ghs_obat/...)
        if verbose:
            print("[ap1_enrich] (auto-obat) Ergänze GHS-OBAT-Informationen ...")
        base_buildings = _enrich_with_ghs_obat(ctx, base_buildings, ghs_obat_gdf)

    # ------------------------------------------------------------------
    # 2) Fall: nur DIVIS
    # ------------------------------------------------------------------
    if only_divis and not only_zensus:
        if skip_divis:
            raise ValueError('Konflikt: --only-divis kann nicht mit --skip-divis kombiniert werden.')

        if verbose:
            print("[ap1_enrich] Ergänze nur DIVIS-Informationen (ohne Zensus)...")
        _ = _enrich_with_divis(ctx, base_buildings)
        return divis_gpkg

    # ------------------------------------------------------------------
    # 2b) ZENSUS im Standardlauf früh ausführen (zensus-first)
    # ------------------------------------------------------------------
    if not only_divis:
        buildings_for_zensus_early = base_buildings
        if (not skip_divis) and "DIVIS_flag" not in buildings_for_zensus_early.columns:
            cached_divis_for_zensus_early = _try_load_cached_layer(
                divis_gpkg,
                "buildings_divis",
                expected_len=len(buildings_for_zensus_early),
                verbose=verbose,
                label="DIVIS-Cache (zensus-first)",
            )
            if cached_divis_for_zensus_early is not None:
                buildings_for_zensus_early = cached_divis_for_zensus_early

        # Wenn Zensus-Ausgabe bereits existiert und zur Basis passt, wiederverwenden
        if zensus_gpkg.exists():
            cached_zensus = _try_load_cached_layer(
                zensus_gpkg,
                "buildings_zensus",
                expected_len=len(buildings_for_zensus_early),
                verbose=verbose,
                label="Zensus-Cache (zensus-first)",
            )
            if cached_zensus is not None:
                if verbose:
                    print(f"[ap1_enrich] Zensus bereits vorhanden (Cache): {zensus_gpkg}")
                zensus_ran_early = True

        if not zensus_ran_early:
            if verbose:
                print("[ap1_enrich] (zensus-first) Lade/erstelle Zensus-100m-Grid...")
            zensus_gdf_early = _load_or_create_zensus_dataset(ctx)

            if verbose:
                print("[ap1_enrich] (zensus-first) Ergänze Zensus-Informationen (inkl. Baujahrsphasen)...")
            _ = _enrich_with_zensus(ctx, buildings_for_zensus_early, zensus_gdf=zensus_gdf_early)
            zensus_ran_early = True

        if only_zensus:
            return zensus_gpkg

    # ------------------------------------------------------------------
    # 3) Basis für den ZENSUS-Schritt bestimmen
    # ------------------------------------------------------------------
    if not only_zensus and (not skip_divis):
        if verbose:
            print("[ap1_enrich] Ergänze DIVIS-Informationen...")
        buildings_for_zensus = _enrich_with_divis(ctx, base_buildings)
    elif not only_zensus and skip_divis:
        buildings_for_zensus = base_buildings
    else:
        cached_obat_for_zensus = _try_load_cached_layer(
            obat_gpkg,
            "buildings_ghs_obat",
            expected_len=len(base_buildings),
            verbose=verbose,
            label="OBAT-Basis (only-zensus)",
        )
        if cached_obat_for_zensus is not None:
            if verbose:
                print(
                    f"[ap1_enrich] only-zensus: Lade Basis aus OBAT-GPKG {obat_gpkg} "
                    "(enthält AP1+DIVIS+OBAT)..."
                )
            buildings_for_zensus = cached_obat_for_zensus
        else:
            cached_divis_for_zensus = _try_load_cached_layer(
                divis_gpkg,
                "buildings_divis",
                expected_len=len(base_buildings),
                verbose=verbose,
                label="DIVIS-Basis (only-zensus)",
            )
            if cached_divis_for_zensus is not None:
                if verbose:
                    print(f"[ap1_enrich] only-zensus: Lade vorhandenes DIVIS-Ergebnis {divis_gpkg} ...")
                buildings_for_zensus = cached_divis_for_zensus
            else:
                buildings_for_zensus = base_buildings

    # ------------------------------------------------------------------
    # 4) ZENSUS-100m-Grid laden/erzeugen
    # ------------------------------------------------------------------
    if zensus_ran_early and zensus_gpkg.exists():
        if verbose:
            print(f"[ap1_enrich] Zensus bereits im Lauf erstellt – überspringe erneute Berechnung: {zensus_gpkg}")
    else:
        if verbose:
            print("[ap1_enrich] Lade/erstelle Zensus-100m-Grid...")
        zensus_gdf = _load_or_create_zensus_dataset(ctx)

        if verbose:
            print("[ap1_enrich] Ergänze Zensus-Informationen (inkl. Baujahrsphasen)...")
        _ = _enrich_with_zensus(ctx, buildings_for_zensus, zensus_gdf=zensus_gdf)

    # ------------------------------------------------------------------ #
    # 6) Optional: statistische Analyse der Zensus-Ergebnisse
    # ------------------------------------------------------------------ #
    if getattr(settings, "enrich_analyse", False):
        zensus_csv = zensus_out_dir / "ap1_buildings_enriched_zensus.csv"
        if zensus_csv.exists():
            if verbose:
                print(f"[ZENSUS] Starte Statistik-Analyse für {zensus_csv} ...")
            stats = AP1EnrichCSVStatistics()
            stats.run(
                csv_path=zensus_csv,
                out_dir=zensus_out_dir,
                gpkg_path=zensus_gpkg,
            )
            if verbose:
                print("[ZENSUS] Statistik-Analyse abgeschlossen.")
        else:
            if verbose:
                print(
                    f"[ZENSUS] Statistik-Analyse übersprungen – "
                    f"CSV {zensus_csv} wurde nicht gefunden."
                )

    return zensus_gpkg
