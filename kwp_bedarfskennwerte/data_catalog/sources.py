"""
Datenquellenzugriff und Harmonisierung.

Konsolidiert den Zugriff auf die relevanten Datensätze der Studie:
- LoD2 (Geometrie und Nutzungscodes aus ALKIS/ATKIS)
- OSM (Gebäudenutzung, Adressen, Zusatzattribute)
- Basemap.de (Gebäudeflächen- und Nutzungslayer)
- Zensus 2022 (100m-Gitter, Heizträger, Baujahre)
- GHS-OBAT (Baujahr-/Form-/Höhenattribute)
- DIVIS (Denkmalliste Sachsen via WMS)

Die Quelle dient als Schnittstelle zur AP1-Pipeline und standardisiert
CRS, Attributnamen und QA-Ausgaben.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Dict, Optional, Iterable, Tuple, Any, List
import os
import gzip
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shp_box, Polygon, shape as shp_shape
from shapely.ops import unary_union
from pyproj import Transformer, CRS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ..config.runtime import PipelineContext
from ..config.paths import DIR_BAUJAHRE_OBAT, rel_zensus2022_dir
from urllib.parse import urlencode
import logging
from pathlib import Path


# =============================================================================
# Gemeinsamer Contract + Registry
# =============================================================================

class Source(Protocol):
    """Protokoll für Datenquellen in der Registry."""

    def available(self, ctx: PipelineContext) -> bool:
        """Prüft, ob die Quelle im Kontext verfügbar ist."""
        ...

    def load(self, ctx: PipelineContext) -> gpd.GeoDataFrame:
        """Lädt die Quelle in ein GeoDataFrame."""
        ...

@dataclass
class SourceConfig:
    """Konfigurationseintrag für eine Datenquelle."""
    name: str
    required: bool = True

SOURCE_REGISTRY: Dict[str, type] = {}

def register(name: str):
    """Registriert eine Datenquelle unter einem Namen."""
    def _wrap(cls):
        SOURCE_REGISTRY[name] = cls
        return cls
    return _wrap

def load_all(ctx: PipelineContext, names: Optional[Iterable[str]] = None) -> Dict[str, gpd.GeoDataFrame]:
    """Lädt alle registrierten Quellen oder eine benannte Auswahl."""
    selected = (names or SOURCE_REGISTRY.keys())
    outputs: Dict[str, gpd.GeoDataFrame] = {}
    for name in selected:
        cls = SOURCE_REGISTRY.get(name)
        if cls is None:
            print(f"[sources] WARN: Quelle nicht registriert: {name}")
            continue
        src: Source = cls()  # type: ignore
        try:
            if not src.available(ctx):
                print(f"[sources] SKIP: {name} → not available")
                continue
            gdf = src.load(ctx)
            outputs[name] = gdf
            print(f"[sources] OK: {name} → {len(gdf)} Features")
        except Exception as exc:
            print(f"[sources] ERROR: {name}: {exc}")
            if getattr(src, "required", True):
                raise
    return outputs


# =============================================================================
# LoD2 – Nutzungstypen aus lokaler Datei (3D erhalten; UNITID als String)
# =============================================================================

@register("lod2")
class LoD2CityGMLSource:
    """
    Liest LoD1/LoD2 robust ein (Z-Koordinate bleibt erhalten), fixiert die Geometriespalte,
    konvertiert UNITID zu String, verwirft Nicht-Gebäude (leere 'GebFunkion').

    - Wenn Spalte 'ELEMCLASS' vorhanden ist, werden nur Features mit ELEMCLASS == "Ground"
      berücksichtigt (Grundflächen / Footprints).
    - Nur für diese Ground-Features wird die 2D-Grundfläche berechnet.
    - Pro Gebäude-ID wird genau ein Eintrag behalten (größte Grundfläche).
    - LOD_UNITID ist identisch zu building_id.
    - LOD_Nutzung_original enthält – soweit möglich – die Zahlenkennung (z.B. 31001_1000),
      LOD_Nutzung_vereinheitlicht den zugehörigen textlichen Nutzungstyp.
    """

    _ID_CANDS = [
        "UNITID","unitid","UNIT_ID","unit_id","GEBID","GEBIDBY","ALKISOID",
        "gml_id","GML_ID","building_id","BUILDING_ID","obj_id","OBJ_ID","uuid","UUID",
    ]
    _FUNC_CANDS = [
        "GebFunkion","gebfunkion",
        "GEBFUNKTION","gebfunktion","FUNKT_NAME","funktion","function","FUNCTION",
        "nutzung","NUTZUNG","GEBAEUDEART","Bauwerksfunktion","Bauwerksart","Bauwerkstyp",
    ]
    _NAME_CANDS = ["UNITNAME","unitname","name","NAME","bezeich","BEZEICH"]
    # typische Kandidaten für Gebäudehöhe
    _H_CANDS = [
        "GebMsHoehe","GEBMSHOEHE",
        "GebHoehe","GEBHOEHE",
        "HOEHEGEB","HoeheGeb",
        "height","HEIGHT",
        "measuredheight","MEASUREDHEIGHT",
    ]

    # Offizielle Zuordnung Zahlenkennung → Nutzungstyp
    _FUNC_CODE_TO_LABEL = {
        "31001_1000": "Wohngebäude",
        "31001_2000": "Gebäude für Wirtschaft oder Gewerbe",
        "31001_2072": "Jugendherberge",
        "31001_2461": "Parkhaus",
        "31001_2465": "Tiefgarage",
        "31001_2513": "Wasserbehälter",
        "31001_2523": "Umformer",
        "31001_3000": "Gebäude für öffentliche Zwecke",
        "31001_3012": "Rathaus",
        "31001_3017": "Kreisverwaltung",
        "31001_3018": "Bezirksregierung",
        "31001_3020": "Gebäude für Bildung und Forschung",
        "31001_3031": "Schloss",
        "31001_3038": "Burg, Festung",
        "31001_3041": "Kirche",
        "31001_3042": "Synagoge",
        "31001_3043": "Kapelle",
        "31001_3046": "Moschee",
        "31001_3047": "Tempel",
        "31001_3048": "Kloster",
        "31001_3051": "Krankenhaus",
        "31001_3052": "Heilanstalt, Pflegeanstalt, Pflegestation",
        "31001_3065": "Kinderkrippe, Kindergarten, Kindertagesstätte",
        "31001_3071": "Polizei",
        "31001_3072": "Feuerwehr",
        "31001_3073": "Kaserne",
        "31001_3075": "Justizvollzugsanstalt",
        "31001_3091": "Bahnhofsgebäude",
        "31001_3242": "Sanatorium",
        "31001_3290": "Touristisches Informationszentrum",
        "31001_9998": "Nach Quellenlage nicht zu spezifizieren",
        "51009_1610": "Überdachung",
    }

    # Heuristische Beispiele für Fallback-Mapping aus Freitext
    _FUNC_MAP_EXAMPLES = {
        "wohn": "Wohngebäude",
        "schule": "Gebäude für Bildung und Forschung",
        "hochschule": "Gebäude für Bildung und Forschung",
        "verwaltung": "Gebäude für öffentliche Zwecke",
        "kirche": "Kirche",
        "kranken": "Krankenhaus",
        "industrie": "Gebäude für Wirtschaft oder Gewerbe",
        "gewerbe": "Gebäude für Wirtschaft oder Gewerbe",
        "büro": "Gebäude für Wirtschaft oder Gewerbe",
        "buero": "Gebäude für Wirtschaft oder Gewerbe",
        "hotel": "Gebäude für Wirtschaft oder Gewerbe",
    }

    required = True

    @staticmethod
    def _find_first(cols, candidates):
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    def available(self, ctx: PipelineContext) -> bool:
        """
        Prüft die Verfügbarkeit der Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        p = ctx.settings.data.lod2_path
        return bool(p and Path(p).exists())

    # --- robustes Lesen (Z erhalten) ---
    def _read_vector(self, path: Path) -> gpd.GeoDataFrame:
        # 1) normal (behält meist Z)
        try:
            return gpd.read_file(path)
        except Exception:
            pass
        # 2) Fiona-Fallback – nur (Multi)Polygon; Z bleibt
        import fiona
        feats = []
        with fiona.open(path) as src:
            crs = src.crs_wkt or src.crs
            for rec in src:
                geom = rec.get("geometry")
                if not geom:
                    continue
                try:
                    g = shp_shape(geom)
                except Exception:
                    continue
                if g.is_empty or g.geom_type not in ("Polygon","MultiPolygon"):
                    continue
                props = rec.get("properties") or {}
                feats.append({"geometry": g, **props})
        return gpd.GeoDataFrame(feats, geometry="geometry", crs=crs)

    @staticmethod
    def _force_geometry(gdf: gpd.GeoDataFrame, out_dir: Path) -> gpd.GeoDataFrame:
        # Spaltenliste loggen
        try:
            (out_dir / "lod2_columns.txt").write_text("\n".join(map(str, gdf.columns)), encoding="utf-8")
        except Exception:
            pass

        # aktive Geometrie->
        try:
            _ = gdf.geometry
            current = gdf.geometry.name
        except Exception:
            current = None

        if current is None:
            for cand in ["geometry","geom","GEOMETRY","wkb_geometry","the_geom","Shape","shape"]:
                if cand in gdf.columns:
                    try:
                        gdf = gdf.set_geometry(cand)
                        current = cand
                        break
                    except Exception:
                        continue

        if current is None:
            msg = f"Keine aktive Geometriespalte. Spalten: {list(gdf.columns)}"
            (out_dir / "lod2_geometry_error.txt").write_text(msg, encoding="utf-8")
            raise ValueError(msg)

        if current != "geometry":
            if "geometry" in gdf.columns and gdf.geometry.name != "geometry":
                gdf = gdf.rename(columns={"geometry":"_geometry_old"})
            gdf = gdf.rename(columns={current:"geometry"})
            gdf.set_geometry("geometry", inplace=True)
        return gdf

    @staticmethod
    def _to_2d(g):
        from shapely.ops import transform as shp_transform
        try:
            return shp_transform(lambda x, y, z=None: (x, y), g)
        except Exception:
            return g

    def _map_function_fallback(self, series: pd.Series) -> pd.Series:
        """Heuristik für Fälle ohne bekannte Zahlenkennung."""
        def map_one(x):
            """
            Mappt one.
            
            Args:
                x: Beschreibung.
            
            Returns:
                Beschreibung.
            """
            if pd.isna(x):
                return "Nach Quellenlage nicht zu spezifizieren"
            s = str(x).strip().lower()
            for k, lbl in self._FUNC_MAP_EXAMPLES.items():
                if k in s:
                    return lbl
            return "Nach Quellenlage nicht zu spezifizieren"
        return series.map(map_one)

    def load(self, ctx: PipelineContext) -> gpd.GeoDataFrame:
        """
        Lädt die Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        out_dir = Path(ctx.settings.out_dir) / "ap1"
        out_dir.mkdir(parents=True, exist_ok=True)

        p = Path(ctx.settings.data.lod2_path)
        if not p.exists():
            print("[lod2] Datei fehlt.")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{ctx.settings.target_epsg}")

        gdf = self._read_vector(p)
        gdf = self._force_geometry(gdf, out_dir)

        # CRS harmonisieren – danach sind Koordinaten im Ziel-CRS (z.B. EPSG:25833 → Meter → m²)
        if gdf.crs is None or str(gdf.crs).strip().lower() in ("","none"):
            gdf.set_crs(epsg=ctx.settings.target_epsg, inplace=True)
        else:
            try:
                tgt = CRS.from_epsg(ctx.settings.target_epsg)
                if CRS.from_user_input(gdf.crs).to_epsg() != tgt.to_epsg():
                    gdf = gdf.to_crs(epsg=ctx.settings.target_epsg)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # IDs & Grundflächen (strenge UNITID-Variante, keine frühen Filter)
        # ------------------------------------------------------------------

        # 1) ID-Spalte ermitteln (strenge Variante: es muss eine echte ID aus _ID_CANDS geben)
        id_col = self._find_first(gdf.columns, self._ID_CANDS)
        if not id_col:
            print("[lod2] WARN: Keine ID-Spalte aus _ID_CANDS gefunden – gebe leeres GeoDataFrame zurück.")
            return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

        # Nur Zeilen mit gesetzter ID behalten
        gdf = gdf[~gdf[id_col].isna() & (gdf[id_col].astype(str).str.strip() != "")].copy()
        gdf[id_col] = gdf[id_col].astype(str)

        # 2) Funktions- und Namensspalten nachziehen (können später für Nutzungsmapping genutzt werden)
        fun_col = self._find_first(gdf.columns, self._FUNC_CANDS)  # meist Zahlenkennung
        name_col = self._find_first(gdf.columns, self._NAME_CANDS)  # UNITNAME

        # WICHTIG:
        # - Kein Filter mehr auf leere GebFunkion (Nicht-Gebäude) – Gebäude ohne gepflegte Funktion bleiben drin.
        # - Kein Filter mehr auf ELEMCLASS == "Ground" – alle Flächen mit UNITID bleiben erhalten.
        #   (Ground-/Nicht-Ground-Flächen werden später in der Pipeline je UNITID behandelt.)

        # 3) 2D-Footprint und Grundfläche für alle Flächen (in m², da CRS metrisch)
        gdf["__geom2d"] = gdf.geometry.apply(self._to_2d)
        gdf["area_m2"] = gdf["__geom2d"].area

        # ------------------------------------------------------------------
        # Nutzung: LOD_Nutzung_original (Kennung) & LOD_Nutzung_vereinheitlicht (Text)
        # ------------------------------------------------------------------
        if fun_col and fun_col in gdf.columns:
            raw_fun = gdf[fun_col].astype(str).str.strip()
        elif name_col and name_col in gdf.columns:
            raw_fun = gdf[name_col].astype(str).str.strip()
        else:
            raw_fun = pd.Series([""] * len(gdf), index=gdf.index)

        # Kennung nur dann übernehmen, wenn sie im Mapping vorkommt
        code_series = raw_fun.where(raw_fun.isin(self._FUNC_CODE_TO_LABEL.keys()), pd.NA)

        # Textlabel: zuerst aus Kennung, sonst heuristischer Fallback
        label_series = raw_fun.map(lambda v: self._FUNC_CODE_TO_LABEL.get(v, None))
        missing_mask = label_series.isna()
        if missing_mask.any():
            # Fallback aus Freitext (oder "Nach Quellenlage nicht zu spezifizieren")
            label_series[missing_mask] = self._map_function_fallback(raw_fun[missing_mask])

        # ------------------------------------------------------------------
        # Ausgabe-DataFrame bauen
        # ------------------------------------------------------------------
        out = gdf.copy()
        out.rename(columns={id_col: "building_id"}, inplace=True)

        # LOD_Nutzung_*-Spalten
        out["LOD_Nutzung_original"] = code_series
        out["LOD_Nutzung_vereinheitlicht"] = label_series

        # function_label aus der vereinheitlichten Nutzung ableiten
        out["function_label"] = out["LOD_Nutzung_vereinheitlicht"]

        # WG/NWG/Unbekannt aus LOD_Nutzung_vereinheitlicht ableiten
        def _mk_nwg(lbl: str) -> str:
            if not isinstance(lbl, str):
                return "Unbekannt"
            if lbl == "Wohngebäude":
                return "WG"
            if lbl == "Nach Quellenlage nicht zu spezifizieren":
                return "Unbekannt"
            return "NWG"

        out["LOD_NWGoderWG"] = out["LOD_Nutzung_vereinheitlicht"].map(_mk_nwg)
        out["is_residential"]   = out["LOD_NWGoderWG"].eq("WG")
        out["is_func_unknown"]  = out["LOD_Nutzung_vereinheitlicht"].eq("Nach Quellenlage nicht zu spezifizieren")
        out["function_code"]    = out["LOD_Nutzung_original"]
        out["function_group"]   = out["LOD_Nutzung_vereinheitlicht"].map(
            lambda x: "wohngebaeude" if x == "Wohngebäude"
            else ("unbekannt" if x == "Nach Quellenlage nicht zu spezifizieren" else "nichtwohn_gebaeude")
        )
        out["lod"] = "LoD2"

        # gewünschte Originalfelder sauber exponieren
        # LOD_UNITID exakt gleich building_id
        out["LOD_UNITID"] = out["building_id"].astype(str)

        if "Dachform" in out.columns:
            out["LOD_Dachform"] = out["Dachform"]

        # Gebäudehöhe aus typischen Kandidaten ableiten
        h_col = self._find_first(out.columns, self._H_CANDS)
        if h_col:
            out["LOD_GebHoehe"] = pd.to_numeric(out[h_col], errors="coerce")
        else:
            out["LOD_GebHoehe"] = pd.NA

        # LOD_Stockwerke: wenn es eine Spalte "Stockwerke" gibt → numerisch, sonst komplett leer
        if "Stockwerke" in out.columns:
            out["LOD_Stockwerke"] = pd.to_numeric(out["Stockwerke"], errors="coerce")
        else:
            out["LOD_Stockwerke"] = pd.NA

        # 2D-Grundfläche aus Footprint (in m², da Ziel-CRS metrisch)
        out["LOD_Grundflaeche_m2"] = out["area_m2"]

        keep = [
            "building_id","geometry","__geom2d",
            fun_col if (fun_col and fun_col in out.columns) else None,
            name_col if (name_col and name_col in out.columns) else None,
            "LOD_Nutzung_original","LOD_Nutzung_vereinheitlicht","LOD_NWGoderWG",
            "function_label","is_residential","is_func_unknown",
            "function_code","function_group","lod","area_m2",
            "LOD_UNITID","LOD_Dachform","LOD_GebHoehe","LOD_Stockwerke","LOD_Grundflaeche_m2",
        ]
        keep = [c for c in keep if c]
        if "geometry" not in keep:
            keep.append("geometry")

        # Debug
        try:
            out.drop(columns="geometry", errors="ignore").head(50).to_csv(
                out_dir / "lod2_debug_head.csv", index=False, encoding="utf-8-sig"
            )
        except Exception:
            pass

        return out[keep]


# =============================================================================
# OSM – Overpass (wie in deinem Stand)
# =============================================================================

@register("osm")
class OSMSource(SourceConfig):
    """
    Datenquelle für OSM.
    """
    def __init__(self, overpass_url: str = None, timeout: int = 20, use_out_geom: bool = True, *args, **kwargs):
        super().__init__(name="osm", required=False)
        # 1) Endpoints: einzeln ODER Komma-Liste ODER Fallback-Defaults
        urls = (overpass_url or "").strip()
        if "," in urls:
            self.endpoints = [u.strip() for u in urls.split(",") if u.strip()]
        elif urls:
            self.endpoints = [urls]
        else:
            self.endpoints = [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter",
                "https://overpass.openstreetmap.ru/api/interpreter",
                "https://overpass.openstreetmap.fr/api/interpreter",
            ]
        self.timeout = int(timeout)
        self.use_out_geom = bool(use_out_geom)

        # HTTP-Session mit Retries
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def available(self, ctx: PipelineContext) -> bool:
        """
        Prüft die Verfügbarkeit der Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        return True  # wir haben immer Fallback-Endpoints

    def load(self, ctx: PipelineContext) -> gpd.GeoDataFrame:
        """
        Lädt OSM-Gebäudedaten für die Projektbbox und schreibt Debug-Ausgaben:
          - osm_columns.txt  (alle Spalten)
          - osm_debug_head.csv  (Head ohne Geometrie)
        """
        bbox = ctx.settings.region.bbox_25833
        if not bbox:
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{ctx.settings.target_epsg}")

        gdf = self.load_bbox(
            bbox,
            target_epsg=ctx.settings.target_epsg,
            fast=getattr(ctx.settings, "fast", False),
        )

        # Debug-Ausgaben wie bei LOD2/Basemap
        out_dir = Path(ctx.settings.out_dir) / "ap1"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Spaltenliste
            (out_dir / "osm_columns.txt").write_text(
                "\n".join(map(str, gdf.columns)),
                encoding="utf-8",
            )
        except Exception:
            pass

        try:
            # Head ohne Geometrie
            gdf.drop(columns=["geometry"], errors="ignore").head(50).to_csv(
                out_dir / "osm_debug_head.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception:
            pass

        return gdf

    def load_bbox(self, bbox: Tuple[float,float,float,float], target_epsg: int, fast: bool=False) -> gpd.GeoDataFrame:
        # 25833 -> 4326
        """
        Lädt die Quelle für die Projekt-BBOX.
        
        Args:
            bbox: Beschreibung.
            target_epsg: Beschreibung.
            fast: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        transformer = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        w, s = transformer.transform(minx, miny)
        e, n = transformer.transform(maxx, maxy)

        # Slice-Größe (in Grad) je nach "fast"
        deg = float(getattr(os, "environ", {}).get("KWP_OSM_DEG", "0.02"))
        if fast:
            deg = max(deg, 0.02)
        slices = self._slice_bbox((s, w, n, e), step=deg)

        mode_geom = self.use_out_geom
        all_polys = []
        all_recs  = []
        for i, (ss, ww, nn, ee) in enumerate(slices, 1):
            print(f"[osm] ({i}/{len(slices)}) slice ({ss}, {ww}, {nn}, {ee}) -> fetch (out_geom={mode_geom})")
            ok = False
            # Versuche nacheinander Endpunkte
            for j, url in enumerate(self.endpoints, 1):
                try:
                    recs, geoms = self._fetch_overpass(url, (ss, ww, nn, ee), out_geom=mode_geom)
                    all_recs.extend(recs)
                    all_polys.extend(geoms)
                    ok = True
                    break
                except requests.exceptions.ReadTimeout:
                    print(f"[osm]   Endpoint {j} Timeout → nächster Endpoint")
                except requests.exceptions.RequestException as ex:
                    print(f"[osm]   Endpoint {j} Fehler: {ex} → nächster Endpoint")
            if not ok and mode_geom:
                # einmaliger Fallback: in klassischem Modus ohne 'geom' erneut versuchen
                print("[osm] out geom lieferte keine Polygone → Fallback auf klassischen Modus")
                mode_geom = False
                for j, url in enumerate(self.endpoints, 1):
                    try:
                        recs, geoms = self._fetch_overpass(url, (ss, ww, nn, ee), out_geom=mode_geom)
                        all_recs.extend(recs)
                        all_polys.extend(geoms)
                        ok = True
                        break
                    except requests.exceptions.RequestException:
                        continue
            if not ok:
                print(f"[osm]   Slice {i} übersprungen (alle Endpunkte fehlgeschlagen)")
        if not all_polys:
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")
        gdf = gpd.GeoDataFrame(
            all_recs,
            geometry=gpd.GeoSeries(all_polys, crs="EPSG:4326"),
            crs="EPSG:4326"
        ).to_crs(epsg=target_epsg)

        # Ableitung
        def derive(row):
            """
            Leitet Zielattribute aus Quellfeldern ab.
            
            Args:
                row: Beschreibung.
            
            Returns:
                Beschreibung.
            """
            if pd.notna(row.get("building")):
                v = str(row.get("building")).lower()
                mapping = {
                    "residential":"Wohngebäude","apartments":"Wohngebäude","house":"Wohngebäude",
                    "detached":"Wohngebäude","semidetached_house":"Wohngebäude","terrace":"Wohngebäude",
                    "school":"Schule","kindergarten":"Schule","university":"Schule","college":"Schule",
                    "church":"Kirche","hospital":"Krankenhaus",
                    "office":"Gebäude für Wirtschaft oder Gewerbe","retail":"Gebäude für Wirtschaft oder Gewerbe",
                    "commercial":"Gebäude für Wirtschaft oder Gewerbe","industrial":"Gebäude für Wirtschaft oder Gewerbe",
                    "warehouse":"Gebäude für Wirtschaft oder Gewerbe","hotel":"Gebäude für Wirtschaft oder Gewerbe",
                    "train_station":"Bahnhofsgebäude","transportation":"Bahnhofsgebäude","parking":"Parkhaus",
                }
                if v in mapping:
                    lbl = mapping[v]
                    return lbl, (lbl == "Wohngebäude"), False, f"building={v}"
            for k in ["amenity","landuse"]:
                if pd.notna(row.get(k)):
                    v = str(row.get(k)).lower()
                    map2 = {
                        ("landuse","residential"): "Wohngebäude",
                        ("landuse","retail"): "Gebäude für Wirtschaft oder Gewerbe",
                        ("landuse","industrial"): "Gebäude für Wirtschaft oder Gewerbe",
                        ("landuse","commercial"): "Gebäude für Wirtschaft oder Gewerbe",
                        ("landuse","religious"): "Kirche",
                    }
                    if (k, v) in map2:
                        lbl = map2[(k, v)]
                        return lbl, (lbl == "Wohngebäude"), False, f"{k}={v}"
            raw = None
            for k in ["amenity","building","landuse","shop","office"]:
                if pd.notna(row.get(k)):
                    raw = f"{k}={row.get(k)}"; break
            return "Unbekannt", False, True, raw

        tmp = gdf.apply(derive, axis=1, result_type="expand")
        gdf["function_label"]  = tmp[0]
        gdf["is_residential"]  = tmp[1].astype("boolean")
        gdf["is_func_unknown"] = tmp[2].astype("boolean")
        gdf["function_raw"]    = tmp[3]
        gdf["function_code"]   = pd.NA
        gdf["function_group"]  = gdf["function_label"].map(
            lambda x: "wohngebaeude" if x == "Wohngebäude" else ("nichtwohn_gebaeude" if x != "Unbekannt" else "unbekannt")
        )
        gdf["lod"]         = "OSM"
        gdf["building_id"] = gdf.get("osm_id", pd.Series(range(len(gdf)), index=gdf.index)).astype(str)
        gdf["levels"]      = pd.to_numeric(gdf.get("levels"), errors="coerce")
        keep = [
            "building_id","geometry",
            "function_label","is_residential","is_func_unknown","function_raw",
            "function_code","function_group","lod",
            "building","amenity","landuse","name","levels","osm_id","historic",
        ]
        keep = [c for c in keep if c in gdf.columns] + (["geometry"] if "geometry" not in keep else [])
        return gdf[keep]

    def _slice_bbox(self, bbox_ll: Tuple[float,float,float,float], step: float=0.02) -> List[Tuple[float,float,float,float]]:
        s, w, n, e = bbox_ll
        if step <= 0:
            return [(s, w, n, e)]
        out = []
        lat = s
        while lat < n:
            lat2 = min(lat + step, n)
            lon = w
            while lon < e:
                lon2 = min(lon + step, e)
                out.append((lat, lon, lat2, lon2))
                lon = lon2
            lat = lat2
        return out

    def _fetch_overpass(self, url: str, bbox_ll: Tuple[float,float,float,float], out_geom: bool = True) -> Tuple[List[dict], List[Polygon]]:
        ss, ww, nn, ee = bbox_ll
        if out_geom:
            q = f"""
            [out:json][timeout:{self.timeout}];
            (
              way["building"]({ss},{ww},{nn},{ee});
            );
            out tags geom;
            """.strip()
        else:
            q = f"""
            [out:json][timeout:{self.timeout}];
            (
              way["building"]({ss},{ww},{nn},{ee});
            );
            out body; >; out skel qt;
            """.strip()

        r = self.session.post(url, data={"data": q}, timeout=self.timeout+5)
        r.raise_for_status()
        data = r.json()

        geoms: List[Polygon] = []
        recs: List[dict] = []

        if out_geom:
            for el in data.get("elements", []):
                if el.get("type") != "way":
                    continue
                coords = [(pt["lon"], pt["lat"]) for pt in el.get("geometry", [])]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    poly = Polygon(coords)
                    if not poly.is_valid or poly.is_empty:
                        continue

                    t = el.get("tags", {}) or {}
                    # Level-/Höhen-Tags harmonisieren
                    levels = t.get("building:levels") or t.get("levels")
                    height = t.get("building:height") or t.get("height")

                    geoms.append(poly)
                    recs.append({
                        "osm_id": f"way/{el['id']}",
                        "building": t.get("building"),
                        "amenity": t.get("amenity"),
                        "landuse": t.get("landuse"),
                        "name": t.get("name"),
                        "levels": levels,
                        "height": height,
                        "start_date": t.get("start_date"),
                        "year_built": t.get("year_built"),
                        "historic": t.get("historic"),
                        "source": "overpass",
                    })
        else:
            nodes = {
                el["id"]: (el.get("lon"), el.get("lat"))
                for el in data.get("elements", [])
                if el.get("type") == "node"
            }
            for el in data.get("elements", []):
                if el.get("type") != "way":
                    continue
                node_ids = el.get("nodes") or []
                coords = [nodes.get(nid) for nid in node_ids]
                if any(c is None for c in coords):
                    continue
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    poly = Polygon(coords)
                    if not poly.is_valid or poly.is_empty:
                        continue

                    t = el.get("tags", {}) or {}
                    levels = t.get("building:levels") or t.get("levels")
                    height = t.get("building:height") or t.get("height")

                    geoms.append(poly)
                    recs.append({
                        "osm_id": f"way/{el['id']}",
                        "building": t.get("building"),
                        "amenity": t.get("amenity"),
                        "landuse": t.get("landuse"),
                        "name": t.get("name"),
                        "levels": levels,
                        "height": height,
                        "start_date": t.get("start_date"),
                        "year_built": t.get("year_built"),
                        "historic": t.get("historic"),
                        "source": "overpass",
                    })

        return recs, geoms

# =============================================================================
# Basemap.de – Vektor-Tiles (V2)  (mit Nutzungs-Polygone)
# =============================================================================

@dataclass
class BasemapCfg:
    """
    Konfiguration für Basemap.
    """
    mvt_url_template: str
    headers: Optional[Dict[str, str]] = None

@register("basemap")
class BasemapContextSource(SourceConfig):
    """
    Basemap-Quelle auf Basis des Pipeline-Contexts.
    """
    def __init__(self, cfg: BasemapCfg, zoom: int = 15, max_tiles: int | None = None, include_pois: bool = False, *args, **kwargs):
        super().__init__(name="basemap", required=False)
        self.cfg = cfg
        self.zoom = int(zoom)
        self.max_tiles = max_tiles if (max_tiles is None or int(max_tiles) > 0) else None
        self.include_pois = bool(include_pois)

    def available(self, ctx: PipelineContext) -> bool:
        """
        Prüft die Verfügbarkeit der Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        return bool(getattr(ctx.settings, "basemap", None)) or bool(getattr(ctx.settings, "basemap_mvt_template", None))

    # --------- main ----------
    def load_basemap(self, bbox_25833: Tuple[float, float, float, float], to_crs: int):
        """
        Rückgabe:
          - buildings_gdf: Gebäude (Layer 'Gebaeudeflaeche') im Ziel-CRS
          - context_gdf:   Mix aus
              * 'Siedlungsflaeche' (POLYGON, mit Attribut 'klasse')
              * 'Weitere_Nutzung_Flaeche' (POLYGON)
              * 'Gebaeudepunkt' (POINT, mit Attribut 'klasse')
            Alle mit zusätzlicher Spalte 'bm_layer', die den Ursprungs-Layer-Namen trägt.
          - meta:          Metadaten

        Damit können in der AP1-Pipeline je LOD2-Gebäude über räumliche Beziehungen
        sowohl die 'Gebaeudeflaeche.klasse' als auch 'Siedlungsflaeche.klasse' und
        'Gebaeudepunkt.klasse' ausgewertet werden.
        """
        if not bbox_25833:
            empty = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{to_crs}")
            return empty, empty, {"zoom": self.zoom, "tile_count": 0, "source": "basemap.de MVT", "target_epsg": to_crs}

        # 25833 -> 3857 (nur für spätere Geometrie-Umrechnung)
        tr_3857 = Transformer.from_crs("EPSG:25833", "EPSG:3857", always_xy=True)
        minx, miny, maxx, maxy = bbox_25833
        xmin, ymin = tr_3857.transform(minx, miny)
        xmax, ymax = tr_3857.transform(maxx, maxy)
        bbox_3857 = (xmin, ymin, xmax, ymax)

        # 25833 -> 4326 (WGS84) **NUR** für Tile-Indexierung
        tr_ll = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
        w_min, s_min = tr_ll.transform(minx, miny)
        w_max, s_max = tr_ll.transform(maxx, maxy)
        bbox_ll = (w_min, s_min, w_max, s_max)

        # Jetzt korrekt: mercantile erwartet Lon/Lat
        tiles = self._tiles_for_bbox_ll(bbox_ll, z=self.zoom)

        # optional begrenzen
        if self.max_tiles:
            tiles = self._take_centered_tiles(tiles, bbox_3857, self.max_tiles)

        print(f"[basemap] Tiles: {len(tiles)} (zoom={self.zoom}, include_pois={self.include_pois})")

        # Sammel-Listen
        b_feats: List[dict] = []     # Gebäude (Gebaeudeflaeche)
        u_feats: List[dict] = []     # Nutzung + Punkte (Siedlungsflaeche, Weitere_Nutzung_Flaeche, Gebaeudepunkt)
        # p_feats: List[dict] = []   # weitere POIs (Name_Punkt, Bauwerkspunkt, Adresse …) – aktuell nicht benötigt

        for i, t in enumerate(tiles, 1):
            if i == 1 or (i % 3 == 0):
                print(f"[basemap] hole Tile {i}/{len(tiles)} {t}")
            content = self._fetch_mvt(self.cfg.mvt_url_template, t, headers=self.cfg.headers)
            if not content:
                continue
            layers = self._decode_mvt(content)

            # Gebäude-Polygone
            if "Gebaeudeflaeche" in layers:
                b_feats.extend(self._mvt_features_to_geo(layers["Gebaeudeflaeche"], 3857, t, layer_name="Gebaeudeflaeche"))

            # Nutzungspolygone
            if "Siedlungsflaeche" in layers:
                feats = self._mvt_features_to_geo(layers["Siedlungsflaeche"], 3857, t, layer_name="Siedlungsflaeche")
                u_feats.extend(feats)
            if "Weitere_Nutzung_Flaeche" in layers:
                feats = self._mvt_features_to_geo(layers["Weitere_Nutzung_Flaeche"], 3857, t, layer_name="Weitere_Nutzung_Flaeche")
                u_feats.extend(feats)

            # Gebäudepunkte (wichtige Klasse, insb. Baudenkmal, etc.) – IMMER aufnehmen
            if "Gebaeudepunkt" in layers:
                feats = self._mvt_features_to_geo(layers["Gebaeudepunkt"], 3857, t, layer_name="Gebaeudepunkt")
                u_feats.extend(feats)

            # Optional: weitere POIs (z.B. Adresse, Name_Punkt …) – bei Bedarf aktivierbar
            if self.include_pois:
                for lname in ("Name_Punkt", "Bauwerkspunkt", "Besonderer_Punkt", "Adresse"):
                    if lname in layers:
                        # Wir hängen sie ebenfalls an u_feats an, so dass sie in der zweiten
                        # GeoDataFrame sichtbar sind (inkl. bm_layer)
                        feats = self._mvt_features_to_geo(layers[lname], 3857, t, layer_name=lname)
                        u_feats.extend(feats)

        # in Ziel-CRS
        b_gdf = self._to_target_crs(b_feats, src_epsg=3857, dst_epsg=int(CRS.from_user_input(to_crs).to_epsg()))
        u_gdf = self._to_target_crs(u_feats, src_epsg=3857, dst_epsg=int(CRS.from_user_input(to_crs).to_epsg()))

        # Gebäude filtern & Fläche
        if not b_gdf.empty:
            b_gdf = b_gdf[b_gdf.geometry.type.isin(["Polygon","MultiPolygon"])].copy()
            b_gdf["area_m2"] = b_gdf.geometry.area
            b_gdf = b_gdf[b_gdf["area_m2"] > 10.0]

        meta = {"zoom": self.zoom, "tile_count": len(tiles), "source": "basemap.de MVT", "target_epsg": to_crs}
        return b_gdf.reset_index(drop=True), u_gdf.reset_index(drop=True), meta

        # TODO A(Basemap):
        #  - Sicherstellen, dass aus den folgenden Basemap-Layern die relevanten Attribute
        #    im Output an AP1 übergeben werden:
        #      * Gebaeudeflaeche: klasse, art, hoehe, name
        #      * Siedlungsflaeche / Weitere_Nutzung_Flaeche: klasse
        #      * Bauwerkspunkt / Gebaeudepunkt: klasse (insb. zur Kennzeichnung "Baudenkmal")
        #      * Adresse: strasse, hausnummer (Adressinformationen)
        #  - Diese Properties werden zwar in den GeoDataFrames vorhanden sein, müssen aber
        #    in ap1_pipeline explizit auf LOD2-Gebäude gemappt und als BMAP_* Spalten
        #    (z.B. BMAP_Strasse, BMAP_Hausnummer, BMAP_DenkmalFlag) ins Endschema übernommen werden.

    # --------- helpers ----------
    @staticmethod
    def _fetch_mvt(tmpl: str, tile, headers: Optional[Dict[str,str]] = None) -> bytes:
        z, x, y = tile[2], tile[0], tile[1]
        url = tmpl.format(z=z, x=x, y=y)
        try:
            r = requests.get(url, headers=headers or {}, timeout=10)
            if r.status_code == 404:
                return b""
            r.raise_for_status()
            content = r.content
            return gzip.decompress(content) if content[:2] == b"\x1f\x8b" else content
        except Exception:
            return b""

    @staticmethod
    def _decode_mvt(content: bytes) -> Dict[str, Any]:
        import mapbox_vector_tile
        return mapbox_vector_tile.decode(content)

    def _tiles_for_bbox(self, bbox_3857: Tuple[float,float,float,float], z: int) -> List[Tuple[int,int,int]]:
        from mercantile import tiles
        minx, miny, maxx, maxy = bbox_3857
        return [(t.x, t.y, t.z) for t in tiles(minx, miny, maxx, maxy, [z])]

    def _tiles_for_bbox_ll(self, bbox_ll: tuple[float, float, float, float], z: int) -> list[tuple[int, int, int]]:
        # bbox_ll: (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
        from mercantile import tiles
        lon_min, lat_min, lon_max, lat_max = bbox_ll
        return [(t.x, t.y, t.z) for t in tiles(lon_min, lat_min, lon_max, lat_max, [z])]

    def _tile_bounds_3857(self, x: int, y: int, z: int) -> Tuple[float,float,float,float]:
        from mercantile import xy_bounds
        b = xy_bounds(x, y, z)
        return (b.left, b.bottom, b.right, b.top)

    def _take_centered_tiles(self, tiles: List[Tuple[int,int,int]], bbox_3857, k: int) -> List[Tuple[int,int,int]]:
        (minx, miny, maxx, maxy) = bbox_3857
        cx, cy = (minx+maxx)/2.0, (miny+maxy)/2.0
        def center(t):
            """
            Berechnet den Mittelpunkt.
            
            Args:
                t: Beschreibung.
            
            Returns:
                Beschreibung.
            """
            x, y, z = t
            bx0, by0, bx1, by1 = self._tile_bounds_3857(x,y,z)
            return ((bx0+bx1)/2.0, (by0+by1)/2.0)
        tiles_sorted = sorted(tiles, key=lambda t: (center(t)[0]-cx)**2 + (center(t)[1]-cy)**2)
        return tiles_sorted[:k]

    def _mvt_features_to_geo(self, layer: dict, epsg: int, tile, layer_name: Optional[str] = None) -> List[dict]:
        """
        Wandelt einen MVT-Layer in Feature-Dictionaries um und fügt optional
        'bm_layer' hinzu, um den Ursprungs-Layernamen (z.B. 'Gebaeudeflaeche',
        'Siedlungsflaeche', 'Gebaeudepunkt') mitzugeben.
        """
        extent = layer.get("extent", 4096)
        feats: List[dict] = []
        for feat in layer.get("features", []):
            geom = feat.get("geometry")
            props = feat.get("properties") or {}
            if layer_name is not None:
                props = {**props, "bm_layer": layer_name}
            if geom is None:
                continue
            shp = self._mvt_geom_to_shapely(geom, extent, tile)
            if shp is None or shp.is_empty:
                continue
            feats.append({"geometry": shp, **props})
        return feats

    @staticmethod
    def _mvt_geom_to_shapely(geom, extent: int, tile):
        from shapely.geometry import shape as to_shape
        from shapely import affinity
        from mercantile import xy_bounds
        z, x, y = tile[2], tile[0], tile[1]
        bounds = xy_bounds(x, y, z)
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
        sx = (maxx - minx) / float(extent)
        sy = (maxy - miny) / float(extent)
        g = to_shape(geom)
        g = affinity.scale(g, xfact=sx, yfact=sy, origin=(0.0, 0.0))
        g = affinity.translate(g, xoff=minx, yoff=miny)
        return g

    @staticmethod
    def _to_target_crs(feats: list[dict], src_epsg: int, dst_epsg: int) -> gpd.GeoDataFrame:
        if not feats:
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{dst_epsg}")
        df = pd.DataFrame(feats)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=f"EPSG:{src_epsg}")
        if gdf.crs is None:
            gdf.set_crs(epsg=src_epsg, inplace=True)
        gdf = gdf.to_crs(epsg=dst_epsg)
        return gdf

from urllib.parse import urlencode  # sicherstellen, dass dieser Import oben vorhanden ist

@register("divis")
class DivisWmsSource(SourceConfig):
    """
    DIVIS-WMS-Quelle (LfD Sachsen) auf Basis von GetFeatureInfo.

    - Verwendet BBOX in EPSG:25833 (entweder aus ctx.settings.region.bbox_25833
      oder aus einem explizit übergebenen bbox_25833-Argument im Konstruktor).
    - Fragt über mehrere Pixelpositionen pro BBOX GetFeatureInfo mit
      INFO_FORMAT=application/geo+json ab und sammelt die Features.
    - Nutzt die Layer L1, L2, L3 (Flächen, Linien, Punkte), weil die Denkmale
      verteilt auf alle drei Geometrietypen vorliegen.
    - Ergebnis: GeoDataFrame mit Kulturdenkmal-Features im Ziel-CRS
      (ctx.settings.target_epsg).

    Zusätzlich:
    - Schreibt einen Roh-Export nach <out_dir>/divis:
      * divis_wms_full.gpkg  (Layer: 'divis_wms')
      * divis_wms_head.csv   (nur Attribut-Head, ohne Geometrie)
    """

    BASE_URL = "https://cardomap3.idu.de/lfds/public/ogc.ashx"

    COMMON_PARAMS = {
        "PKGID": 5,
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        # Vektor-Layer:
        # L1 = Kulturdenkmale (Flächen)
        # L2 = Kulturdenkmale (Linien)
        # L3 = Kulturdenkmale (Punkte)
        "LAYERS": "L1,L2,L3",
        "QUERY_LAYERS": "L1,L2,L3",
        "STYLES": "",
        "CRS": "EPSG:25833",
        "INFO_FORMAT": "application/geo+json",
    }

    def __init__(
        self,
        bbox_25833: tuple[float, float, float, float] | None = None,
        width: int = 2048,
        height: int = 2048,
        *args,
        **kwargs,
    ):
        # required=False, damit Pipeline nicht hart fehlschlägt
        super().__init__(name="divis", required=False)
        self.bbox_25833 = bbox_25833
        self.width = int(width)
        self.height = int(height)

        # HTTP-Session mit Retries (analog OSM/Basemap)
        self.session = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # ------------------------------------------------------------------ helpers

    def _resolve_bbox(self, ctx: PipelineContext) -> Optional[tuple[float, float, float, float]]:
        """
        Ermittelt die BBOX (EPSG:25833), entweder aus dem Konstruktor
        oder aus ctx.settings.region.bbox_25833.
        """
        if self.bbox_25833 is not None:
            return self.bbox_25833
        region = getattr(ctx.settings, "region", None)
        if region is None:
            return None
        return getattr(region, "bbox_25833", None)

    def _sample_pixels(self, step_px: int = 2) -> list[tuple[int, int]]:
        """
        Erzeugt eine Liste von Pixelpositionen (I,J) innerhalb des Bildes
        (WIDTH x HEIGHT), an denen GetFeatureInfo-Abfragen gestellt werden.

        Mit width=height=1024 und step_px=64 ergibt das ca. 16x16 = 256 Punkte,
        was das Gebiet relativ dicht abtastet.
        """
        w, h = self.width, self.height
        is_ = list(range(step_px // 2, w, step_px))
        js_ = list(range(step_px // 2, h, step_px))
        return [(i, j) for i in is_ for j in js_]

    def _getfeatureinfo_geojson(
        self,
        bbox: tuple[float, float, float, float],
        i: int,
        j: int,
    ) -> Optional[dict]:
        """
        Holt eine GetFeatureInfo-Antwort als GeoJSON für eine BBOX und eine Pixelposition (i,j).
        """
        params = self.COMMON_PARAMS.copy()
        params.update({
            "REQUEST": "GetFeatureInfo",
            "BBOX": ",".join(map(str, bbox)),
            "WIDTH": self.width,
            "HEIGHT": self.height,
            "I": i,
            "J": j,
            # Anzahl gewünschter Features pro Anfrage erhöhen
            "FEATURE_COUNT": 1000,
            "feature_count": 1000,
        })
        url = f"{self.BASE_URL}->{urlencode(params)}"

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[divis] GetFeatureInfo-Fehler ({i},{j}): {exc}")
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

    # ------------------------------------------------------------------ Source-API

    def available(self, ctx: PipelineContext) -> bool:
        """
        Prüft die Verfügbarkeit der Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        return self._resolve_bbox(ctx) is not None

    def load(self, ctx: PipelineContext) -> gpd.GeoDataFrame:
        """
        Führt die WMS-Abtastung durch und gibt ein GeoDataFrame mit allen
        gefundenen Kulturdenkmal-Features zurück. Zusätzlich wird ein
        Roh-Export nach <out_dir>/divis geschrieben.
        """
        bbox = self._resolve_bbox(ctx)
        if not bbox:
            print("[divis] Keine BBOX verfügbar – leeres GeoDataFrame.")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{ctx.settings.target_epsg}")

        print(f"[divis] Verwende BBOX (EPSG:25833): {bbox}")

        all_features: list[dict] = []
        for (i, j) in self._sample_pixels(step_px=64):
            gj = self._getfeatureinfo_geojson(bbox, i=i, j=j)
            if gj is None:
                continue
            feats = gj.get("features", [])
            all_features.extend(feats)

        if not all_features:
            print("[divis] Keine Features aus WMS erhalten.")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{ctx.settings.target_epsg}")

        # GeoJSON → GeoDataFrame inkl. Geometrie
        gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:25833")

        # Duplikate nach ID-Feld herausfiltern, falls vorhanden
        id_candidates = ["id", "OB_ID", "OBJEKTID", "LFDNR"]
        id_col = next((c for c in id_candidates if c in gdf.columns), None)
        if id_col:
            before = len(gdf)
            gdf = gdf.drop_duplicates(subset=[id_col])
            after = len(gdf)
            print(f"[divis] Drop duplicates nach {id_col}: {before} → {after}")

        # In Ziel-CRS transformieren
        target_epsg = int(getattr(ctx.settings, "target_epsg", 25833))
        if gdf.crs is None:
            gdf.set_crs(epsg=25833, inplace=True)
        if gdf.crs.to_epsg() != target_epsg:
            gdf = gdf.to_crs(epsg=target_epsg)

        # Bounding Box im Ziel-CRS ausgeben (zur Kontrolle gegen Gebäude-BBOX)
        tb = gdf.total_bounds  # (minx, miny, maxx, maxy)
        print(f"[divis] WMS-Ergebnis BBOX in EPSG:{target_epsg}: {tb}")
        print(f"[divis] Anzahl DIVIS-Features (nach Duplikatfilter): {len(gdf)}")

        # ------------------------------------------------------------------
        # Roh-Export nach <out_dir>/divis
        # ------------------------------------------------------------------
        out_base = Path(getattr(ctx.settings, "out_dir", "out"))
        divis_dir = out_base / "divis"
        divis_dir.mkdir(parents=True, exist_ok=True)

        full_gpkg = divis_dir / "divis_wms_full.gpkg"
        head_csv = divis_dir / "divis_wms_head.csv"

        # Versuche, das komplette GPKG zu schreiben
        try:
            gdf.to_file(full_gpkg, layer="divis_wms", driver="GPKG")
            print(f"[divis] Roh-GPKG nach {full_gpkg} geschrieben ({len(gdf)} Features).")
        except Exception as exc:
            print(f"[divis] WARN: Konnte Roh-GPKG {full_gpkg} nicht schreiben: {exc}")

        # Zusätzlich: Head der Attribute (ohne Geometrie) als CSV
        try:
            gdf.drop(columns=["geometry"], errors="ignore").head(200).to_csv(
                head_csv, index=False, encoding="utf-8-sig"
            )
            print(f"[divis] Attribut-Head nach {head_csv} geschrieben.")
        except Exception as exc:
            print(f"[divis] WARN: Konnte Head-CSV {head_csv} nicht schreiben: {exc}")

        return gdf

@register("zensus_100m")
class ZensusGridSource(SourceConfig):
    """
    Zensus 2022 – 100-m-Gitter (Deutschland) als zusätzliche Quelle.

    Lädt für die aktuelle BBOX (EPSG:25833) die 100-m-Gitterzellen aus dem
    bundesweiten Zensus-Grid-FeatureService und speichert sie als GeoPackage
    in data/Zensus2022/zensus_100m.gpkg.

    Hinweise:
    - Der verwendete Dienst 'Zensus2022_grid_final' enthält u.a.:
      * Gebäude nach Baujahresklassen
      * Wohnungen nach Energieträger
      * Wohnungen nach Heizungsart
      * Eigentümerquote, Leerstandsquote
      * Anteil der ab 65-Jährigen usw.
    - Die eigentliche Ableitung deiner Zielgrößen (z.B. ZENSUS_Baujahr_Klasse,
      ZENSUS_Eigentumsquote, ZENSUS_Leerstandsquote, ZENSUS_Anteil_65plus)
      erfolgt später in building_typing.py.
    """

    # ArcGIS FeatureServer (100m/1km/10km Grid, ganz Deutschland)
    _DEFAULT_FEATURE_URL = (
        "https://services2.arcgis.com/jUpNdisbWqRpMo35/arcgis/rest/services/"
        "Zensus2022_grid_final/FeatureServer/0/query"
    )

    def __init__(
        self,
        feature_url: str | None = None,
        crs_buildings: str | int = 25833,
        *args,
        **kwargs,
    ):
        super().__init__(name="zensus_100m", required=False)
        self.feature_url = feature_url or self._DEFAULT_FEATURE_URL
        # CRS der Gebäude-/Projektgeometrien (LoD2, Basemap, OSM etc.)
        self.crs_buildings = crs_buildings

    # ------------------------------------------------------------------ #
    # Hilfsfunktionen für Kontext / Pfade
    # ------------------------------------------------------------------ #

    def _get_bbox_from_context(self, context) -> tuple[float, float, float, float]:
        """
        Versucht, die BBOX der Pipeline aus dem Kontext zu holen.

        Erwartet EPSG:25833 (wie im restlichen Projekt).

        Unterstützte Varianten:
        - context.bbox = (minx, miny, maxx, maxy)
        - context.settings.region.bbox_25833 (wie im aktuellen CLI umgesetzt)
        """
        # 1) Klassischer Fall: direkte BBOX am Context
        bbox = getattr(context, "bbox", None)

        # 2) Aktueller CLI-Fall: BBOX in settings.region.bbox_25833
        if bbox is None:
            settings = getattr(context, "settings", None)
            if settings is not None:
                region = getattr(settings, "region", None)
                if region is not None:
                    bbox = getattr(region, "bbox_25833", None)

        # 3) Wenn immer noch nichts gefunden wurde → sauberer Fehler
        if bbox is None:
            raise RuntimeError(
                "ZensusGridSource benötigt eine BBOX im PipelineContext "
                "(z.B. context.bbox oder settings.region.bbox_25833 in EPSG:25833)."
            )

        if len(bbox) != 4:
            raise ValueError(f"Unerwartetes BBOX-Format: {bbox!r}")

        # Sicherstellen, dass es wirklich floats sind
        return tuple(float(v) for v in bbox)


    def _get_out_dir(self, context) -> Path:
        """
        Liefert den Ausgabeordner .../data/Zensus2022.
        """
        base = getattr(context, "workdir", None)
        base = Path(base) if base is not None else Path(".")
        out_dir = base / rel_zensus2022_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ------------------------------------------------------------------ #
    # Datenabruf
    # ------------------------------------------------------------------ #

    def _bbox_25833_to_3857(self, bbox_25833) -> tuple[float, float, float, float]:
        """
        Transformiert die BBOX von EPSG:25833 nach EPSG:3857 (Web Mercator),
        da der Zensus-Grid-Dienst 'Zensus2022_grid_final' in 102100/3857 arbeitet.
        """
        gdf_bbox = gpd.GeoDataFrame(
            index=[0],
            geometry=[shp_box(*bbox_25833)],
            crs=f"EPSG:{self.crs_buildings}",  # i.d.R. 25833
        )
        gdf_3857 = gdf_bbox.to_crs(3857)
        minx, miny, maxx, maxy = gdf_3857.total_bounds
        return float(minx), float(miny), float(maxx), float(maxy)

    def _fetch_grid_for_bbox(self, bbox_25833):
        # 1) BBOX 25833 -> 3857 transformieren (nur für Anfrage)
        minx, miny, maxx, maxy = self._bbox_25833_to_3857(bbox_25833)

        # ArcGIS: Pagination über resultOffset/resultRecordCount
        page_size = 2000  # konservativ; 2000 funktioniert in der Regel zuverlässig
        offset = 0
        all_features = []

        base_params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "3857",
            "spatialRel": "esriSpatialRelIntersects",
            "f": "geojson",
            "resultRecordCount": page_size,
        }

        while True:
            params = dict(base_params)
            params["resultOffset"] = offset

            resp = requests.get(self.feature_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            feats = data.get("features") or []
            if not feats:
                break

            all_features.extend(feats)

            # Wenn weniger als page_size zurückkommt, sind wir fertig
            if len(feats) < page_size:
                break

            offset += page_size

        # ArcGIS-FeatureServer gibt GeoJSON i.d.R. in EPSG:4326 zurück
        if not all_features:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(self.crs_buildings)

        gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
        gdf = gdf.to_crs(self.crs_buildings)
        return gdf

    # ------------------------------------------------------------------ #
    # Öffentliche API für die Pipeline
    # ------------------------------------------------------------------ #

    def load(self, context) -> gpd.GeoDataFrame:
        """
        Haupt-Einstiegspunkt für die Pipeline.

        - liest BBOX aus dem Kontext,
        - lädt die passenden Zensus-Gitterzellen,
        - speichert sie als GeoPackage unter data/Zensus2022/zensus_100m.gpkg,
        - gibt das GeoDataFrame zurück.
        """
        bbox = self._get_bbox_from_context(context)
        gdf = self._fetch_grid_for_bbox(bbox)

        out_dir = self._get_out_dir(context)
        out_path = out_dir / "zensus_100m.gpkg"

        if not gdf.empty:
            gdf.to_file(out_path, driver="GPKG")
            logging.info(
                "[zensus] %d Zellen nach %s geschrieben.", len(gdf), out_path
            )
        else:
            logging.info(
                "[zensus] Keine Daten geschrieben (GeoDataFrame leer)."
            )

        return gdf

class GhsObatSource(SourceConfig):
    """
    Quelle für den Datensatz:
      GHS-OBAT_GLOBE_R2024A (hier: Deutschland / Sachsen-Ausschnitt, GPKG)

    Datensatzcharakteristik (laut Spezifikation 3.1.1):

    - Geometrie:
        * Im GPKG ist die Geometrie ein Punkt (Point),
          der den Zentroid des Gebäude-Footprints repräsentiert.
        * CRS: WGS84 (EPSG:4326). Wir reprojizieren intern nach EPSG:25833.

    - Attribute (relevante Auswahl):

        geometry (Point)
            Zentroid des Overture-Gebäudefootprints.

        id : string
            32-stellige HEX-ID (GERS-ID) – stabiler Identifier zur Verknüpfung
            mit der Overture-Geometrie.

        country : string
            ISO 3166-1 alpha-3 Ländercode (z. B. "DEU").

        adm1 : string
            Name der administrativen Einheit (Level 1), z. B. Bundesland.

        height : float (m)
            Mittlere Gebäudehöhe (ANBH, GHS-BUILT-H), in Metern, mind. 2.5 m.

        shapefactor : float (m2/m3)
            Formfaktor S/V (Oberfläche / Volumen), in m²/m³.

        use : int (0–2)
            Funktionale Nutzung:
                0 = außerhalb des "built-up domain"
                1 = "residential"
                2 = "non-residential"

        epoch : int (0–5)
            Baualters-Epoche basierend auf GHS-AGE/GHS-BUILT-S:
                0 = außerhalb des built-up domain
                1 = vor 1980
                2 = 1980–1990
                3 = 1990–2000
                4 = 2000–2010
                5 = 2010–2020

        area : float (m2)
            Grundfläche des Overture-Buildings (m², lokale UTM).

        perimeter : float (m)
            Umfang des Overture-Buildings (m, lokale UTM).

    Erwartete Datei:
      <workdir>/data/Baujahre_OBAT/ghs_obat_gpkg_deu_e2020_r2024a_v1_0__ghs_obat_e2020_deu_sax_r2024a_v1_0.gpkg

    Verhalten:
      - available():
          Prüft nur, ob die GPKG-Datei existiert.
      - load():
          * liest die GPKG,
          * setzt CRS (WGS84) und transformiert nach target_epsg (standard 25833),
          * optional Clip auf Projekt-BBOX (EPSG:25833),
          * reduziert auf die relevanten Attribute,
          * schreibt einen Clip in out/ghs_obat/ghs_obat_clip.gpkg,
          * gibt den gecroppten GeoDataFrame zurück.
    """

    DEFAULT_FILENAME = (
        "ghs_obat_gpkg_deu_e2020_r2024a_v1_0__"
        "ghs_obat_e2020_deu_sax_r2024a_v1_0.gpkg"
    )

    def __init__(self, path: str | Path | None = None, *args, **kwargs):
        super().__init__(name="ghs_obat", required=False)
        self._path = Path(path) if path is not None else None

    # -------------------- Hilfen für Pfade/BBOX --------------------

    def _resolve_path(self, ctx: PipelineContext) -> Path:
        """
        Ermittelt den Pfad zur GHS-OBAT-GPKG-Datei.

        Priorität:
        1) explizit im Konstruktor gesetzter Pfad
        2) settings.data.ghs_obat_path
        3) <workdir>/data/Baujahre_OBAT/<DEFAULT_FILENAME>
        """
        if self._path is not None:
            return self._path

        settings = getattr(ctx, "settings", ctx)
        data_cfg = getattr(settings, "data", settings)
        cfg_path = getattr(data_cfg, "ghs_obat_path", None)
        if cfg_path:
            return Path(cfg_path)

        # Fallback: Projektarbeitsverzeichnis + data/Baujahre_OBAT
        workdir = Path(getattr(ctx, "workdir", "."))
        return workdir / "data" / DIR_BAUJAHRE_OBAT / self.DEFAULT_FILENAME

    def _get_bbox_from_context(self, ctx: PipelineContext) -> Optional[tuple[float, float, float, float]]:
        """
        BBOX (EPSG:25833) aus dem Context lesen.

        Unterstützt:
        - ctx.bbox
        - ctx.settings.region.bbox_25833
        """
        bbox = getattr(ctx, "bbox", None)
        if bbox is None:
            settings = getattr(ctx, "settings", None)
            if settings is not None:
                region = getattr(settings, "region", None)
                if region is not None:
                    bbox = getattr(region, "bbox_25833", None)

        if bbox is None:
            return None

        if len(bbox) != 4:
            raise ValueError(f"GHS-OBAT: Unerwartetes BBOX-Format: {bbox!r}")

        return tuple(float(v) for v in bbox)

    # -------------------- Source-API --------------------

    def available(self, ctx: PipelineContext) -> bool:
        """
        Prüft die Verfügbarkeit der Quelle.
        
        Args:
            ctx: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        p = self._resolve_path(ctx)
        return p.exists()

    def load(self, ctx: PipelineContext) -> gpd.GeoDataFrame:
        """
        Lädt die GHS-OBAT-GPKG-Datei, bringt sie ins Ziel-CRS,
        clippt auf die Projekt-BBOX und speichert das Ergebnis
        als out/ghs_obat/ghs_obat_clip.gpkg.
        """
        settings = getattr(ctx, "settings", ctx)
        target_epsg = int(getattr(settings, "target_epsg", getattr(ctx, "target_epsg", 25833)))
        out_base = Path(getattr(settings, "out_dir", getattr(ctx, "out_dir", "out")))
        out_dir = out_base / "ghs_obat"
        out_dir.mkdir(parents=True, exist_ok=True)

        clip_path = out_dir / "ghs_obat_clip.gpkg"

        # Wenn der Clip bereits existiert, nutzen wir ihn direkt
        if clip_path.exists():
            try:
                gdf_clip = gpd.read_file(clip_path)
                # ggf. CRS noch harmonisieren
                if gdf_clip.crs is None:
                    gdf_clip.set_crs(epsg=target_epsg, inplace=True)
                elif gdf_clip.crs.to_epsg() != target_epsg:
                    gdf_clip = gdf_clip.to_crs(epsg=target_epsg)
                print(f"[GHS-OBAT] Verwende vorhandenen Clip: {clip_path}")
                return gdf_clip
            except Exception as exc:
                print(f"[GHS-OBAT] WARN: Konnte bestehenden Clip {clip_path} nicht lesen: {exc}")

        # Sonst: Original lesen und clippen
        src_path = self._resolve_path(ctx)
        if not src_path.exists():
            print(f"[GHS-OBAT] Datei nicht gefunden: {src_path}")
            return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

        print(f"[GHS-OBAT] Lade Original-GPKG: {src_path}")
        gdf = gpd.read_file(src_path)

        # CRS setzen/transformieren (laut Spezifikation: WGS84)
        if gdf.crs is None:
            # laut Doku: WGS84 (EPSG:4326) – Geometrie = Zentroid-Point
            gdf.set_crs(epsg=4326, inplace=True)
        try:
            if gdf.crs.to_epsg() != target_epsg:
                gdf = gdf.to_crs(epsg=target_epsg)
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: CRS-Transformation fehlgeschlagen ({gdf.crs} -> EPSG:{target_epsg}): {exc}")

        # BBOX-Clip (falls verfügbar)
        bbox_25833 = self._get_bbox_from_context(ctx)
        if bbox_25833 is not None:
            print(f"[GHS-OBAT] Clip auf BBOX (EPSG:25833): {bbox_25833}")
            bbox_poly = shp_box(*bbox_25833)
            gdf = gdf[gdf.geometry.intersects(bbox_poly)].copy()
        else:
            print("[GHS-OBAT] Keine BBOX im Context – verwende gesamten Datensatz.")

        if gdf.empty:
            print("[GHS-OBAT] Ergebnis leer – kein Clip geschrieben.")
            return gdf

        # Nur zentrale Attribute behalten (Id, Epoch, Height, Use, Area, Perimeter etc.)
        cols_keep = ["geometry"]
        # Namen exakt nach Spezifikation:
        candidate_cols = [
            # Identifikator
            "id",
            # Admin/Location
            "country", "adm1",
            # physische Eigenschaften
            "height", "shapefactor", "area", "perimeter",
            # Nutzung und Baualtersklasse
            "use", "epoch",
            # evtl. Varianten in der Datei (Groß/Kleinschreibung)
            "Id", "ID",
            "Country", "Adm1",
            "Height", "ShapeFactor", "Shape_Factor",
            "Area", "Perimeter",
            "Use", "Epoch",
        ]
        for c in candidate_cols:
            if c in gdf.columns and c not in cols_keep:
                cols_keep.append(c)

        gdf = gdf[cols_keep].copy()

        # Optional: Debug-Head ohne Geometrie zur Kontrolle
        try:
            gdf.drop(columns=["geometry"], errors="ignore").head(50).to_csv(
                out_dir / "ghs_obat_head.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte Head-CSV nicht schreiben: {exc}")

        # Clip-Datei schreiben
        try:
            gdf.to_file(clip_path, layer="ghs_obat", driver="GPKG")
            print(f"[GHS-OBAT] Clip nach {clip_path} geschrieben ({len(gdf)} Features).")
        except Exception as exc:
            print(f"[GHS-OBAT] WARN: Konnte Clip-GPKG {clip_path} nicht schreiben: {exc}")

        return gdf



