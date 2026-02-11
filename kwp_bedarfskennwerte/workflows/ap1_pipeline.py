"""
AP1-Pipeline: Geometrische und strukturelle Anreicherung.

Schritte (vereinfacht):
1) LoD2-Import, QA und Geometrieaggregation
2) Basemap/OSM-Integration (Nutzung/Attribute)
3) Vereinheitlichung der Nutzungsklassen
4) QA-Exports (CSV/GPKG)

Ausgabe ist ein standardisierter Geb?udelayer f?r AP1-Enrichment/AP2.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import re
import os
import time
import json
import pandas as pd
import geopandas as gpd
import shapely
#from geopandas.array import is_geometry_type
from shapely import wkb
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.errors import GEOSException

from ..config.runtime import PipelineContext
from ..data_catalog.sources import (
    LoD2CityGMLSource,
    BasemapContextSource,
    BasemapCfg,
    OSMSource
)

from ..data_catalog.analyzer import  (
    DataProfiler,
    AP1CSVStatistics,
    BasicGeometryAnalysisLOD2Shapefile
)



def _write_qa_summary(
        step_name: str,
        input_name: str,
        input_gdf: gpd.GeoDataFrame,
        output_name: str,
        output_gdf: gpd.GeoDataFrame,
        qa_dir: Path,
):
    """
    Schreibt eine kleine QA-CSV mit Kennzahlen zum Eingabe- und Ausgabedatensatz.
    """
    qa_dir.mkdir(parents=True, exist_ok=True)

    def _summ(label: str, gdf: gpd.GeoDataFrame) -> dict:
        if gdf is None or gdf.empty:
            return {
                "dataset": label,
                "n_features": 0,
                "n_unique_LOD_UNITID": 0,
                "crs": "",
                "area_col": "",
                "area_sum_m2": 0.0,
                "area_min_m2": 0.0,
                "area_max_m2": 0.0,
            }
        # geeignete Flächenspalte suchen
        area_col = None
        for cand in ["LOD_Grundflaeche_m2", "area_m2", "__area"]:
            if cand in gdf.columns:
                area_col = cand
                break

        if area_col:
            s = pd.to_numeric(gdf[area_col], errors="coerce")
            s_valid = s[s > 0].dropna()
            area_sum = float(s_valid.sum()) if not s_valid.empty else 0.0
            area_min = float(s_valid.min()) if not s_valid.empty else 0.0
            area_max = float(s_valid.max()) if not s_valid.empty else 0.0
        else:
            area_col = ""
            area_sum = area_min = area_max = 0.0

        n_unitid = 0
        if "LOD_UNITID" in gdf.columns:
            n_unitid = int(gdf["LOD_UNITID"].astype(str).nunique())

        crs_str = ""
        try:
            crs_str = str(gdf.crs) if gdf.crs is not None else ""
        except Exception:
            pass

        return {
            "dataset": label,
            "n_features": int(len(gdf)),
            "n_unique_LOD_UNITID": n_unitid,
            "crs": crs_str,
            "area_col": area_col,
            "area_sum_m2": area_sum,
            "area_min_m2": area_min,
            "area_max_m2": area_max,
        }

    rows = [
        _summ(input_name, input_gdf),
        _summ(output_name, output_gdf),
    ]
    df = pd.DataFrame(rows)
    out_csv = qa_dir / f"qa_{step_name}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[QA] {step_name}: {out_csv}")


def _first_nonnull(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if not s2.empty else None


def _mode_or_first(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return None
    m = s2.mode()
    return m.iloc[0] if not m.empty else s2.iloc[0]


def compute_priority_unified(
    lod_on_lod: pd.DataFrame,
    bmap_on_lod: pd.DataFrame,
    osm_on_lod: pd.DataFrame,
) -> pd.DataFrame:
    """
    Priorisiert Nutzung aus LOD2 > Basemap > OSM.
    Erwartete Spalten in *_on_lod (Index = LOD_UNITID):
      <SRC>_Nutzung_original, <SRC>_Nutzung_vereinheitlicht, <SRC>_NWGoderWG
    Default: WG, wenn alles leer.
    """
    # zusammenführen (äußerer Join, um überall Index zu haben)
    df = lod_on_lod.join(bmap_on_lod, how="outer").join(osm_on_lod, how="outer")

    def _pick(*vals):
        for v in vals:
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                s = str(v).strip()
                if s != "" and s.lower() != "nan":
                    return s
        return None

    # priorisiert: Original, vereinheitlicht, NWG/WG
    out = pd.DataFrame(index=df.index)
    out["Final_Nutzung_original"] = df.apply(
        lambda r: _pick(
            r.get("LOD_Nutzung_original"),
            r.get("BMAP_Nutzung_original"),
            r.get("OSM_Nutzung_original"),
        ), axis=1
    )
    out["Final_Nutzung_vereinheitlicht"] = df.apply(
        lambda r: _pick(
            r.get("LOD_Nutzung_vereinheitlicht"),
            r.get("BMAP_Nutzung_vereinheitlicht"),
            r.get("OSM_Nutzung_vereinheitlicht"),
        ), axis=1
    )
    # WG/NWG: falls alles fehlt -> WG (Default)
    def _pick_nwg(r):
        v = _pick(
            r.get("LOD_NWGoderWG"),
            r.get("BMAP_NWGoderWG"),
            r.get("OSM_NWGoderWG"),
        )
        return v if v in ("WG", "NWG") else "WG"

    out["Final_NWGoderWG"] = df.apply(_pick_nwg, axis=1)

    return out


# --- HILFSFUNKTIONEN CRS, ungültige Geometrien und Schnittflächenberechnung --------------------


def _ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    """
    Setzt bei Bedarf ein CRS und reprojiziert auf EPSG=epsg.
    Kommt ohne pyproj.CRS aus (nur GeoPandas APIs).
    """
    if gdf.crs is None or str(gdf.crs).strip().lower() in ("", "none"):
        gdf = gdf.set_crs(epsg=epsg)
    try:
        current = gdf.crs.to_epsg()
    except Exception:
        current = None
    if current != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf


def _fix_invalid_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Repariert ungültige Geometrien. Versucht zuerst make_valid (falls vorhanden),
    sonst buffer(0). Lässt fehlerhafte Geometrien als leere Polygone stehen.
    """
    try:
        # GeoPandas/Shapely >= 2: make_valid verfügbar
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        # Fallback für ältere Umgebungen
        def _safe_buf0(geom):
            try:
                if geom is None:
                    return geom
                return geom.buffer(0)
            except Exception:
                return geom  # notfalls unverändert
        gdf["geometry"] = gdf.geometry.apply(_safe_buf0)
    return gdf


def _safe_intersection_area(geom_a, geom_b) -> float:
    """
    Schnelle und robuste Schnittflächenberechnung.
    Gibt 0.0 zurück bei ungültigen/empty Geometrien oder GEOS-Fehlern.
    """
    if geom_a is None or geom_b is None:
        return 0.0
    if geom_a.is_empty or geom_b.is_empty:
        return 0.0
    try:
        return geom_a.intersection(geom_b).area
    except GEOSException:
        return 0.0
    except Exception:
        return 0.0


# --- FUNKTION FÜR RÄUMLICHES MAPPING: Basemap/OSM -> LOD2 --------------------


def attach_best_overlap(
    src_gdf: gpd.GeoDataFrame,
    lod2_build: gpd.GeoDataFrame,
    prefix: str,
    target_epsg: int,
) -> pd.DataFrame:
    """
    Räumliches Mapping: pro LOD2-Feature den besten überlappenden Treffer aus src_gdf
    wählen und die gewünschten Attribute als Spalten mit Präfix zurückgeben.

    Erwartet in src_gdf (sofern vorhanden):
      - geometry
      - orig_usage, unified_detailed, is_residential
      - height_m, area_m2
      - source_id / id / bkg_id / osm_id

    WICHTIG: Der Index der Rückgabe ist LOD_UNITID (String),
    damit er konsistent mit lod_on_lod ist.
    """
    # Leere Hülle, falls Quelle leer
    out_cols = [
        f"{prefix}_Nutzung_original",
        f"{prefix}_Nutzung_vereinheitlicht",
        f"{prefix}_NWGoderWG",
        f"{prefix}_height_m",
        f"{prefix}_area_m2",
        f"{prefix}_source_id",
        f"{prefix}_year",
    ]
    if src_gdf is None or len(src_gdf) == 0:
        idx = lod2_build["LOD_UNITID"].astype(str) if "LOD_UNITID" in lod2_build.columns else lod2_build.index
        df_out = pd.DataFrame(index=idx, columns=out_cols)
        return df_out

    # Index: LOD_UNITID, falls vorhanden
    if "LOD_UNITID" in lod2_build.columns:
        idx = lod2_build["LOD_UNITID"].astype(str)
    else:
        idx = lod2_build.index

    # 1) CRS angleichen + Geometrien fixen
    lod_tmp = _ensure_crs(lod2_build[["geometry"]].copy(), target_epsg)
    lod_tmp.index = idx
    src_tmp = _ensure_crs(src_gdf.copy(), target_epsg)
    lod_tmp = _fix_invalid_geoms(lod_tmp)
    src_tmp = _fix_invalid_geoms(src_tmp)

    # 2) Kandidaten per intersects
    joined = gpd.sjoin(lod_tmp, src_tmp, how="left", predicate="intersects")
    if joined.empty:
        df_out = pd.DataFrame(index=idx, columns=out_cols)
        return df_out

    # 3) Overlap robust berechnen
    geom_l = joined.geometry
    geom_r = joined["geometry_right"] if "geometry_right" in joined.columns else joined["geometry"]
    joined["__overlap"] = [
        _safe_intersection_area(a, b) for a, b in zip(geom_l, geom_r)
    ]

    # 4) pro LOD2-Index den besten Treffer
    best_idx = joined.groupby(joined.index)["__overlap"].idxmax()
    best = joined.loc[best_idx].copy()

    # 5) Aus best -> Mapping-Dicts bauen (Index = LOD2-Index), dann via .map() zuweisen
    def pick_id_col(df: pd.DataFrame) -> str:
        """
        F?hrt pick_id_col aus.
        
        Args:
            df: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        for cand in ("source_id", "id", "bkg_id", "osm_id"):
            if cand in df.columns:
                return cand
        return "source_id"  # fallback (fehlt dann ggf. -> NaN)

    id_col = pick_id_col(best)

    # Helper: Werte zu Labels
    def wg_label(v):
        """
        F?hrt wg_label aus.
        
        Args:
            v: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        try:
            return "WG" if bool(v) else "NWG"
        except Exception:
            return pd.NA

    maps = {
        f"{prefix}_Nutzung_original":        best.get("orig_usage", pd.Series(index=best.index)).to_dict(),
        f"{prefix}_Nutzung_vereinheitlicht": best.get("unified_detailed", pd.Series(index=best.index)).to_dict(),
        f"{prefix}_NWGoderWG":               best.get("is_residential", pd.Series(index=best.index)).map(wg_label)
                                            if "is_residential" in best.columns else pd.Series(index=best.index),
        f"{prefix}_height_m":                best.get("height_m", pd.Series(index=best.index)).to_dict(),
        f"{prefix}_area_m2":                 best.get("area_m2", pd.Series(index=best.index)).to_dict(),
        f"{prefix}_source_id":               best.get(id_col, pd.Series(index=best.index)).to_dict(),
        f"{prefix}_year":                    best.get("year", pd.Series(index=best.index)).to_dict(),
    }

    # falls NWGoderWG als Series vorliegt, erst in Dict wandeln
    if not isinstance(maps[f"{prefix}_NWGoderWG"], dict):
        maps[f"{prefix}_NWGoderWG"] = maps[f"{prefix}_NWGoderWG"].to_dict()

    # 6) Output-DF mit Index = LOD_UNITID befüllen
    df_out = pd.DataFrame(index=idx, columns=out_cols)
    for col, mapping in maps.items():
        df_out[col] = [mapping.get(i, pd.NA) for i in df_out.index]

    return df_out


def map_basemap_classes(
    lod2_build: gpd.GeoDataFrame,
    bm_buildings: gpd.GeoDataFrame,
    bm_context: gpd.GeoDataFrame | None,
    target_epsg: int,
) -> pd.DataFrame:
    """
    Liefert für jedes LOD2-Gebaeude (Index = LOD_UNITID) eine Zuordnung der Basemap-Layer:

      - BMAP_Gebaeudeflaeche_klasse   (POLYGON, größter Overlap)
      - BMAP_Gebaeudepunkt_klasse     (POINT, Punkt innerhalb Gebaeude -> Modus)
      - BMAP_Siedlungsflaeche_klasse  (POLYGON, größter Overlap)
    """

    # Index = LOD_UNITID
    if "LOD_UNITID" in lod2_build.columns:
        idx = lod2_build["LOD_UNITID"].astype(str)
    else:
        idx = lod2_build.index

    df_out = pd.DataFrame(index=idx)

    # Helper: LOD2-Geometrie vorbereiten
    lod_tmp = _ensure_crs(lod2_build[["geometry"]].copy(), target_epsg)
    lod_tmp.index = idx
    lod_tmp = _fix_invalid_geoms(lod_tmp)

    # --- 1) Gebaeudeflaeche.klasse (bm_buildings) – Polygon-Overlap ----------------
    if bm_buildings is not None and not bm_buildings.empty and "klasse" in bm_buildings.columns:
        src = bm_buildings.copy()
        src = src[src.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not src.empty:
            src = _ensure_crs(src[["geometry", "klasse"]].copy(), target_epsg)
            src = _fix_invalid_geoms(src)

            joined = gpd.sjoin(lod_tmp, src, how="left", predicate="intersects")
            if not joined.empty:
                geom_l = joined.geometry
                geom_r = joined["geometry_right"] if "geometry_right" in joined.columns else joined["geometry"]
                joined["__overlap"] = [
                    _safe_intersection_area(a, b) for a, b in zip(geom_l, geom_r)
                ]
                best_idx = joined.groupby(joined.index)["__overlap"].idxmax()
                best = joined.loc[best_idx]
                mapping = best["klasse"].to_dict()
                df_out["BMAP_Gebaeudeflaeche_klasse"] = [mapping.get(i, pd.NA) for i in df_out.index]
            else:
                df_out["BMAP_Gebaeudeflaeche_klasse"] = pd.NA
        else:
            df_out["BMAP_Gebaeudeflaeche_klasse"] = pd.NA
    else:
        df_out["BMAP_Gebaeudeflaeche_klasse"] = pd.NA

    # Wenn kein Kontext vorhanden ist, sind die anderen Felder leer
    if bm_context is None or bm_context.empty:
        df_out["BMAP_Gebaeudepunkt_klasse"] = pd.NA
        df_out["BMAP_Siedlungsflaeche_klasse"] = pd.NA
        return df_out

    ctx = bm_context.copy()

    # --- 2) Gebaeudepunkt.klasse – Punkt innerhalb Gebaeude (Modus) ------------------
    gp = ctx[(ctx.get("bm_layer") == "Gebaeudepunkt") & (ctx.geometry.type.isin(["Point", "MultiPoint"]))].copy()
    if not gp.empty and "klasse" in gp.columns:
        gp = _ensure_crs(gp[["geometry", "klasse"]].copy(), target_epsg)
        gp = _fix_invalid_geoms(gp)

        # Join: Punkte innerhalb Gebaeude
        joined = gpd.sjoin(lod_tmp, gp, how="left", predicate="contains")
        if not joined.empty:
            # groupby LOD_UNITID -> Modus von klasse_right
            if "klasse_right" in joined.columns:
                cls_col = "klasse_right"
            else:
                cls_col = "klasse"

            def _mode_or_first_series(s: pd.Series):
                s2 = s.dropna()
                if s2.empty:
                    return pd.NA
                m = s2.mode()
                return m.iloc[0] if not m.empty else s2.iloc[0]

            ser = joined.groupby(joined.index)[cls_col].agg(_mode_or_first_series)
            mapping = ser.to_dict()
            df_out["BMAP_Gebaeudepunkt_klasse"] = [mapping.get(i, pd.NA) for i in df_out.index]
        else:
            df_out["BMAP_Gebaeudepunkt_klasse"] = pd.NA
    else:
        df_out["BMAP_Gebaeudepunkt_klasse"] = pd.NA

    # --- 3) Siedlungsflaeche.klasse – Polygon-Overlap -------------------------------
    sf = ctx[(ctx.get("bm_layer") == "Siedlungsflaeche") & (ctx.geometry.type.isin(["Polygon", "MultiPolygon"]))].copy()
    if not sf.empty and "klasse" in sf.columns:
        sf = _ensure_crs(sf[["geometry", "klasse"]].copy(), target_epsg)
        sf = _fix_invalid_geoms(sf)

        joined = gpd.sjoin(lod_tmp, sf, how="left", predicate="intersects")
        if not joined.empty:
            geom_l = joined.geometry
            geom_r = joined["geometry_right"] if "geometry_right" in joined.columns else joined["geometry"]
            joined["__overlap"] = [
                _safe_intersection_area(a, b) for a, b in zip(geom_l, geom_r)
            ]
            best_idx = joined.groupby(joined.index)["__overlap"].idxmax()
            best = joined.loc[best_idx]
            mapping = best["klasse"].to_dict()
            df_out["BMAP_Siedlungsflaeche_klasse"] = [mapping.get(i, pd.NA) for i in df_out.index]
        else:
            df_out["BMAP_Siedlungsflaeche_klasse"] = pd.NA
    else:
        df_out["BMAP_Siedlungsflaeche_klasse"] = pd.NA

    return df_out


class AP1Pipeline:
    """
    AP1 – Quellenaufnahme, Harmonisierung, QA/Outputs
        AP1-Pipeline:
      - LOD2 laden, bereinigen, pro UNITID zu Gebaeuden aggregieren (3D-Geometrie beibehalten, 2D-Footprint intern)
      - Basemap (Gebaeudeflächen) aus MVT holen
      - OSM (building=*), robust via Endpoint-Rotation/Slicing (in OSMSource)
      - räumliches Mapping: Basemap/OSM -> LOD2 (größte Überlappung)
      - Priorisierung Nutzung: LOD2 > Basemap > OSM
      - Ausgaben:
          * out/ap1/Compare/compare_lod2_bmap_osm.csv
          * out/ap1/Compare/nutzungsspezifikation_vereinheitlicht.gpkg
    """

    def run(self, ctx):
        """
        AP1: LOD2 (3D beibehalten) + Basemap + OSM
        - LOD2: Aggregation je UNITID (größte 2D-Grundfläche) + Zusatzlayer "lod2_surfaces" (alle Teile)
        - Basemap: Nutzung aus Gebaeudefläche.klasse, optional via nächstgelegenen Gebaeudepunkt.klasse ergänzt;
                   zusätzlich Mapping der Basemap-Layer:
                       * Gebaeudeflaeche.klasse
                       * Gebaeudepunkt.klasse
                       * Siedlungsflaeche.klasse
                     auf jedes LOD2-Gebaeude.
        - OSM: Gebaeude (Overpass) + Vereinheitlichung auf LOD2-Basisnutzungstypen (Mapping-Tabelle)
        - Räumliches Mapping auf LOD2 (größter Overlap)
        - Compare-CSV + GPKG (im Compare-Unterordner)
        """
        import pandas as pd
        import geopandas as gpd
        from pathlib import Path
        from shapely.geometry.base import BaseGeometry
        from shapely import wkb


        # ------------------------------------------------------------
        # Sonderfall: Nur Geometrie-Analyse der Eingangs-LoD2-Datei
        # (CLI-Flag: --analyse)
        # ------------------------------------------------------------
        settings = getattr(ctx, "settings", ctx)

        if bool(getattr(settings, "analyse", False)):
            # LoD2-Pfad aus Settings
            data_cfg = getattr(settings, "data", None)
            lod2_path = getattr(data_cfg, "lod2_path", None) if data_cfg else None
            if not lod2_path:
                raise ValueError(
                    "Für --analyse wird ein LoD2-Pfad benötigt "
                    "(CLI: --lod2-path)."
                )

            out_dir = Path(settings.out_dir) / "ap1" / "geometry_analysis"
            analyzer = BasicGeometryAnalysisLOD2Shapefile(
                target_epsg=int(getattr(settings, "target_epsg", 25833)),
                verbose=bool(getattr(settings, "verbose", False)),
            )
            result_paths = analyzer.run(lod2_path=Path(lod2_path), out_dir=out_dir)

            if getattr(settings, "verbose", False):
                print("[ap1] Nur Geometrieanalyse der Eingangs-LoD2-Datei ausgeführt.")
                print(f"[ap1] Ergebnisse in: {out_dir}")
                for key, path in result_paths.items():
                    print(f"  - {key}: {path}")

            # WICHTIG: hier frühzeitig abbrechen – keine Basemap/OSM-Requests etc.
            return result_paths



        settings = ctx.settings
        target_epsg = int(getattr(settings, "target_epsg", 25833))
        out_dir = Path(getattr(settings, "out_dir", "out")) / "ap1"
        out_dir.mkdir(parents=True, exist_ok=True)

        compare_dir = out_dir / "Compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        qa_dir = out_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)


        # ------------------------------------------------------------
        # Hilfsfunktionen (lokal)
        # ------------------------------------------------------------
        from typing import Tuple

        def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
            cols = []
            seen = {}
            for c in df.columns:
                if c not in seen:
                    seen[c] = 1
                    cols.append(c)
                else:
                    seen[c] += 1
                    cols.append(f"{c}_{seen[c]}")
            df = df.copy()
            df.columns = cols
            return df

        # --- helpers: force 2D -------------------------------------------------------

        def _geom_to_2d(gdf):
            """
            Gibt ein GeoDataFrame mit 2D-Geometrien zurück.
            Entfernt Z (und M) robust via WKB-Re/Serialize.
            """
            if gdf is None or gdf.empty:
                return gdf

            def _drop_z(geom: BaseGeometry):
                if geom is None:
                    return None
                try:
                    return wkb.loads(wkb.dumps(geom, output_dimension=2))
                except Exception:
                    return geom

            gdf = gdf.copy()
            if gdf._geometry_column_name not in gdf.columns:
                raise ValueError("Geometry column not found in GeoDataFrame.")
            gdf[gdf._geometry_column_name] = gdf.geometry.map(_drop_z)
            return gdf

        def _ensure_same_crs(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
            if a.crs != b.crs:
                b = b.to_crs(a.crs)
            return a, b

        # --------- LOD2 vereinheitlichen (unter Erhalt der LOD_* Felder) ---------

        def _unify_lod2(lod2_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """
            Vereinheitlicht LOD2, aber respektiert Felder, die bereits von LoD2CityGMLSource
            geliefert wurden (insb. LOD_UNITID, LOD_Nutzung_*, LOD_GebHoehe, LOD_Stockwerke,
            LOD_Grundflaeche_m2, function_label, is_func_unknown).
            """
            gdf = lod2_gdf.copy()

            if "building_id" in gdf.columns:
                gdf["building_id"] = gdf["building_id"].astype(str)

            if "LOD_UNITID" in gdf.columns:
                gdf["LOD_UNITID"] = gdf["LOD_UNITID"].astype(str)
            else:
                if "building_id" in gdf.columns:
                    gdf["LOD_UNITID"] = gdf["building_id"].astype(str)
                elif "UNITID" in gdf.columns:
                    gdf["LOD_UNITID"] = gdf["UNITID"].astype(str)
                else:
                    gdf["LOD_UNITID"] = gdf.index.astype(str)

            have_usage_cols = all(
                c in gdf.columns
                for c in ["LOD_Nutzung_original", "LOD_Nutzung_vereinheitlicht", "LOD_NWGoderWG"]
            )

            if not have_usage_cols:
                raw = []
                for _, r in gdf.iterrows():
                    v = None
                    if pd.notna(r.get("GebFunkion")) and str(r.get("GebFunkion")).strip():
                        v = str(r.get("GebFunkion")).strip()
                    elif pd.notna(r.get("UNITNAME")) and str(r.get("UNITNAME")).strip():
                        v = str(r.get("UNITNAME")).strip()
                    raw.append(v)

                if "LOD_Nutzung_original" not in gdf.columns:
                    gdf["LOD_Nutzung_original"] = raw

                def _map(s: str) -> str:
                    if not isinstance(s, str) or not s:
                        return "Nach Quellenlage nicht zu spezifizieren"
                    s_l = s.lower()
                    if "wohn" in s_l:
                        return "Wohngebäude"
                    if any(k in s_l for k in ["schule", "kindergarten", "hochschule", "universität", "university"]):
                        return "Gebaeude für Bildung und Forschung"
                    if "kirche" in s_l:
                        return "Kirche"
                    if "kranken" in s_l or "hospital" in s_l:
                        return "Krankenhaus"
                    if any(k in s_l for k in
                           ["büro", "buero", "office", "retail", "gewerbe", "gewerb", "handel", "industrie",
                            "industrial", "commercial", "factory", "fabrik"]):
                        return "Gebaeude für Wirtschaft oder Gewerbe"
                    return "Nach Quellenlage nicht zu spezifizieren"

                if "LOD_Nutzung_vereinheitlicht" not in gdf.columns:
                    gdf["LOD_Nutzung_vereinheitlicht"] = gdf["LOD_Nutzung_original"].map(_map)
                if "LOD_NWGoderWG" not in gdf.columns:
                    gdf["LOD_NWGoderWG"] = gdf["LOD_Nutzung_vereinheitlicht"].map(
                        lambda x: "WG" if isinstance(x, str) and "wohn" in x.lower()
                        else ("NWG" if x not in (None, "Nach Quellenlage nicht zu spezifizieren") else "Unbekannt")
                    )

                for c in ["Dachform", "GebHoehe", "Stockwerke"]:
                    if c not in gdf.columns:
                        gdf[c] = pd.NA

            return gdf

        def _agg_lod2_to_buildings(
            lod2_u: gpd.GeoDataFrame,
            min_area_m2: float = 25.0,
            target_epsg: int = 25833,
        ):
            """
            Aggregiert LOD2-Teilflächen zu Gebaeuden:

            - nutzt LOD_UNITID als Gebaeude-ID (muss vorhanden sein),
            - transformiert Geometrien in target_epsg (falls nötig),
            - berechnet 2D-Grundflächen (__area) bevorzugt aus der
              Spalte '__geom2d' (falls vorhanden), sonst direkt aus
              geometry (Shapely ignoriert dabei Z),
            - filtert Gebaeude anhand der SUMME der Flächen je LOD_UNITID
              (Mindestfläche),
            - liefert:
                * lod2_build:   ein Footprint (größte Teilfläche) pro Gebaeude
                                (Geometrie bleibt 3D erhalten)
                * lod2_surfaces: alle Teilflächen der gefilterten Gebaeude
                                  (Geometrie bleibt 3D erhalten)
            """
            if lod2_u is None or lod2_u.empty:
                return (
                    gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}"),
                    gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}"),
                )

            gdf = lod2_u.copy()

            # CRS harmonisieren
            if gdf.crs is None:
                gdf.set_crs(epsg=target_epsg, inplace=True)
            elif gdf.crs.to_epsg() != target_epsg:
                gdf = gdf.to_crs(epsg=target_epsg)

            if "LOD_UNITID" not in gdf.columns:
                raise ValueError("[LoD2] LOD_UNITID fehlt im vereinheitlichten LOD2-DataFrame.")

            # 2D-Geometrie für Flächenberechnung:
            # bevorzugt __geom2d (aus LoD2CityGMLSource), sonst geometry
            if "__geom2d" in gdf.columns:
                geom2d = gpd.GeoSeries(gdf["__geom2d"], crs=gdf.crs)
            else:
                geom2d = gdf.geometry

            gdf["__area"] = geom2d.area

            # Mindestfläche je Gebaeude (aggregiert über alle Teilflächen)
            area_by_id = gdf.groupby("LOD_UNITID")["__area"].sum()
            before_ids = int(area_by_id.shape[0])
            valid_ids = area_by_id[area_by_id >= float(min_area_m2)].index
            after_ids = int(len(valid_ids))

            gdf = gdf[gdf["LOD_UNITID"].isin(valid_ids)].copy()

            print(
                f"[LoD2] Mindestfläche {min_area_m2} m² (Summe je LOD_UNITID): "
                f"{before_ids} -> {after_ids} Gebaeude"
            )

            # Footprint = größte Teilfläche je Gebaeude (nach __area)
            gdf["_rank"] = gdf.groupby("LOD_UNITID")["__area"].rank(
                method="first", ascending=False
            )

            lod2_build = gdf[gdf["_rank"] == 1.0].drop(columns=["_rank"]).copy()
            lod2_surfaces = gdf.drop(columns=["_rank"]).copy()

            # Geometrie bleibt in beiden DataFrames unverändert (inkl. Z)
            return lod2_build, lod2_surfaces



        # --- Basemap vereinheitlichen ------------------------------------------------

        def _unify_basemap(bm_buildings: gpd.GeoDataFrame, bm_pois: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
            if bm_buildings is None or bm_buildings.empty:
                return gpd.GeoDataFrame(geometry=[],
                                        crs=bm_buildings.crs if bm_buildings is not None else f"EPSG:{target_epsg}")

            b = bm_buildings.copy()
            b["bmap_usage_raw_bldg"] = b.get("klasse")
            b["bmap_hoehe"] = pd.to_numeric(b.get("hoehe"), errors="coerce")

            if isinstance(bm_pois, gpd.GeoDataFrame) and (not bm_pois.empty):
                pois = bm_pois.copy()
                poi_use_col = None
                if "klasse" in pois.columns:
                    poi_use_col = "klasse"
                elif "class" in pois.columns:
                    poi_use_col = "class"

                if poi_use_col is not None:
                    if "bm_layer" in pois.columns:
                        gp = pois[pois["bm_layer"] == "Gebaeudepunkt"].copy()
                        if gp.empty:
                            gp = pois.copy()
                    elif "source_layer" in pois.columns:
                        gp = pois[pois["source_layer"].str.contains("Gebaeudepunkt", case=False, na=False)].copy()
                        if gp.empty:
                            gp = pois.copy()
                    else:
                        gp = pois.copy()

                    b, gp = _ensure_same_crs(b, gp)

                    try:
                        joined = gpd.sjoin_nearest(
                            b[["geometry"]],
                            gp[[poi_use_col, "geometry"]],
                            how="left",
                            max_distance=30
                        )
                        b["bmap_usage_raw_poi"] = joined[poi_use_col].values
                    except Exception:
                        b["bmap_usage_raw_poi"] = pd.NA
                else:
                    b["bmap_usage_raw_poi"] = pd.NA
            else:
                b["bmap_usage_raw_poi"] = pd.NA

            def _pick(a, b_):
                if pd.notna(a) and str(a).strip():
                    return str(a).strip()
                if pd.notna(b_) and str(b_).strip():
                    return str(b_).strip()
                return None

            b["orig_usage"] = [_pick(p, q) for p, q in zip(b["bmap_usage_raw_poi"], b["bmap_usage_raw_bldg"])]

            def _map_usage(u):
                if not isinstance(u, str) or not u.strip():
                    return "Nach Quellenlage nicht zu spezifizieren"
                s = u.lower()
                if "wohn" in s:
                    return "Wohngebäude"
                if any(k in s for k in ["schule", "hochschule", "kindergarten", "bildung", "universität"]):
                    return "Gebaeude für Bildung und Forschung"
                if "kirche" in s or "relig" in s:
                    return "Kirche"
                if "kranken" in s or "hospital" in s:
                    return "Krankenhaus"
                if any(k in s for k in
                       ["büro", "buero", "office", "retail", "gewerbe", "gewerb", "handel", "industrie",
                        "industrial", "gewerbegebiet", "gewerbefläche", "factory", "fabrik", "gewerbegebiet"]):
                    return "Gebaeude für Wirtschaft oder Gewerbe"
                return "Nach Quellenlage nicht zu spezifizieren"

            b["unified_detailed"] = b["orig_usage"].map(_map_usage)
            b["is_residential"] = b["unified_detailed"].eq("Wohngebäude")

            if "ALKIS_ID" in b.columns:
                b["building_id"] = b["ALKIS_ID"].astype(str)
            elif "gml_id" in b.columns:
                b["building_id"] = b["gml_id"].astype(str)
            else:
                b["building_id"] = ["bm_" + str(i) for i in range(len(b))]

            keep = ["building_id", "geometry", "orig_usage", "unified_detailed", "is_residential", "bmap_hoehe",
                    "bmap_usage_raw_bldg", "bmap_usage_raw_poi"]
            keep = [c for c in keep if c in b.columns] + (["geometry"] if "geometry" not in keep else [])
            return b[keep]

        # --- OSM vereinheitlichen (inkl. Mapping-Tabelle auf LOD2-Basistypen) ------

        def _unify_osm(osm_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            if osm_gdf is None or osm_gdf.empty:
                return gpd.GeoDataFrame(geometry=[], crs=osm_gdf.crs if osm_gdf is not None else f"EPSG:{target_epsg}")
            g = osm_gdf.copy()

            raw = g.get("function_raw")
            if raw is None:
                for c in ["amenity", "building", "landuse", "shop", "office"]:
                    if c in g.columns:
                        raw = g[c]
                        break
            g["orig_usage"] = raw

            # Mapping-Tabelle OSM -> LOD2-Basistypen (Text)
            OSM_MAP = {
                "building=apartments": "Wohngebäude",
                "building=yes": "Nach Quellenlage nicht zu spezifizieren",
                "building=industrial": "Gebaeude für Wirtschaft oder Gewerbe",
                "<NA>": "Nach Quellenlage nicht zu spezifizieren",
                "building=garages": "Parkhaus",
                "building=allotment_house": "Gebaeude für öffentliche Zwecke",
                "building=residential": "Wohngebäude",
                "building=shed": "Nach Quellenlage nicht zu spezifizieren",
                "building=house": "Nach Quellenlage nicht zu spezifizieren",
                "building=garage": "Parkhaus",
                "building=retail": "Gebaeude für Wirtschaft oder Gewerbe",
                "building=service": "Gebaeude für Wirtschaft oder Gewerbe",
                "building=hospital": "Krankenhaus",
                "building=school": "Gebaeude für Bildung und Forschung",
                "building=office": "Bürogebäude",
                "building=detached": "Wohngebäude",
                "building=carport": "Parkhaus",
                "building=civic": "Nach Quellenlage nicht zu spezifizieren",
                "amenity=kindergarten": "Kinderkrippe, Kindergarten, Kindertagesstätte",
                "building=hotel": "Jugendherberge",
                "building=semidetached_house": "Wohngebäude",
                "building=commercial": "Gebaeude für Wirtschaft oder Gewerbe",
                "building=villa": "Wohngebäude",
                "building=roof": "Nach Quellenlage nicht zu spezifizieren",
                "amenity=police": "Polizei",
                "building=kindergarten": "Allgemein bildende Schule",
                "building=warehouse": "Gebaeude für Wirtschaft oder Gewerbe",
                "building=construction": "Nach Quellenlage nicht zu spezifizieren",
                "amenity=place_of_worship": "Nach Quellenlage nicht zu spezifizieren",
                "building=parking": "Parkhaus",
                "building=fire_station": "Feuerwehr",
                "amenity=restaurant": "Gebaeude für Wirtschaft oder Gewerbe",
                "building=train_station": "Bahnhofsgebäude",
                "amenity=car_wash": "Nach Quellenlage nicht zu spezifizieren",
                "building=silo": "Nach Quellenlage nicht zu spezifizieren",
                "amenity=post_depot": "Gebaeude für öffentliche Zwecke",
                "building=government": "Gebaeude für öffentliche Zwecke",
                "building=public": "Gebaeude für öffentliche Zwecke",
                "amenity=crematorium": "Trauerhalle",
                "building=university": "Hochschulgebäude (Fachhochschule,Universität)",
                "building=toilets": "Nach Quellenlage nicht zu spezifizieren",
                "building=factory": "Gebaeude für Wirtschaft oder Gewerbe",
                "amenity=bicycle_parking": "Nach Quellenlage nicht zu spezifizieren",
                "amenity=school": "Allgemein bildende Schule",
                "building=kiosk": "Nach Quellenlage nicht zu spezifizieren",
                "building=ruins": "Nach Quellenlage nicht zu spezifizieren",
                "building=shelter": "Nach Quellenlage nicht zu spezifizieren",
                "building=arch": "Nach Quellenlage nicht zu spezifizieren",
                "building=bungalow": "Nach Quellenlage nicht zu spezifizieren",
            }

            def _map_osm(u):
                if u is None or (isinstance(u, float) and pd.isna(u)):
                    return "Nach Quellenlage nicht zu spezifizieren"
                s = str(u).strip()
                if s in OSM_MAP:
                    return OSM_MAP[s]
                s_l = s.lower()
                # Fallback-Heuristik
                if "residential" in s_l or "apart" in s_l or "house" in s_l or "villa" in s_l or "bungalow" in s_l:
                    return "Wohngebäude"
                if "school" in s_l or "kindergarten" in s_l or "university" in s_l or "college" in s_l:
                    return "Gebaeude für Bildung und Forschung"
                if "church" in s_l or "place_of_worship" in s_l:
                    return "Kirche"
                if "hospital" in s_l:
                    return "Krankenhaus"
                if any(k in s_l for k in ["office", "retail", "commercial", "industrial", "warehouse", "shop", "hotel",
                                           "factory"]):
                    return "Gebaeude für Wirtschaft oder Gewerbe"
                return "Nach Quellenlage nicht zu spezifizieren"

            g["unified_detailed"] = g["orig_usage"].map(_map_osm)
            g["is_residential"] = g["unified_detailed"].str.contains("wohn", case=False, na=False)

            # --------------------------------------------------------
            # Höhen- und Flächenableitung
            # --------------------------------------------------------
            def _parse_height_osm(h, lvl):
                """
                Höhe in m:
                  1) explizit aus height/building:height (Zahl im String)
                  2) Fallback: levels * 3.0 m
                """
                val = None
                # explizite Höhe
                if isinstance(h, str):
                    m = re.search(r"(\d+(?:\.\d+)?)", h)
                    if m:
                        try:
                            val = float(m.group(1))
                        except ValueError:
                            val = None
                elif isinstance(h, (int, float)) and not pd.isna(h):
                    val = float(h)

                # Fallback über Geschosse
                if val is None and pd.notna(lvl):
                    try:
                        val = float(lvl) * 3.0
                    except Exception:
                        pass
                return val

            # falls building:levels vorhanden, als levels nutzen
            if "building:levels" in g.columns and (
                "levels" not in g.columns or g["levels"].isna().all()
            ):
                g["levels"] = g["building:levels"]

            g["height_m"] = g.apply(
                lambda r: _parse_height_osm(r.get("height"), r.get("levels")),
                axis=1,
            )
            g["area_m2"] = g.geometry.area

            # --------------------------------------------------------
            # Baujahr aus start_date / year_built
            # --------------------------------------------------------
            def _parse_osm_year(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return pd.NA
                s = str(v)
                m = re.search(r"\b(\d{4})\b", s)
                if not m:
                    return pd.NA
                year = int(m.group(1))
                return year if 1200 <= year <= 2100 else pd.NA

            year_candidates = []
            for col in ("start_date", "year_built", "construction:year"):
                if col in g.columns:
                    year_candidates.append(col)

            if year_candidates:
                def _pick_year(row):
                    for c in year_candidates:
                        y = _parse_osm_year(row.get(c))
                        if not pd.isna(y):
                            return y
                    return pd.NA

                g["year"] = g.apply(_pick_year, axis=1).astype("Int64")
            else:
                g["year"] = pd.Series(pd.NA, index=g.index, dtype="Int64")

            # --------------------------------------------------------
            # source_id für spätere Zuordnung (wird zu OSM_source_id)
            # --------------------------------------------------------
            if "osm_id" in g.columns:
                g["source_id"] = g["osm_id"].astype(str)
            else:
                # Fallback: building_id (wird gleich gesetzt)
                g["source_id"] = g.index.astype(str)


            if "osm_id" in g.columns:
                g["building_id"] = g["osm_id"].astype(str)
            else:
                g["building_id"] = ["osm_" + str(i) for i in range(len(g))]

            keep = [
                "building_id",
                "geometry",
                "orig_usage",
                "unified_detailed",
                "is_residential",
                "height_m",
                "area_m2",
                "source_id",
                "year",
            ]
            keep = [c for c in keep if c in g.columns] + (["geometry"] if "geometry" not in keep else [])
            return g[keep]


        # ------------------------------------------------------------
        # 1) LOD2
        # ------------------------------------------------------------
        print("[LoD2] start")
        lod2_src = LoD2CityGMLSource()
        lod2_gdf = lod2_src.load(ctx)

        # Vereinheitlichung LOD2
        lod2_u = _unify_lod2(lod2_gdf)

        # QA: Quelle -> vereinheitlicht
        _write_qa_summary(
            step_name="lod2_unify",
            input_name="lod2_raw",
            input_gdf=lod2_gdf,
            output_name="lod2_unified",
            output_gdf=lod2_u,
            qa_dir=qa_dir,
        )

        # Aggregation zu Gebaeuden (Footprints & Teilflächen)
        min_area = float(getattr(settings, "min_lod2_area_m2", 25.0))
        lod2_build, lod2_surfaces = _agg_lod2_to_buildings(
            lod2_u,
            min_area_m2=min_area,
            target_epsg=target_epsg,
        )

        # QA: vereinheitlicht -> Gebaeude (Footprints)
        _write_qa_summary(
            step_name="lod2_agg_buildings",
            input_name="lod2_unified",
            input_gdf=lod2_u,
            output_name="lod2_build",
            output_gdf=lod2_build,
            qa_dir=qa_dir,
        )

        # QA: vereinheitlicht -> Teilflächen
        _write_qa_summary(
            step_name="lod2_surfaces",
            input_name="lod2_unified",
            input_gdf=lod2_u,
            output_name="lod2_surfaces",
            output_gdf=lod2_surfaces,
            qa_dir=qa_dir,
        )

        print("[LoD2] done")

        # C-1: Roh-CSV + .stat vor Mapping
        qu_dir = out_dir / "Qu"
        analyse_dir = out_dir / "Analyse"
        qu_dir.mkdir(parents=True, exist_ok=True)
        analyse_dir.mkdir(parents=True, exist_ok=True)

        try:
            lod2_u_no_geom = lod2_u.drop(columns=["geometry"], errors="ignore")
            lod2_before_csv = qu_dir / "lod2_before_mapping.csv"
            lod2_u_no_geom.to_csv(lod2_before_csv, index=False)
            print("[LoD2][C-1] Roh-CSV vor Mapping:", lod2_before_csv)
        except Exception as ex:
            print("[LoD2][C-1] WARN:", ex)

        try:
            profiler = DataProfiler()
            cov = profiler.coverage_report(lod2_u)
            lod2_stat_csv = analyse_dir / "lod2_before_mapping.stat.csv"
            cov.to_csv(lod2_stat_csv, index=False)
            print("[LoD2][C-1] Stat-Datei:", lod2_stat_csv)
        except Exception as ex:
            print("[LoD2][C-1] WARN(stat):", ex)

        try:
            lod2_u.head(30).drop(columns=["geometry"], errors="ignore").to_csv(
                out_dir / "lod2_debug_head.csv", index=False
            )
        except Exception:
            pass

        # ------------------------------------------------------------
        # 2) Basemap
        # ------------------------------------------------------------
        print("[Basemap] start")
        bm_buildings = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")
        bm_ctx = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

        try:
            bms = BasemapContextSource(
                BasemapCfg(
                    mvt_url_template=settings.basemap.mvt_url_template,
                    headers=getattr(settings.basemap, "headers", None),
                ),
                zoom=int(getattr(settings.basemap, "zoom", 15)),
                include_pois=True,
            )
            bm_buildings, bm_ctx, bm_meta = bms.load_basemap(
                bbox_25833=settings.region.bbox_25833,
                to_crs=target_epsg,
            )
        except Exception as ex:
            print("[Basemap] ERROR:", ex)

        try:
            if not bm_buildings.empty or (isinstance(bm_ctx, gpd.GeoDataFrame) and not bm_ctx.empty):
                inv = []
                if hasattr(bms, "_last_layers_inventory"):
                    inv = bms._last_layers_inventory
                if inv:
                    pd.DataFrame(inv).to_csv(out_dir / "basemap_layers_inventory.csv", index=False)

                _dedupe_columns(bm_buildings.drop(columns=["geometry"], errors="ignore")).head(50).to_csv(
                    out_dir / "basemap_debug_head.csv", index=False)
                if isinstance(bm_ctx, gpd.GeoDataFrame) and not bm_ctx.empty:
                    _dedupe_columns(bm_ctx.drop(columns=["geometry"], errors="ignore")).head(50).to_csv(
                        out_dir / "basemap_pois_head.csv", index=False)
        except Exception:
            pass

        bm_u = _unify_basemap(bm_buildings, bm_ctx) if not bm_buildings.empty else None

        if not lod2_build.empty and (not bm_buildings.empty or (
                isinstance(bm_ctx, gpd.GeoDataFrame) and not bm_ctx.empty)):
            bmap_classes_on_lod = map_basemap_classes(lod2_build, bm_buildings, bm_ctx, target_epsg=target_epsg)
        else:
            if "LOD_UNITID" in lod2_build.columns:
                idx = lod2_build["LOD_UNITID"].astype(str)
            else:
                idx = lod2_build.index
            bmap_classes_on_lod = pd.DataFrame(
                index=idx,
                columns=["BMAP_Gebaeudeflaeche_klasse", "BMAP_Gebaeudepunkt_klasse", "BMAP_Siedlungsflaeche_klasse"]
            )
        print("[Basemap] done")

        # ------------------------------------------------------------
        # 3) OSM
        # ------------------------------------------------------------
        print("[OSM] start")
        try:
            osm_src = OSMSource(overpass_url=getattr(settings, "overpass_url", None),
                               timeout=20, use_out_geom=True)
            osm_raw = osm_src.load(ctx)
        except Exception as ex:
            print("[OSM] ERROR:", ex)
            osm_raw = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{target_epsg}")

        osm_u = _unify_osm(osm_raw) if not osm_raw.empty else None
        print("[OSM] done")

        # ------------------------------------------------------------
        # 4) pro Quelle auf LOD2 mappen (Overlap)
        # ------------------------------------------------------------
        lod_on_lod = lod2_build.set_index("LOD_UNITID")[
            ["LOD_Nutzung_original", "LOD_Nutzung_vereinheitlicht", "LOD_NWGoderWG"]
        ].copy()
        lod_on_lod.columns = [f"LOD_{c.split('LOD_')[1]}" for c in lod_on_lod.columns]

        bmap_on_lod = attach_best_overlap(bm_u, lod2_build, "BMAP",
                                          target_epsg=target_epsg) if (
                bm_u is not None and not bm_u.empty) else pd.DataFrame(index=lod2_build["LOD_UNITID"])
        osm_on_lod = attach_best_overlap(osm_u, lod2_build, "OSM",
                                         target_epsg=target_epsg) if (
                osm_u is not None and not osm_u.empty) else pd.DataFrame(index=lod2_build["LOD_UNITID"])

        # ------------------------------------------------------------
        # 5) Compare-DF (LOD2 + Mappings)
        # ------------------------------------------------------------
        lod2_compare_cols = [
            c for c in [
                "LOD_UNITID",
                "building_id",
                "LOD_Nutzung_original",
                "LOD_Nutzung_vereinheitlicht",
                "LOD_NWGoderWG",
                "function_label",
                "is_func_unknown",
                "LOD_GebHoehe",
                "LOD_Stockwerke",
                "LOD_Grundflaeche_m2",
                "Dachform",
                "GebHoehe",
                "Stockwerke",
                "geometry",
            ] if c in lod2_build.columns
        ]

        compare_df = (
            lod2_build[lod2_compare_cols]
            .set_index("LOD_UNITID")
            .join(bmap_on_lod, how="left")
            .join(osm_on_lod, how="left")
            .join(bmap_classes_on_lod, how="left")
        )

        # 6) Finale Nutzung (Original + vereinheitlicht) mit neuer
        #    Priorität:
        #    1) LOD_Nutzung_original   (≠ 31001_9998)
        #    2) BMAP_Nutzung_original  (≠ "Nach Quellenlage nicht zu spezifizieren")
        #    3) BMAP_Siedlungsflaeche_klasse (falls nicht leer)
        #    4) OSM_Nutzung_original   (falls nicht leer)
        #    5) Fallback: Wohngebäude / Quelle = "Vorgabe"
        # ------------------------------------------------------------
        # Mapping auf vereinheitlichte Nutzungstypen
        osm_unified = {
            "building=apartments": "Mehrfamilienhäuser",
            "building=yes": "nicht zu spezifizieren",
            "building=industrial": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "<NA>": "nicht zu spezifizieren",
            "building=garages": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "building=allotment_house": "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
            "building=residential": "Mehrfamilienhäuser",
            "building=shed": "Technikgebäude (Ver- und Entsorgung)",
            "building=house": "Einfamilienhäuser",
            "building=garage": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "building=retail": "Handelsgebäude",
            "building=service": "Handelsgebäude",
            "building=hospital": "Gebaeude für Gesundheit und Pflege",
            "building=school": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "building=office": "Büro-, Verwaltungs- oder Amtsgebäude",
            "building=detached": "Einfamilienhäuser",
            "building=carport": "nicht zu spezifizieren",
            "building=civic": "Büro-, Verwaltungs- oder Amtsgebäude",
            "amenity=kindergarten": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "building=hotel": "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
            "building=semidetached_house": "Reihen- und Doppelhäuser",
            "building=commercial": "Handelsgebäude",
            "building=villa": "Einfamilienhäuser",
            "building=roof": "nicht zu spezifizieren",
            "amenity=police": "Büro-, Verwaltungs- oder Amtsgebäude",
            "building=kindergarten": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "building=warehouse": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "building=construction": "nicht zu spezifizieren",
            "amenity=place_of_worship": "Gebaeude für Kultur und Freizeit",
            "building=parking": "Technikgebäude (Ver- und Entsorgung)",
            "building=fire_station": "Büro-, Verwaltungs- oder Amtsgebäude",
            "amenity=restaurant": "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
            "building=train_station": "Verkehrsgebäude",
            "amenity=car_wash": "Technikgebäude (Ver- und Entsorgung)",
            "building=silo": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "amenity=post_depot": "Büro-, Verwaltungs- oder Amtsgebäude",
            "building=government": "Büro-, Verwaltungs- oder Amtsgebäude",
            "building=public": "Büro-, Verwaltungs- oder Amtsgebäude",
            "amenity=crematorium": "Gebaeude für Kultur und Freizeit",
            "building=university": "Gebaeude für Forschung und Hochschullehre",
            "building=toilets": "Technikgebäude (Ver- und Entsorgung)",
            "building=factory": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "amenity=bicycle_parking": "Technikgebäude (Ver- und Entsorgung)",
            "amenity=school": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "building=kiosk": "Handelsgebäude",
            "building=ruins": "nicht zu spezifizieren",
            "building=shelter": "nicht zu spezifizieren",
            "building=arch": "nicht zu spezifizieren",
            "building=bungalow": "Einfamilienhäuser",
            "building=yes": "Mehrfamilienhäuser",
            "building=roof": "Mehrfamilienhäuser",
        }

        func_unified = {
            "Gebaeude für öffentliche Zwecke": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Parlament": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Rathaus": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Gericht": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Kreisverwaltung": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Finanzamt": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Allgemein bildende Schule": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "Berufsbildende Schule": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "Hochschulgebäude (Fachhochschule,Universität)": "Gebaeude für Forschung und Hochschullehre",
            "Forschungsinstitut": "Gebaeude für Forschung und Hochschullehre",
            "Schloss": "Gebaeude für Kultur und Freizeit",
            "Theater, Oper": "Gebaeude für Kultur und Freizeit",
            "Konzertgebäude": "Gebaeude für Kultur und Freizeit",
            "Museum": "Gebaeude für Kultur und Freizeit",
            "Veranstaltungsgebäude": "Gebaeude für Kultur und Freizeit",
            "Burg, Festung": "Gebaeude für Kultur und Freizeit",
            "Gebaeude für religiöse Zwecke": "Gebaeude für Kultur und Freizeit",
            "Kirche": "Gebaeude für Kultur und Freizeit",
            "Synagoge": "Gebaeude für Kultur und Freizeit",
            "Kapelle": "Gebaeude für Kultur und Freizeit",
            "Gotteshaus": "Gebaeude für Kultur und Freizeit",
            "Moschee": "Gebaeude für Kultur und Freizeit",
            "Tempel": "Gebaeude für Kultur und Freizeit",
            "Kloster": "Gebaeude für Kultur und Freizeit",
            "Krankenhaus": "Gebaeude für Gesundheit und Pflege",
            "Polizei": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Feuerwehr": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Kaserne": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Justizvollzugsanstalt": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Trauerhalle": "Gebaeude für Kultur und Freizeit",
            "Bahnhofsgebäude": "Verkehrsgebäude",
            "Gebaeude für öffentliche Zwecke mit Wohnen": "Mehrfamilienhäuser",
            "Gebaeude für Erholungszwecke": "Gebaeude für Kultur und Freizeit",
            "Sport-, Turnhalle": "Sportgebäude",
            "Hallenbad": "Sportgebäude",
            "Gebaeude im Stadion": "Sportgebäude",
            "Bürogebäude": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Nach Quellenlage nicht zu spezifizieren": "nicht zu spezifizieren",
        }

        bmap_siedlung_unified = {
            "Wohnbaufläche": "Mehrfamilienhäuser",
            "Sportanlage": "Sportgebäude",
            "Soziales": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "Sicherheit und Ordnung": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Regierung und Verwaltung": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Industrie und Gewerbe": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "Handel und Dienstleistung": "Handelsgebäude",
            "Gesundheit, Kur": "Gebaeude für Gesundheit und Pflege",
            "Bildung und Wissenschaft": "Gebaeude für Forschung und Hochschullehre",
            "Fläche gemischter Nutzung": "Mehrfamilienhäuser", #ggf. anpassen
            "Park": "Gebaeude für Kultur und Freizeit",
            "Friedhof": "Gebaeude für Kultur und Freizeit",
            "Religiöse Einrichtung": "Gebaeude für Kultur und Freizeit",
            "Kleingarten": "Gebaeude für Kultur und Freizeit",
            "Gärtnerei": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "Kultur": "Gebaeude für Kultur und Freizeit",
            "Versorgungsanlage": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "Grünanlage": "Gebaeude für Kultur und Freizeit",
        }

        bmap_gebaeude_unified = {
            "Nach Quellenlage nicht zu spezifizieren": "nicht zu spezifizieren",
            "Gebaeude für Wirtschaft oder Gewerbe": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "Allgemein bildende Schule": "Schule, Kindertagesstätte und sonstiges Betreuungsgebäude",
            "Museum": "Gebaeude für Kultur und Freizeit",
            None: "nicht zu spezifizieren",
            "Hochschulgebäude (Fachhochschule, Universität)": "Gebaeude für Forschung und Hochschullehre",
            "Kapelle": "Gebaeude für Kultur und Freizeit",
            "Polizei": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Gebaeude für Bewirtung": "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
            "Krankenhaus": "Gebaeude für Gesundheit und Pflege",
            "Garage": "Produktions-, Werkstatt-, Lager- oder Betriebsgebäude",
            "Gebaeude für Beherbergung": "Beherbergungs- oder Unterbringungsgebäude, Gastronomie- oder Verpflegungsgebäude",
            "Kirche": "Gebaeude für Kultur und Freizeit",
            "Verwaltungsgebäude": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Gebaeude für Handel und Dienstleistungen": "Handelsgebäude",
            "Feuerwehr": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Freizeit und Vergnügungsstätte": "Gebaeude für Kultur und Freizeit",
            "Sport-, Turnhalle": "Sportgebäude",
            "Justizvollzugsanstalt": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Gebaeude zum Parken": "Technikgebäude (Ver- und Entsorgung)",
            "Theater, Oper": "Gebaeude für Kultur und Freizeit",
            "Rathaus": "Büro-, Verwaltungs- oder Amtsgebäude",
            "Freizeit- und Vergnügungsstätte": "Gebaeude für Kultur und Freizeit",
            "Gericht": "Büro-, Verwaltungs- oder Amtsgebäude",
        }

        def _norm_str(val: object) -> str:
            if val is None or pd.isna(val):
                return ""
            return str(val).strip()

        def _is_defined_lod(code: object) -> bool:
            s = _norm_str(code)
            return bool(s) and s != "31001_9998"

        def _is_defined_bmap_usage(val: object) -> bool:
            s = _norm_str(val)
            return bool(s) and s != "Nach Quellenlage nicht zu spezifizieren"

        def _is_defined_generic(val: object) -> bool:
            return bool(_norm_str(val))

        def _map_unified(src: str, orig: str, row: pd.Series) -> str:
            if src == "LOD2":
                label = _norm_str(row.get("function_label"))
                if label:
                    return func_unified.get(label, label)
                # falls keine Funktionsbezeichnung vorhanden ist
                return "nicht zu spezifizieren"
            if src in ("BMAP_Gebaeudeflaeche", "Basemap"):
                return bmap_gebaeude_unified.get(orig, bmap_gebaeude_unified.get(None, "nicht zu spezifizieren"))
            if src == "BMAP_Siedlungsflaeche":
                return bmap_siedlung_unified.get(orig, "nicht zu spezifizieren")
            if src == "OSM":
                return osm_unified.get(orig, "nicht zu spezifizieren")
            if src == "Vorgabe":
                # Default: Wohngebäude als Mehrfamilienhaus interpretieren
                return "Mehrfamilienhäuser"
            return "nicht zu spezifizieren"

        def _is_residential(unified: str) -> bool:
            u = _norm_str(unified).lower()
            return u in {
                "wohngebäude",
                "einfamilienhäuser",
                "mehrfamilienhäuser",
                "reihen- und doppelhäuser",
            }

        def _pick_final(row: pd.Series) -> tuple[str, str, str, str]:
            # 1) LOD2: Code, sofern nicht 31001_9998 (unbekannt)
            lod_code = row.get("LOD_Nutzung_original")
            if _is_defined_lod(lod_code):
                orig = _norm_str(lod_code)
                src = "LOD2"
                unified = _map_unified(src, orig, row)
                nwg = "WG" if _is_residential(unified) else "NWG"
                return orig, unified, src, nwg

            # 2) Basemap-Nutzung (Gebaeudefläche)
            bmap_usage = row.get("BMAP_Nutzung_original")
            if _is_defined_bmap_usage(bmap_usage):
                orig = _norm_str(bmap_usage)
                src = "BMAP_Nutzung"
                unified = _map_unified("Basemap", orig, row)
                nwg = "WG" if _is_residential(unified) else "NWG"
                return orig, unified, src, nwg

            # 3) Basemap-Siedlungsfläche (Klasse)
            bmap_siedl = row.get("BMAP_Siedlungsflaeche_klasse")
            if _is_defined_generic(bmap_siedl):
                orig = _norm_str(bmap_siedl)
                src = "BMAP_Siedlungsflaeche"
                unified = _map_unified(src, orig, row)
                nwg = "WG" if _is_residential(unified) else "NWG"
                return orig, unified, src, nwg

            # 4) OSM-Nutzung
            osm_orig = row.get("OSM_Nutzung_original")
            if _is_defined_generic(osm_orig):
                orig = _norm_str(osm_orig)
                src = "OSM"
                unified = _map_unified(src, orig, row)
                nwg = "WG" if _is_residential(unified) else "NWG"
                return orig, unified, src, nwg

            # 5) Fallback: Wohngebäude / Vorgabe
            orig = "Wohngebäude"
            src = "Vorgabe"
            unified = _map_unified(src, orig, row)
            nwg = "WG"
            return orig, unified, src, nwg

        final_vals = compare_df.apply(_pick_final, axis=1, result_type="expand")
        compare_df["Final_Nutzung_original"] = final_vals[0]
        compare_df["Final_Nutzung_vereinheitlicht"] = final_vals[1]
        compare_df["Final_Nutzung_Quelle"] = final_vals[2]
        compare_df["Final_NWGoderWG"] = final_vals[3]


        # ------------------------------------------------------------
        # 7) Compare-CSV schreiben (ohne je-Quelle *_Nutzung_vereinheitlicht)
        # ------------------------------------------------------------
        order_cols = [
            "building_id",
            "function_label", "is_func_unknown",
            "LOD_GebHoehe", "LOD_Stockwerke", "LOD_Grundflaeche_m2",
            "Dachform", "GebHoehe", "Stockwerke",
            "LOD_Nutzung_original", "LOD_NWGoderWG",
            "BMAP_Nutzung_original",
            "OSM_Nutzung_original",
            "BMAP_Gebaeudeflaeche_klasse",
            "BMAP_Gebaeudepunkt_klasse",
            "BMAP_Siedlungsflaeche_klasse",
            # neue Zeilen:
            "BMAP_year",
            "OSM_year",
            # Ende der neuen Zeilen
            "Final_Nutzung_original",
            "Final_Nutzung_vereinheitlicht",
            "Final_Nutzung_Quelle",
            "Final_NWGoderWG",
        ]

        compare_csv = compare_dir / "compare_lod2_bmap_osm.csv"
        compare_df.reset_index()[["LOD_UNITID"] + [c for c in order_cols if c in compare_df.columns]].to_csv(
            compare_csv, index=False
        )
        print("[COMPARE] geschrieben:", compare_csv)

        # zusätzlich Basemap-Buildings (mit unified) als eigenständige CSV
        if bm_u is not None and not bm_u.empty:
            bm_out_csv = out_dir / "basemap_buildings_with_unified.csv"
            _dedupe_columns(bm_u.drop(columns=["geometry"], errors="ignore")).to_csv(bm_out_csv, index=False)
            print("[Basemap] buildings CSV:", bm_out_csv)

        # ------------------------------------------------------------
        # 8) Finales GPKG (nur eine Geometriespalte, per LOD2)
        # ------------------------------------------------------------
        def _is_geom_series(s: pd.Series) -> bool:
            try:
                return str(getattr(s, "dtype", "")).lower() == "geometry" or \
                       s.map(lambda v: (v is None) or isinstance(v, BaseGeometry)).all()
            except Exception:
                return False

        attr_df = compare_df.copy()

        drop_usage_cols = [
            "LOD_Nutzung_vereinheitlicht",
            "BMAP_Nutzung_vereinheitlicht",
            "OSM_Nutzung_vereinheitlicht",
        ]
        for c in drop_usage_cols:
            if c in attr_df.columns:
                attr_df.drop(columns=[c], inplace=True)

        for c in list(attr_df.columns):
            try:
                if _is_geom_series(attr_df[c]):
                    attr_df.drop(columns=[c], inplace=True)
            except Exception:
                pass

        lod2_idxed = lod2_build.set_index("LOD_UNITID")
        attr_df = attr_df.reindex(lod2_idxed.index)

        gdf_out = gpd.GeoDataFrame(attr_df, geometry=lod2_idxed.geometry, crs=lod2_idxed.crs)

        for col in list(gdf_out.columns):
            if col != gdf_out.geometry.name:
                try:
                    if _is_geom_series(gdf_out[col]):
                        gdf_out[col] = gdf_out[col].apply(lambda g: None if g is None else g.wkt)
                except Exception:
                    pass

        gpkg_path = compare_dir / "nutzungsspezifikation_vereinheitlicht.gpkg"
        # Layer 1: Gebaeude-Footprints mit allen Attributen (Mapping, Final_Nutzung, …)
        gdf_out.to_file(gpkg_path, layer="buildings", driver="GPKG")
        print(f"[GPKG] geschrieben (Layer 'buildings'): {gpkg_path}")

        # Layer 2: alle LOD2-Teilflächen je LOD_UNITID (inkl. 3D-Geometrie)
        try:
            lod2_surfaces_out = lod2_surfaces.copy()

            # Zusätzliche Geometriespalten (z. B. __geom2d) in WKT/Text umwandeln,
            # damit nur EINE echte Geometry-Spalte übrig bleibt.
            for col in list(lod2_surfaces_out.columns):
                if col != lod2_surfaces_out.geometry.name:
                    try:
                        if _is_geom_series(lod2_surfaces_out[col]):
                            lod2_surfaces_out[col] = lod2_surfaces_out[col].apply(
                                lambda g: None if g is None else g.wkt
                            )
                    except Exception:
                        pass

            # (Optional) explizit __geom2d entfernen, wenn du sie nicht brauchst:
            # if "__geom2d" in lod2_surfaces_out.columns:
            #     lod2_surfaces_out = lod2_surfaces_out.drop(columns=["__geom2d"])

            lod2_surfaces_out.to_file(gpkg_path, layer="lod2_surfaces", driver="GPKG")
            print(f"[GPKG] ergänzt (Layer 'lod2_surfaces'): {gpkg_path}")
        except Exception as ex:
            print("[GPKG] WARN: Konnte Layer 'lod2_surfaces' nicht schreiben:", ex)

        # ------------------------------------------------------------
        # 9) Statistik-Dateien und Abbildungen erzeugen
        # ------------------------------------------------------------
        stats = AP1CSVStatistics()
        stats.run(
            csv_path=compare_csv,
            out_dir=out_dir / "Analyse"
        )

ap1_pipeline = AP1Pipeline()

