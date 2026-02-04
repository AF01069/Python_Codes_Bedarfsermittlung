"""HK-DE address import and building enrichment (HK-DE v5.2).

Dieses Modul ergänzt einen Gebäude-(Geo)DataFrame um Adressfelder aus einer
HK-DE Datei (v5.2). Es unterstützt:

1) Primär: ID-Join (building_id_col ↔ oid)
2) Fallback: räumliches Matching (Nearest Neighbour) über HK-UTM-Koordinaten

Wichtig: HK-DE hat i. d. R. ein Feld `zone` (32/33), das bestimmt, ob die
Koordinaten in EPSG:25832 oder EPSG:25833 liegen. Für Sachsen ist meist Zone 33
relevant. Dieses Modul wertet `zone` aus und matched zone-spezifisch.

Außerdem wird eine kurze Trefferstatistik ausgegeben, damit man sieht, ob der
Fallback tatsächlich greift.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Union, Tuple

import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HKAddressColumns:
    """Zielspaltennamen im Gebäude-Layer."""

    plz: str = "HK_postplz"
    post_city: str = "HK_postonm"
    post_city_add: str = "HK_postonmzus"
    post_district: str = "HK_postott"
    street: str = "HK_str"
    house_no: str = "HK_hnr"
    addr_add: str = "HK_adz"
    quality: str = "HK_qua"
    src: str = "HK_src"

    easting: str = "HK_ostwert"
    northing: str = "HK_nordwert"
    utm_zone: str = "HK_zone"

    match_method: str = "HK_match_method"
    match_dist_m: str = "HK_match_dist_m"
    match_flag: str = "HK_match"


def _find_hk_file(base_dir: Path) -> Optional[Path]:
    """Auto-Suche für HK-Datei.

    Reihenfolge:
    1) env var KWP_HK_PATH
    2) <base_dir>/Data
    3) <base_dir>/data
    """

    env = os.environ.get("KWP_HK_PATH")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p.exists():
            LOG.info("HK-DE path from env KWP_HK_PATH: %s", p)
            return p

    for d in (base_dir / "data" / "Adressen", base_dir / "Data", base_dir / "data"):
        if not d.exists():
            continue
        # bevorzugt typische Namensmuster
        patterns = [
            "adressen-*.txt",
            "adressen-*.csv",
            "*hk*adressen*.txt",
            "*hk*adressen*.csv",
            "*.txt",
            "*.csv",
        ]
        cand: list[Path] = []
        for pat in patterns:
            cand.extend([p for p in d.glob(pat) if p.is_file()])
        if not cand:
            continue
        # heuristik: hk + adresse + txt ist am wahrscheinlichsten
        cand = sorted(
            cand,
            key=lambda p: (
                -("hk" in p.name.casefold()),
                -("adresse" in p.name.casefold()),
                -(p.suffix.casefold() == ".txt"),
                p.name.casefold(),
            ),
        )
        LOG.info("HK-DE address file auto-detected: %s", cand[0])
        return cand[0]
    return None


def _as_path(p: Optional[Union[str, Path]], *, base_dir: Path) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = (base_dir / pp).resolve()
    return pp


def load_hk_addresses(
    base_dir: Path,
    *,
    hk_path: Optional[Union[str, Path]] = None,
    place_filter: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Liest HK-DE (v5.2) als DataFrame ein und setzt Index=oid.

    Wichtige Designentscheidungen:
    - dtype=str, damit PLZ führende Nullen behält.
    - `nba == 'L'` (Löschsätze) werden entfernt.
    - Mehrere Zeilen pro oid werden deterministisch auf "beste" Zeile reduziert.
    """

    hk_file = _as_path(hk_path, base_dir=base_dir) or _find_hk_file(base_dir)
    if hk_file is None or not hk_file.exists():
        raise FileNotFoundError(
            "HK-DE address file not found. Put it into <project>/Data/ (or /data) "
            "or pass --hk-path / set env var KWP_HK_PATH."
        )

    # HK-DE: CSV im UTF-8, Feldtrenner ';'
    try:
        df = pd.read_csv(
            hk_file,
            sep=";",
            encoding="utf-8",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            hk_file,
            sep=";",
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            low_memory=False,
        )

    # Pflichtfelder für unsere Zwecke
    required = {"oid", "postplz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"HK-DE file schema mismatch. Missing columns: {sorted(missing)}. "
            "Expected at least 'oid' and 'postplz'."
        )

    # optionale Felder ergänzen, damit downstream code robust bleibt
    optional = [
        "nba",
        "qua",
        "postonm",
        "postonmzus",
        "postott",
        "str",
        "hnr",
        "adz",
        "ostwert",
        "nordwert",
        "zone",
        "gmd",
    ]
    for c in optional:
        if c not in df.columns:
            df[c] = ""

    # optionaler Ortsfilter
    if place_filter:
        pf = [str(x).strip().casefold() for x in place_filter if str(x).strip()]
        if pf:
            before = len(df)
            mask = (
                df["postonm"].fillna("").astype(str).str.casefold().isin(pf)
                | df["gmd"].fillna("").astype(str).str.casefold().isin(pf)
            )
            df = df.loc[mask].copy()
            LOG.info("HK-DE place_filter reduced rows: %s -> %s", before, len(df))

    # Löschsätze entfernen
    before = len(df)
    df["nba"] = df["nba"].fillna("").astype(str)
    df = df[df["nba"].str.upper() != "L"].copy()
    if len(df) != before:
        LOG.info("HK-DE dropped deletions (nba='L'): %s -> %s", before, len(df))

    # Duplikate pro oid: beste Zeile wählen
    qual_rank = {"A": 30, "C": 20, "B": 10}  # häufige Reihenfolge in HK-Exports
    df["_score"] = (
        df["qua"].fillna("").astype(str).str.upper().map(qual_rank).fillna(0)
        + (df["postplz"].fillna("").astype(str).str.len() > 0).astype(int) * 5
        + (df["str"].fillna("").astype(str).str.len() > 0).astype(int) * 2
        + (df["hnr"].fillna("").astype(str).str.len() > 0).astype(int) * 1
    )
    df = df.sort_values(
        ["oid", "_score", "postplz", "postonm", "str", "hnr", "adz"],
        ascending=[True, False, True, True, True, True, True],
        kind="mergesort",
    )
    df = df.drop_duplicates("oid", keep="first").drop(columns=["_score"])

    df = df.set_index("oid", drop=True)
    df.index = df.index.astype(str)
    return df


def _ensure_geodataframe(buildings_df):
    """Konvertiert DataFrame -> GeoDataFrame (falls möglich).

    Wir nehmen die vorhandene Geometriespalte (typisch 'geometry') und CRS.
    """

    try:
        import geopandas as gpd  # noqa
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "GeoPandas is required for geometry fallback matching, but is not installed."
        ) from e

    if "geometry" not in buildings_df.columns:
        raise ValueError("buildings_df has no geometry column; geometry fallback not possible.")

    if hasattr(buildings_df, "crs"):
        return buildings_df

    # plain DataFrame but with geometry column
    gdf = gpd.GeoDataFrame(buildings_df, geometry="geometry")
    return gdf


def _parse_zone(val: str) -> Optional[int]:
    """Normiert Zone aus HK-DE.

    Akzeptiert z. B. '32', '33', '32N', 'UTM32', '', None.
    """

    if val is None:
        return None
    s = str(val).strip().upper()
    if not s:
        return None
    # extrahiere erste Zahlengruppe
    num = "".join(ch for ch in s if ch.isdigit())
    if not num:
        return None
    try:
        z = int(num)
    except ValueError:
        return None
    return z if z in (32, 33) else None


def _build_hk_points(hk: pd.DataFrame) -> Tuple["geopandas.GeoDataFrame", "geopandas.GeoDataFrame"]:
    """Erzeugt zwei GeoDataFrames: HK-Punkte in Zone 32 und Zone 33.

    HK-Index ist oid. Wir legen oid zusätzlich als Spalte '__oid__' ab, damit
    GeoPandas-Versionen ohne index_right-Spalte robust funktionieren.
    """

    import geopandas as gpd
    from shapely.geometry import Point

    tmp = hk.copy()
    tmp["__oid__"] = tmp.index.astype(str)

    # Koordinaten in numerisch
    tmp["ostwert"] = pd.to_numeric(tmp.get("ostwert", ""), errors="coerce")
    tmp["nordwert"] = pd.to_numeric(tmp.get("nordwert", ""), errors="coerce")

    # Zone normalisieren
    tmp["__zone__"] = tmp.get("zone", "").apply(_parse_zone)

    # Wenn `zone` fehlt (wie in deinem Beispiel-CSV: HK_zone leer), versuchen wir eine
    # robuste Heuristik über den Ostwert:
    # - UTM32: Ostwerte in Deutschland typ. ~ 500000–900000
    # - UTM33: Ostwerte in Deutschland typ. ~ 150000–500000
    # Für Sachsen liegen die Werte häufig klar in UTM33 (< 500000).
    mask_unknown = tmp["__zone__"].isna()
    tmp.loc[mask_unknown & (tmp["ostwert"] < 500000), "__zone__"] = 33
    tmp.loc[mask_unknown & (tmp["ostwert"] >= 500000), "__zone__"] = 32

    tmp = tmp.dropna(subset=["ostwert", "nordwert"]).copy()
    tmp["geometry"] = [Point(xy) for xy in zip(tmp["ostwert"], tmp["nordwert"])]

    z32 = tmp[tmp["__zone__"] == 32].copy()
    z33 = tmp[tmp["__zone__"] == 33].copy()

    g32 = gpd.GeoDataFrame(z32, geometry="geometry", crs="EPSG:25832")
    g33 = gpd.GeoDataFrame(z33, geometry="geometry", crs="EPSG:25833")
    return g32, g33


def _geometry_fallback_match(
    buildings_gdf,
    hk: pd.DataFrame,
    *,
    max_distance_m: float = 50.0,
) -> Tuple[pd.Series, pd.Series]:
    """Räumliches Matching Gebäude ↔ HK.

    Liefert:
    - oid_series: pro Gebäude (Index buildings) die gematchte HK-oid (oder <NA>)
    - dist_series: Distanz in Metern (oder <NA>)

    Ablauf:
    - Gebäude: wir matchen über Zentroiden (schnell und ausreichend für Adressen)
    - HK: Punktgeometrien aus ostwert/nordwert und zone (25832/25833)
    - sjoin_nearest pro Zone, dann wird pro Gebäude der bessere (kleinere) Treffer genommen

    Wichtig: CRS muss projektiv (Meter) sein, sonst sind Distanzen unsinnig.
    """

    import geopandas as gpd

    if buildings_gdf.crs is None:
        raise ValueError(
            "Buildings GeoDataFrame has no CRS. Please ensure you read the layer with CRS, "
            "or set it before calling address enrichment."
        )

    hk32, hk33 = _build_hk_points(hk)

    # Gebäude-Zentroiden (in ihrem CRS)
    b_cent = buildings_gdf.copy()
    b_cent["geometry"] = b_cent.geometry.centroid

    results = []

    def _join_zone(hk_gdf: gpd.GeoDataFrame, epsg: int):
        if hk_gdf.empty:
            return None
        b_z = b_cent.to_crs(epsg=epsg)
        joined = gpd.sjoin_nearest(
            b_z,
            hk_gdf[["__oid__", "geometry"]],
            how="left",
            max_distance=max_distance_m,
            distance_col="__dist__",
        )
        # sjoin_nearest liefert die rechte Spalte '__oid__' direkt; unabhängig von index_right.
        return joined[["__oid__", "__dist__"]]

    r32 = _join_zone(hk32, 25832)
    if r32 is not None:
        r32 = r32.rename(columns={"__oid__": "oid", "__dist__": "dist"})
        results.append(r32)

    r33 = _join_zone(hk33, 25833)
    if r33 is not None:
        r33 = r33.rename(columns={"__oid__": "oid", "__dist__": "dist"})
        results.append(r33)

    if not results:
        oid = pd.Series(pd.NA, index=buildings_gdf.index, dtype="string")
        dist = pd.Series(pd.NA, index=buildings_gdf.index, dtype="float")
        return oid, dist

    # Best-of-Zones: wähle je Gebäude den kleineren Distanzwert
    best = results[0].copy()
    for r in results[1:]:
        # wenn in best keine dist, nehme r; sonst min
        take_r = best["dist"].isna() | (r["dist"].notna() & (r["dist"] < best["dist"]))
        best.loc[take_r, "dist"] = r.loc[take_r, "dist"]
        best.loc[take_r, "oid"] = r.loc[take_r, "oid"]

    oid_series = best["oid"].astype("string")
    dist_series = pd.to_numeric(best["dist"], errors="coerce")

    return oid_series, dist_series


def enrich_buildings_with_hk_addresses(
    buildings_df,
    base_dir: Path,
    *,
    building_id_col: str = "LOD_UNITID",
    hk_path: Optional[Union[Path, str]] = None,
    place_filter: Optional[Iterable[str]] = None,
    cols: HKAddressColumns = HKAddressColumns(),
    max_distance_m: float = 50.0,
    verbose_print: bool = True,
):
    """Ergänzt Gebäude um HK-DE Adressen.

    Parameter
    ---------
    buildings_df:
        (Geo)DataFrame mit Gebäuden.
    base_dir:
        Projektwurzel (für Auto-Find der HK-Datei).
    building_id_col:
        Spaltenname der Gebäude-ID (Default: LOD_UNITID).
    hk_path:
        Optionaler Pfad zur HK-Datei.
    place_filter:
        Optionaler Ortsfilter, um HK vorab zu reduzieren (z. B. ['Chemnitz']).
    max_distance_m:
        Max. Distanz (Meter) für Geometrie-Fallback.
    verbose_print:
        Zusätzlich zu LOG auch per print() ausgeben (hilfreich in PyCharm Runs).

    Rückgabe
    --------
    buildings_df mit HK_* Spalten.
    """

    if building_id_col not in buildings_df.columns:
        raise KeyError(
            f"Building ID column '{building_id_col}' not found in buildings_df."
        )

    hk = load_hk_addresses(base_dir, hk_path=hk_path, place_filter=place_filter)

    # HK -> Zielspalten
    join_cols = {
        "postplz": cols.plz,
        "postonm": cols.post_city,
        "postonmzus": cols.post_city_add,
        "postott": cols.post_district,
        "str": cols.street,
        "hnr": cols.house_no,
        "adz": cols.addr_add,
        "qua": cols.quality,
        "ostwert": cols.easting,
        "nordwert": cols.northing,
        "zone": cols.utm_zone,
    }

    # vorhandene HK-Spalten entfernen, damit kein _x/_y Chaos entsteht
    target_cols = set(join_cols.values()) | {
        cols.src,
        cols.match_flag,
        cols.match_method,
        cols.match_dist_m,
    }
    buildings_out = buildings_df.drop(columns=[c for c in target_cols if c in buildings_df.columns], errors="ignore")

    # 1) ID Join
    b_key = buildings_out[building_id_col].astype(str)
    hk.index = hk.index.astype(str)

    hk_join = hk[list(join_cols.keys())].rename(columns=join_cols)
    hk_join[cols.src] = "HK-DE"

    out = buildings_out.assign(**{building_id_col: b_key}).merge(
        hk_join,
        how="left",
        left_on=building_id_col,
        right_index=True,
        validate="m:1",
    )

    out[cols.match_method] = "id"
    out[cols.match_dist_m] = pd.NA

    # match flag: matched wenn PLZ vorhanden
    out[cols.match_flag] = out[cols.plz].notna().map({True: "matched", False: "no_address"})

    matched_id = int((out[cols.match_flag] == "matched").sum())
    if verbose_print:
        print(
            f"HK-DE: {matched_id} matches using '{building_id_col}' ↔ oid."  # noqa: T201
        )
    LOG.info("HK-DE: %s matches using '%s' ↔ oid.", matched_id, building_id_col)

    # 2) Geometrie-Fallback, wenn ID-Join nichts gebracht hat
    if matched_id == 0:
        try:
            gbuildings = _ensure_geodataframe(buildings_out)
            if verbose_print:
                print(
                    f"HK-DE: 0 matches via ID. Trying geometry fallback (nearest within {max_distance_m:.1f} m)..."  # noqa: T201
                )

            oid_series, dist_series = _geometry_fallback_match(
                gbuildings,
                hk,
                max_distance_m=max_distance_m,
            )

            # nur da auffüllen, wo aktuell keine Adresse
            needs = out[cols.match_flag] == "no_address"
            has_oid = oid_series.notna()
            fill_mask = needs & has_oid.reindex(out.index, fill_value=False)

            if fill_mask.any():
                # HK-Attribute per oid nachziehen
                hk_fill = hk_join.reindex(oid_series[fill_mask].astype(str).values)
                hk_fill = hk_fill.reset_index(drop=True)

                # out-Teilmenge
                idx = out.index[fill_mask]

                for src_col, dst_col in join_cols.items():
                    # hk_fill hat bereits Zielspaltennamen
                    if dst_col in hk_fill.columns:
                        out.loc[idx, dst_col] = hk_fill[dst_col].values

                out.loc[idx, cols.src] = "HK-DE"
                out.loc[idx, cols.match_method] = "geom_nearest"
                out.loc[idx, cols.match_dist_m] = dist_series.loc[idx].values
                out.loc[idx, cols.match_flag] = "matched"

            # Statistik ausgeben
            matched_geom = int(((out[cols.match_method] == "geom_nearest") & (out[cols.match_flag] == "matched")).sum())
            matched_total = int((out[cols.match_flag] == "matched").sum())
            total = len(out)

            # Distanzstats nur für geom
            geom_d = pd.to_numeric(out.loc[out[cols.match_method] == "geom_nearest", cols.match_dist_m], errors="coerce")
            if len(geom_d) > 0:
                d_min = float(geom_d.min())
                d_med = float(geom_d.median())
                d_p95 = float(geom_d.quantile(0.95))
            else:
                d_min = d_med = d_p95 = float("nan")

            msg = (
                f"HK-DE geometry fallback summary: matched_geom={matched_geom}, "
                f"matched_total={matched_total}/{total}, "
                f"dist_m(min/median/p95)={d_min:.2f}/{d_med:.2f}/{d_p95:.2f}"
            )
            if verbose_print:
                print(msg)  # noqa: T201
            LOG.info(msg)

        except Exception as e:
            # Im Fehlerfall: keine harten Abbrüche; Adressen sind Zusatzinfo.
            if verbose_print:
                print(f"HK-DE geometry fallback failed: {e}")  # noqa: T201
            LOG.warning("HK-DE geometry fallback failed: %s", e, exc_info=True)
            # match_method bleibt 'id' mit 0 matches
            out[cols.match_method] = out[cols.match_method].where(out[cols.match_flag] == "matched", "none")

    else:
        # Wenn ID-Matches vorhanden, aber nicht alle: optional könnte man geom nur für no_address machen.
        # Für deinen aktuellen Fehlerfall (matched_id==0) reicht es, geom nur dann auszuführen.
        out[cols.match_method] = out[cols.match_method].where(out[cols.match_flag] == "matched", "none")

    # PLZ immer als string (führende Nullen)
    if cols.plz in out.columns:
        out[cols.plz] = out[cols.plz].astype("string")

    # finale Kurzstatistik
    matched_final = int((out[cols.match_flag] == "matched").sum())
    if verbose_print:
        print(f"HK-DE final: matched={matched_final}, no_address={len(out) - matched_final} (total={len(out)})")  # noqa: T201
    LOG.info("HK-DE final: matched=%s, no_address=%s (total=%s)", matched_final, len(out) - matched_final, len(out))

    return out
