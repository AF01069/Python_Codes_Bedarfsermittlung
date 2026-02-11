"""
Analyse-Utilities f?r Datenquellen und Geometrien.

Dieses Modul unterst?tzt die Qualit?tssicherung (QA), Statistiken
und Profiling der Eingangsdatens?tze (z. B. LoD2-Geometrien).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Iterable, Optional, Tuple
from shapely.errors import GEOSException

import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd



# ------------------------------------------------------------
# DataProfiler
# ------------------------------------------------------------

class DataProfiler:
    """
    Einfache Berichte zu Abdeckung / Verteilung je Attribut.
    """

    def coverage_report(self, gdf: gpd.GeoDataFrame, attrs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        F?hrt coverage_report aus.
        
        Args:
            gdf: Beschreibung.
            attrs: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        if gdf is None or gdf.empty:
            return pd.DataFrame(columns=["attribute", "count_non_null", "share_non_null"])
        if attrs is None:
            # typische Felder
            cand = [
                "usage_lod2", "usage_osm", "usage_basemap", "usage_ethos", "usage_ml",
                "final_use", "final_year_class", "levels", "height", "area"
            ]
            attrs = [c for c in cand if c in gdf.columns]
        rows = []
        n = len(gdf)
        for a in attrs:
            cnt = int(gdf[a].notna().sum())
            rows.append({"attribute": a, "count_non_null": cnt, "share_non_null": (100.0 * cnt / max(n, 1))})
        return pd.DataFrame(rows).sort_values("share_non_null", ascending=False)

    def distribution_report(self, gdf: gpd.GeoDataFrame, by: str = "source", attrs: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Verteilungen für ausgewählte Attribute. Falls 'by' nicht existiert, wird global gezählt.
        """
        if gdf is None or gdf.empty:
            return {}
        if attrs is None:
            attrs = [c for c in ["usage_lod2", "usage_osm", "usage_basemap", "usage_ethos", "usage_ml"] if c in gdf.columns]
        out: Dict[str, pd.DataFrame] = {}
        if by in gdf.columns:
            groups = gdf.groupby(by)
            for name, df in groups:
                for a in attrs:
                    s = df[a].value_counts(dropna=False).rename_axis(a).reset_index(name="count")
                    s["group"] = name
                    out[f"{a}__{name}"] = s
        else:
            for a in attrs:
                s = gdf[a].value_counts(dropna=False).rename_axis(a).reset_index(name="count")
                out[a] = s
        return out


# ------------------------------------------------------------
# CrossSourceComparator
# ------------------------------------------------------------

class CrossSourceComparator:
    """
    Vergleicht Nutzungstypen aus mehreren Quellen (LoD2/OSM/Basemap/ETHOS/ML).
    - Vereinheitlicht Labels (WG/NWG-Untertypen)
    - Bildet paarweise Confusion/Agreement-Matrizen
    - Liefert Mehrheitslabel, Agreement-Score je Gebäude
    - Identifiziert Hotspots großer Abweichungen
    """

    # Zentrales Mapping auf interne Kategorien
    # WGB-Spez: EFH/RH/DH/MFH; NWG: Büros, Handel, Schule, Gesundheit, Religion, Industrie, Logistik, Hotel, Sport, Verwaltung, Öffentlich, Freizeit
    _MAP = {
        # Wohngebäude aggregiert
        "wgb": "WGB",
        "wohnen": "WGB",
        "residential": "WGB",
        "apartment": "WGB",
        "wgb_efh": "WGB_EFH",
        "wgb_rh": "WGB_RH",
        "wgb_dh": "WGB_DH",
        "wgb_mfh": "WGB_MFH",
        # OSM/Basemap/Nebencodes
        "house": "WGB",
        "detached": "WGB_EFH",
        "semidetached_house": "WGB_DH",
        "terrace": "WGB_RH",
        "apartments": "WGB_MFH",
        # Nichtwohn
        "office": "NWG_OFFICE",
        "verwaltung": "NWG_OFFICE",
        "retail": "NWG_RETAIL",
        "shop": "NWG_RETAIL",
        "school": "NWG_SCHOOL",
        "kindergarten": "NWG_EDU",
        "university": "NWG_EDU",
        "hospital": "NWG_HEALTH",
        "clinic": "NWG_HEALTH",
        "church": "NWG_RELIGIOUS",
        "mosque": "NWG_RELIGIOUS",
        "synagogue": "NWG_RELIGIOUS",
        "industrial": "NWG_INDUSTRY",
        "factory": "NWG_INDUSTRY",
        "warehouse": "NWG_LOGISTIC",
        "logistics": "NWG_LOGISTIC",
        "hotel": "NWG_HOTEL",
        "sports": "NWG_SPORT",
        "stadium": "NWG_SPORT",
        "leisure": "NWG_LEISURE",
        "public": "NWG_PUBLIC",
        "administration": "NWG_ADMIN",
        # Basemap POI-Hints (bereits normiert in BasemapSource)
        "nwg_school": "NWG_SCHOOL",
        "nwg_health": "NWG_HEALTH",
        "nwg_retail": "NWG_RETAIL",
        "nwg_religious": "NWG_RELIGIOUS",
        "nwg_industry": "NWG_INDUSTRY",
        "nwg_logistic": "NWG_LOGISTIC",
        "nwg_hotel": "NWG_HOTEL",
        "nwg_sport": "NWG_SPORT",
        "nwg_office": "NWG_OFFICE",
        "nwg_admin": "NWG_ADMIN",
        "nwg_public": "NWG_PUBLIC",
        "nwg_leisure": "NWG_LEISURE",
    }

    def normalize_usage(self, s: pd.Series) -> pd.Series:
        """
        Normalisiert usage.
        
        Args:
            s: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        def _norm(x):
            if pd.isna(x):
                return None
            v = str(x).strip().lower()
            return self._MAP.get(v, v.upper() if v.startswith("WGB") or v.startswith("NWG") else None)
        return s.map(_norm)

    def _pair_confusion(self, a: pd.Series, b: pd.Series, normalize: bool) -> pd.DataFrame:
        df = pd.crosstab(a.fillna("NA"), b.fillna("NA"))
        if normalize:
            df = df.div(df.sum(axis=1).replace(0, np.nan), axis=0)
        return df

    def conflict_matrix(self, gdf: gpd.GeoDataFrame, source_cols: List[str], normalized: bool = False) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Erzeugt für jedes Quell-Paar (A,B) eine Confusion/Agreement-Matrix der Nutzungstypen.
        """
        res: Dict[Tuple[str, str], pd.DataFrame] = {}
        if gdf is None or gdf.empty or len(source_cols) < 2:
            return res

        df = gdf[source_cols].copy()
        # Normierung je Spalte
        for c in source_cols:
            df[c] = self.normalize_usage(df[c])

        for i in range(len(source_cols)):
            for j in range(i + 1, len(source_cols)):
                a, b = source_cols[i], source_cols[j]
                mat = self._pair_confusion(df[a], df[b], normalize=normalized)
                res[(a, b)] = mat
        return res

    def agreement_summary(self, gdf: gpd.GeoDataFrame, source_cols: List[str]) -> pd.DataFrame:
        """
        Mehrheitsentscheidung je Gebäude & globaler Agreement-Score.
        - majority_label: häufigstes normiertes Label je Zeile
        - agreement_ratio: Anteil der Quellen, die mit majority_label übereinstimmen (0..1)
        - n_sources: Zahl der nicht-leeren Quellen
        """
        if gdf is None or gdf.empty:
            return pd.DataFrame(columns=["n_sources", "agreement_ratio", "majority_label"])

        df = gdf[source_cols].copy()
        for c in source_cols:
            df[c] = self.normalize_usage(df[c])

        def _row_majority(row: pd.Series):
            vals = [v for v in row.tolist() if pd.notna(v)]
            if not vals:
                return None, 0, 0.0
            counts = pd.Series(vals).value_counts()
            maj = counts.index[0]
            n = len(vals)
            ratio = counts.iloc[0] / n
            return maj, n, ratio

        out = df.apply(lambda r: pd.Series(_row_majority(r), index=["majority_label", "n_sources", "agreement_ratio"]), axis=1)
        return out

    def disagreement_hotspots(self, gdf: gpd.GeoDataFrame, source_cols: List[str], top_n: int = 200) -> Optional[gpd.GeoDataFrame]:
        """
        Liefert die Gebäude mit der geringsten Übereinstimmung (kleinster agreement_ratio).
        Gibt GeoDataFrame (mit Geometrie) zurück – nützlich für Kartenkontrolle.
        """
        if gdf is None or gdf.empty or "geometry" not in gdf.columns:
            return None
        summ = self.agreement_summary(gdf, source_cols)
        if summ.empty:
            return None
        tmp = gpd.GeoDataFrame(pd.concat([gdf.reset_index(drop=True), summ.reset_index(drop=True)], axis=1), geometry="geometry", crs=gdf.crs)
        # nur Zeilen mit mind. 2 Quellen
        tmp = tmp[tmp["n_sources"] >= 2].copy()
        tmp.sort_values(["agreement_ratio", "n_sources"], ascending=[True, False], inplace=True)
        return tmp.head(top_n).copy()


# ------------------------------------------------------------
# TypingModels (Stub – hier einfache Heuristik/Delegation)
# ------------------------------------------------------------

class TypingModels:
    """
    Platzhalter für ML-Modelle. Für die AP1-Phase liefern wir eine robuste Heuristik,
    die vorhandene Quellen nutzt. Später durch echte Modelle (RF/XGBoost) ersetzen.
    """

    def predict_use(self, gdf: gpd.GeoDataFrame, features: List[str]) -> pd.Series:
        # Sehr einfache Regel:
        # 1) Wenn Basemap vorhanden → übernehmen
        # 2) Sonst OSM
        # 3) Sonst LoD2
        # 4) Fallback anhand Morphologie (Fläche/Compactness)
        """
        F?hrt predict_use aus.
        
        Args:
            gdf: Beschreibung.
            features: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        pick = []
        for _, r in gdf.iterrows():
            for c in ("usage_basemap", "usage_osm", "usage_lod2"):
                if c in gdf.columns and pd.notna(r.get(c)):
                    pick.append(r.get(c))
                    break
            else:
                area = r.get("area", r.geometry.area if r.get("geometry") is not None else np.nan)
                if pd.notna(area):
                    pick.append("WGB_MFH" if area >= 240 else "WGB_EFH")
                else:
                    pick.append(None)
        return pd.Series(pick, index=gdf.index)


# ------------------------------------------------------------
# FusionEngine (Skizze – später für AP2/Finalisierung)
# ------------------------------------------------------------

class FusionEngine:
    """
    Vereinheitlicht Nutzung/Baujahr nach Prioritätsregeln.
    (Für AP1 nicht kritisch – hier nur Stub für spätere Phasen.)
    """

    def fuse_use_and_year(self, gdf: gpd.GeoDataFrame, policy: Optional[Dict[str, Any]] = None) -> gpd.GeoDataFrame:
        """
        F?hrt fuse_use_and_year aus.
        
        Args:
            gdf: Beschreibung.
            policy: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        df = gdf.copy()
        # einfache Priorität Nutzung
        for a in ("usage_lod2", "usage_basemap", "usage_osm", "usage_ml"):
            if a in df.columns:
                df["final_use"] = df.get("final_use").fillna(df[a]) if "final_use" in df.columns else df[a]
        # Baujahr analog (falls vorhanden)
        for a in ("year_ethos", "year_lod2", "year_osm"):
            if a in df.columns:
                df["final_year"] = df.get("final_year").fillna(df[a]) if "final_year" in df.columns else df[a]
        return df


# ------------------------------------------------------------
# QA/QC Reporter
# ------------------------------------------------------------

class QAQCReporter:
    """
    Schreibt tabellarische QA-Ergebnisse (CSV/Parquet). Plot-Generierung ist optional
    und kann projektseitig ergänzt werden.
    """

    def export_metrics(self, gdf: gpd.GeoDataFrame, out_dir: Path, extra_tables: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Exportiert metrics.
        
        Args:
            gdf: Beschreibung.
            out_dir: Beschreibung.
            extra_tables: Beschreibung.
        
        Returns:
            None
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Grundlegende Kenngrößen
        base_cols = [c for c in ["usage_lod2", "usage_osm", "usage_basemap", "usage_ethos", "usage_ml", "final_use"] if c in gdf.columns]
        summary = {}
        for c in base_cols:
            vc = gdf[c].value_counts(dropna=False)
            summary[c] = vc

        # Speichern
        with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
            for c, vc in summary.items():
                f.write(f"[{c}]\n")
                f.write(vc.to_string())
                f.write("\n\n")

        if extra_tables:
            for name, df in extra_tables.items():
                p = out_dir / f"{name}.csv"
                try:
                    df.to_csv(p, index=False)
                except Exception:
                    # Notfall: als txt
                    df.to_csv(out_dir / f"{name}.txt", index=False, sep=";")


# ------------------------------------------------------------
# AP1 CSV-Statistik (Endprodukt ap1_pipeline)
# ------------------------------------------------------------

class AP1CSVStatistics:
    """
    Erzeugt für die von der ap1-Pipeline erzeugte CSV-Datei (Gebäude-Tabelle)
    eine statistische Auswertung als Textdatei sowie einfache Grafiken je Spalte.

    Fokus:
    - Kategoriale Verteilungen (absolut, prozentual) getrennt nach Datenquelle
      (LoD2, Basemap, OSM) für:
        * LOD_Nutzung_original
        * LOD_Nutzung_vereinheitlicht
        * BMAP_Nutzung_original
        * OSM_Nutzung_original
    - Numerische Kennwerte für:
        * LOD_Stockwerke
        * LOD_GebHoehe
        * Final_Nutzung_vereinheitlicht

    ID-Spalten (eindeutige Identifikatoren) werden automatisch erkannt und
    nicht ausgewertet.
    """

    #: Standard-Kategorien-Spalten (werden nur verwendet, falls in der CSV vorhanden)
    DEFAULT_CATEGORICAL_COLS: List[str] = [
        "LOD_Nutzung_original",
        "LOD_Nutzung_vereinheitlicht",
        "BMAP_Nutzung_original",
        "OSM_Nutzung_original",
        "BMAP_Siedlungsflaeche_klasse",
        "Final_Nutzung_Quelle",
        "Final_NWGoderWG",
        "Final_Nutzung_vereinheitlicht",
    ]

    #: Standard-numerische Spalten
    DEFAULT_NUMERIC_COLS: List[str] = [
        "LOD_Stockwerke",
        "LOD_GebHoehe",
        "LOD_Grundflaeche_m2",
    ]

    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
        id_cols: Optional[List[str]] = None,
    ) -> None:
        self.categorical_cols = categorical_cols or list(self.DEFAULT_CATEGORICAL_COLS)
        self.numeric_cols = numeric_cols or list(self.DEFAULT_NUMERIC_COLS)
        # explizit vom Nutzer angegebene ID-Spalten (werden zusätzlich zu den automatisch
        # erkannten ignoriert)
        self.id_cols = id_cols or []

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def run(self, csv_path: Path | str, out_dir: Path | str) -> Path:
        """
        Führt die CSV-Analyse aus.

        - Liest die ap1-CSV-Datei ein
        - erkennt ID-Spalten und schließt sie von der Auswertung aus
        - schreibt einen Textbericht mit den wichtigsten Statistiken
        - erzeugt je ausgewerteter Spalte eine PNG-Grafik im Ausgabeverzeichnis

        Parameters
        ----------
        csv_path:
            Pfad zur Eingabe-CSV (ap1-Ergebnis).
        out_dir:
            Ausgabeverzeichnis für Bericht und Grafiken.

        Returns
        -------
        Path
            Pfad zur erzeugten Textdatei mit der Auswertung.
        """
        csv_path = Path(csv_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

        df = pd.read_csv(csv_path)

        # ID-Spalten bestimmen
        auto_id_cols = self._detect_id_columns(df)
        all_id_cols = sorted({*auto_id_cols, *self.id_cols})

        # nur existierende Spalten auswerten
        cat_cols = [c for c in self.categorical_cols if c in df.columns and c not in all_id_cols]
        num_cols = [c for c in self.numeric_cols if c in df.columns and c not in all_id_cols]

        # Textbericht
        report_path = out_dir / f"{csv_path.stem}_statistik.txt"
        with report_path.open("w", encoding="utf-8") as f:
            self._write_header(f, df, csv_path)
            self._write_id_info(f, all_id_cols)

            if cat_cols:
                self._write_categorical_section(f, df, cat_cols)
            else:
                f.write("\nKeine der erwarteten kategorialen Nutzungsspalten wurde in der CSV gefunden.\n")

            if num_cols:
                self._write_numeric_section(f, df, num_cols)
            else:
                f.write("\nKeine der erwarteten numerischen Spalten wurde in der CSV gefunden.\n")

        # Grafiken erzeugen
        self._create_plots(df, cat_cols, num_cols, out_dir, all_id_cols)

        return report_path

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------
    def _detect_id_columns(self, df: pd.DataFrame, uniqueness_threshold: float = 0.95) -> List[str]:
        """
        Heuristik zur Erkennung von ID-Spalten:
        - sehr hohe Anzahl unterschiedlicher Werte (≈ jede Zeile einzigartig)
        - Spaltenname enthält typische ID-Bezeichner
        """
        id_like_names = {c for c in df.columns if "id" in c.lower() or c.lower() in {"uuid", "unitid"}}
        auto: List[str] = []

        n = len(df)
        if n == 0:
            return list(id_like_names)

        for col in df.columns:
            unique_count = df[col].nunique(dropna=False)
            if unique_count == 0:
                continue
            # nahezu eindeutige Spalte
            if unique_count / n >= uniqueness_threshold:
                auto.append(col)
            # explizite ID-Bezeichner
            if col in id_like_names:
                auto.append(col)

        return sorted(set(auto))

    def _write_header(self, f, df: pd.DataFrame, csv_path: Path) -> None:
        f.write("AP1 – CSV-Statistikbericht\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Datei: {csv_path}\n")
        f.write(f"Anzahl Gebäude (Zeilen): {len(df)}\n")
        f.write(f"Anzahl Attribute (Spalten): {len(df.columns)}\n\n")

    def _write_id_info(self, f, id_cols: List[str]) -> None:
        if id_cols:
            f.write("Nicht ausgewertete ID-/Schlüsselspalten:\n")
            for col in id_cols:
                f.write(f"  - {col}\n")
        else:
            f.write("Keine eindeutigen ID-Spalten erkannt, alle Spalten prinzipiell auswertbar.\n")
        f.write("\n")

    def _write_categorical_section(self, f, df: pd.DataFrame, cat_cols: List[str]) -> None:
        # Gruppierung nach Datenquelle, nur zur schöneren Gliederung
        groups = {
            "LoD2": ["LOD_Nutzung_original", "LOD_Nutzung_vereinheitlicht"],
            "Basemap": ["BMAP_Nutzung_original"],
            "OSM": ["OSM_Nutzung_original"],
        }

        f.write("--- Kategoriale Nutzungsspezifikationen (Verteilungen) ---\n\n")
        for group_name, cols in groups.items():
            cols = [c for c in cols if c in cat_cols]
            if not cols:
                continue
            f.write(f"### Datenquelle: {group_name}\n\n")
            for col in cols:
                s = df[col]
                vc = s.value_counts(dropna=False)
                total = int(s.shape[0]) or 1

                f.write(f"Spalte: {col}\n")
                f.write(f"  Nicht-leere Einträge: {int(s.notna().sum())} ({s.notna().mean() * 100:5.1f} %)\n")
                f.write("  Verteilung (absolut / %):\n")
                for val, count in vc.items():
                    label = "<NA>" if pd.isna(val) else str(val)
                    share = 100.0 * count / total
                    f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
                f.write("\n")

        # ggf. weitere kategoriale Spalten, die nicht in den bekannten Gruppen sind
        other_cols = [c for c in cat_cols if c not in {
            "LOD_Nutzung_original",
            "LOD_Nutzung_vereinheitlicht",
            "BMAP_Nutzung_original",
            "OSM_Nutzung_original",
        }]
        if other_cols:
            f.write("### Weitere kategoriale Spalten\n\n")
            for col in other_cols:
                s = df[col]
                vc = s.value_counts(dropna=False)
                total = int(s.shape[0]) or 1

                f.write(f"Spalte: {col}\n")
                f.write(f"  Nicht-leere Einträge: {int(s.notna().sum())} ({s.notna().mean() * 100:5.1f} %)\n")
                f.write("  Verteilung (absolut / %):\n")
                for val, count in vc.items():
                    label = "<NA>" if pd.isna(val) else str(val)
                    share = 100.0 * count / total
                    f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
                f.write("\n")

    def _write_numeric_section(self, f, df: pd.DataFrame, num_cols: List[str]) -> None:
        f.write("--- Numerische Kennwerte ---\n\n")
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.dropna().empty:
                f.write(f"Spalte: {col}\n  Keine numerisch auswertbaren Werte vorhanden.\n\n")
                continue

            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

            f.write(f"Spalte: {col}\n")
            f.write(f"  Nicht-leere Einträge: {int(s.notna().sum())} ({s.notna().mean() * 100:5.1f} %)\n")
            f.write(f"  Minimum: {desc['min']:.3f}\n")
            f.write(f"  5. Perzentil: {desc['5%']:.3f}\n")
            f.write(f"  25. Perzentil: {desc['25%']:.3f}\n")
            f.write(f"  Median (50%): {desc['50%']:.3f}\n")
            f.write(f"  75. Perzentil: {desc['75%']:.3f}\n")
            f.write(f"  95. Perzentil: {desc['95%']:.3f}\n")
            f.write(f"  Maximum: {desc['max']:.3f}\n")
            f.write(f"  Mittelwert: {desc['mean']:.3f}\n")
            f.write(f"  Standardabweichung: {desc['std']:.3f}\n\n")

    def _create_plots(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        out_dir: Path,
        id_cols: List[str],
    ) -> None:
        if not cat_cols and not num_cols:
            return

        import matplotlib.pyplot as plt  # nur importieren, falls wir wirklich plotten

        # Kategoriale Spalten: Balkendiagramm (Top-N, Rest aggregiert)
        for col in cat_cols:
            s = df[col].copy()
            if s.dropna().empty:
                continue

            vc = s.fillna("<NA>").value_counts()
            if vc.empty:
                continue

            # Bei sehr vielen Kategorien: Top 25 + Rest aggregieren
            if len(vc) > 25:
                top = vc.head(25)
                rest_sum = int(vc.iloc[25:].sum())
                top.loc["Weitere"] = rest_sum
                vc = top

            plt.figure(figsize=(10, 6))
            vc.plot(kind="bar")
            plt.title(f"Verteilung {col}")
            plt.ylabel("Anzahl Gebäude")
            plt.xticks(rotation=45, ha="right")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{col}_verteilung.png"
            plt.savefig(out_file, dpi=150)
            plt.close()

        # Numerische Spalten: Histogramm
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue

            plt.figure(figsize=(8, 5))
            plt.hist(s, bins=30)
            plt.title(f"Histogramm {col}")
            plt.xlabel(col)
            plt.ylabel("Anzahl Gebäude")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{col}_hist.png"
            plt.savefig(out_file, dpi=150)
            plt.close()

class AP1EnrichCSVStatistics:
    """
    Statistische Auswertung der von ap1_enrich erzeugten CSV-Datei im Unterordner 'zensus'.

    Fokus:
    - Allgemeine Angaben:
        * Anzahl Gebäude, Anzahl Attribute
        * (falls GPKG angegeben) Ausdehnung (BBOX) und Koordinatensystem
        * (falls in Attributen vorhanden) Land / Bundesland / Kommune (ags, Gemeindename, etc.)

    - Geometrische/Gebäudekennwerte (LOD2-basiert):
        * LOD_Grundflaeche_m2
        * LOD_GebHoehe
        * LOD_Stockwerke
        * Final_Stockwerke_schaetzung
        * Final_Nutzflaeche_m2

    - Nutzungstyp-Statistiken:
        * LOD_Nutzung_original / LOD_Nutzung_vereinheitlicht
        * BMAP_Nutzung_original / BMAP_Siedlungsflaeche_klasse
        * OSM_Nutzung_original
        * Final_Nutzung_vereinheitlicht / Final_NWGoderWG / Final_Nutzung_Quelle
        * jeweils Verteilungen, Anteile fehlender Werte

    - Baujahrsstatistiken:
        * DIVIS_Baujahr, DIVIS_Baujahr_Extrakt
        * OBAT_Baujahr_Mitte
        * Final_Baujahr_Mitte, Final_Baujahrklasse, Final_Baujahr_quelle
        * Verteilungen und Anteile fehlender Werte

    - Heizsysteme (Zensus-basiert):
        * Final_Energietraeger, Final_Heizungsart, Final_Heizsystem_quelle
        * Anteile fehlender Werte

    Optional werden einfache Histogramme/Balkendiagramme zu ausgewählten
    numerischen und kategorialen Spalten erzeugt (analog zu AP1CSVStatistics).
    """

    # typische Spaltennamen für Region/Verwaltungseinheit
    REGION_COL_CANDS = [
        "ags", "AGS", "GEMEINDE", "GEMEINDENAME", "Gemeindename",
        "STADT", "Stadt", "Ort", "Ortsname",
        "BUNDESLAND", "Bundesland", "Land"
    ]

    # Geometrie-/Gebäudekennwerte
    GEOMETRY_NUMERIC_COLS = [
        "LOD_Grundflaeche_m2",
        "LOD_GebHoehe",
        "LOD_Stockwerke",
        "Final_Stockwerke_schaetzung",
        "Final_Nutzflaeche_m2",
    ]

    # Nutzungsspalten
    USAGE_CAT_COLS = [
        "LOD_Nutzung_original",
        "LOD_Nutzung_vereinheitlicht",
        "BMAP_Nutzung_original",
        "BMAP_Siedlungsflaeche_klasse",
        "OSM_Nutzung_original",
        "Final_NWGoderWG",
        "Final_Nutzung_vereinheitlicht",
        "Final_Nutzung_Quelle",
    ]

    # Baujahr-/Alters-Spalten
    YEAR_NUMERIC_COLS = [
        "DIVIS_Baujahr",
        "DIVIS_Baujahr_Extrakt",
        "OBAT_Baujahr_Mitte",
        "Final_Baujahr_Mitte",
    ]
    YEAR_CATEGORICAL_COLS = [
        "Final_Baujahrklasse",
        "Final_Baujahr_quelle",
    ]

    # Heizsystem-Spalten
    HEATING_CAT_COLS = [
        "Final_Energietraeger",
        "Final_Heizungsart",
        "Final_Heizsystem_quelle",
    ]

    def __init__(
        self,
        geometry_numeric_cols: Optional[List[str]] = None,
        usage_cat_cols: Optional[List[str]] = None,
        year_numeric_cols: Optional[List[str]] = None,
        year_cat_cols: Optional[List[str]] = None,
        heating_cat_cols: Optional[List[str]] = None,
        id_cols: Optional[List[str]] = None,
    ) -> None:
        self.geometry_numeric_cols = geometry_numeric_cols or list(self.GEOMETRY_NUMERIC_COLS)
        self.usage_cat_cols = usage_cat_cols or list(self.USAGE_CAT_COLS)
        self.year_numeric_cols = year_numeric_cols or list(self.YEAR_NUMERIC_COLS)
        self.year_cat_cols = year_cat_cols or list(self.YEAR_CATEGORICAL_COLS)
        self.heating_cat_cols = heating_cat_cols or list(self.HEATING_CAT_COLS)
        self.id_cols = id_cols or []

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def run(
        self,
        csv_path: Path | str,
        out_dir: Path | str,
        gpkg_path: Optional[Path | str] = None,
    ) -> Path:
        """
        Führt die Analyse für die ap1_enrich-Zensus-CSV aus.

        Parameters
        ----------
        csv_path:
            Pfad zur Zensus-CSV, z. B. out/zensus/ap1_buildings_enriched_zensus.csv
        out_dir:
            Ausgabeverzeichnis für Bericht und Grafiken.
        gpkg_path:
            Optionaler Pfad zur zugehörigen GeoPackage-Datei
            (out/zensus/ap1_buildings_enriched_zensus.gpkg). Wenn gesetzt,
            werden Ausdehnung und CRS aus der Geometrie ermittelt.

        Returns
        -------
        Path
            Pfad zur erzeugten Textdatei mit der Auswertung.
        """
        csv_path = Path(csv_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

        df = pd.read_csv(csv_path)

        gdf: Optional[gpd.GeoDataFrame] = None
        if gpkg_path is not None:
            gpkg_path = Path(gpkg_path)
            if gpkg_path.exists():
                try:
                    # Layer-Name aus ap1_enrich: "buildings_zensus"
                    gdf = gpd.read_file(gpkg_path, layer="buildings_zensus")
                except Exception:
                    # Notfall: Layername nicht bekannt → ersten Layer laden
                    try:
                        gdf = gpd.read_file(gpkg_path)
                    except Exception:
                        gdf = None

        auto_id_cols = self._detect_id_columns(df)
        all_id_cols = sorted({*auto_id_cols, *self.id_cols})

        # Tatsächlich vorhandene Spalten einschränken
        geom_num_cols = [c for c in self.geometry_numeric_cols if c in df.columns and c not in all_id_cols]
        usage_cat_cols = [c for c in self.usage_cat_cols if c in df.columns and c not in all_id_cols]
        year_num_cols = [c for c in self.year_numeric_cols if c in df.columns and c not in all_id_cols]
        year_cat_cols = [c for c in self.year_cat_cols if c in df.columns and c not in all_id_cols]
        heating_cat_cols = [c for c in self.heating_cat_cols if c in df.columns and c not in all_id_cols]

        # Textbericht
        report_path = out_dir / f"{csv_path.stem}_enrich_statistik.txt"
        with report_path.open("w", encoding="utf-8") as f:
            self._write_header(f, df, csv_path, gdf)
            self._write_region_info(f, df)
            self._write_id_info(f, all_id_cols)

            self._write_section_geometry_stats(f, df, geom_num_cols)
            self._write_section_usage_stats(f, df, usage_cat_cols)
            self._write_section_year_stats(f, df, year_num_cols, year_cat_cols)
            self._write_section_heating_stats(f, df, heating_cat_cols)

        # Optional: einfache Plots erzeugen
        self._create_plots(df, geom_num_cols, usage_cat_cols, year_num_cols, heating_cat_cols, out_dir, all_id_cols)

        return report_path

    # ------------------------------------------------------------------
    # Interne Helfer
    # ------------------------------------------------------------------
    def _detect_id_columns(self, df: pd.DataFrame, uniqueness_threshold: float = 0.95) -> List[str]:
        """
        Heuristik zur Erkennung von ID-Spalten:
        - sehr hohe Anzahl unterschiedlicher Werte (≈ jede Zeile einzigartig)
        - Spaltenname enthält typische ID-Bezeichner
        """
        id_like_names = {
            c for c in df.columns
            if "id" in c.lower() or c.lower() in {"uuid", "unitid", "building_id", "LOD_UNITID"}
        }
        auto: List[str] = []

        n = len(df)
        if n == 0:
            return list(id_like_names)

        for col in df.columns:
            unique_count = df[col].nunique(dropna=False)
            if unique_count == 0:
                continue
            if unique_count / n >= uniqueness_threshold:
                auto.append(col)
            if col in id_like_names:
                auto.append(col)

        return sorted(set(auto))

    def _write_header(
        self,
        f,
        df: pd.DataFrame,
        csv_path: Path,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        f.write("AP1 ENRICH – Zensus/Enrichment-Statistikbericht\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Datei (CSV): {csv_path}\n")
        f.write(f"Anzahl Gebäude (Zeilen): {len(df)}\n")
        f.write(f"Anzahl Attribute (Spalten): {len(df.columns)}\n")

        if gdf is not None and not gdf.empty and "geometry" in gdf.columns:
            try:
                minx, miny, maxx, maxy = gdf.total_bounds
                f.write("\nGeometrische Ausdehnung (aus GPKG):\n")
                f.write(f"  BBOX minx, miny, maxx, maxy: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}\n")
                f.write(f"  Breite  (dx): {maxx - minx:.3f} m\n")
                f.write(f"  Höhe   (dy): {maxy - miny:.3f} m\n")
                if gdf.crs is not None:
                    f.write(f"Koordinatensystem (CRS): {gdf.crs}\n")
            except Exception:
                f.write("\nHinweis: Ausdehnung/CRS konnten aus dem GPKG nicht ermittelt werden.\n")
        f.write("\n")

    def _write_region_info(self, f, df: pd.DataFrame) -> None:
        f.write("--- Regionale Metadaten (falls vorhanden) ---\n\n")
        found = False
        for col in self.REGION_COL_CANDS:
            if col in df.columns:
                s = df[col]
                vc = s.value_counts(dropna=False)
                total = len(s) or 1
                f.write(f"Spalte: {col}\n")
                f.write(f"  Nicht-leere Einträge: {int(s.notna().sum())} ({s.notna().mean() * 100:5.1f} %)\n")
                f.write("  Werte (Top 20):\n")
                for val, count in vc.head(20).items():
                    label = "<NA>" if pd.isna(val) else str(val)
                    share = 100.0 * count / total
                    f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
                f.write("\n")
                found = True
        if not found:
            f.write("Keine typischen Spalten für Land/Bundesland/Kommune gefunden.\n\n")

    def _write_id_info(self, f, id_cols: List[str]) -> None:
        if id_cols:
            f.write("--- Nicht ausgewertete ID-/Schlüsselspalten ---\n")
            for col in id_cols:
                f.write(f"  - {col}\n")
        else:
            f.write("--- Keine eindeutigen ID-Spalten erkannt ---\n")
        f.write("\n")

    # --------------------------------------------------------------
    # Geometrie- und Gebäudekennwerte
    # --------------------------------------------------------------
    def _write_section_geometry_stats(
        self,
        f,
        df: pd.DataFrame,
        geom_num_cols: List[str],
    ) -> None:
        f.write("--- Geometrische/Gebäudekennwerte (LOD2/abgeleitet) ---\n\n")
        if not geom_num_cols:
            f.write("Keine der erwarteten Geometrie-/Gebäudekennwerte vorhanden.\n\n")
            return

        for col in geom_num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            n_total = len(s) or 1
            n_non_null = int(s.notna().sum())
            share_non_null = s.notna().mean() * 100.0

            f.write(f"Spalte: {col}\n")
            f.write(f"  Nicht-leere Einträge: {n_non_null} ({share_non_null:5.1f} %)\n")
            if s.dropna().empty:
                f.write("  Keine numerisch auswertbaren Werte vorhanden.\n\n")
                continue

            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            mode_vals = s.mode(dropna=True)
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan

            f.write(f"  Minimum: {desc['min']:.3f}\n")
            f.write(f"  5. Perzentil: {desc['5%']:.3f}\n")
            f.write(f"  25. Perzentil: {desc['25%']:.3f}\n")
            f.write(f"  Median (50%): {desc['50%']:.3f}\n")
            f.write(f"  75. Perzentil: {desc['75%']:.3f}\n")
            f.write(f"  95. Perzentil: {desc['95%']:.3f}\n")
            f.write(f"  Maximum: {desc['max']:.3f}\n")
            f.write(f"  Mittelwert: {desc['mean']:.3f}\n")
            f.write(f"  Standardabweichung: {desc['std']:.3f}\n")
            if np.isfinite(mode_val):
                f.write(f"  Häufigster Wert (Mode): {mode_val:.3f}\n")
            else:
                f.write("  Häufigster Wert (Mode): n/a\n")
            f.write("\n")

    # --------------------------------------------------------------
    # Nutzungstyp-Statistik
    # --------------------------------------------------------------
    def _write_section_usage_stats(
        self,
        f,
        df: pd.DataFrame,
        usage_cat_cols: List[str],
    ) -> None:
        f.write("--- Nutzungstyp-Spezifikationen ---\n\n")
        if not usage_cat_cols:
            f.write("Keine der erwarteten Nutzungsspalten vorhanden.\n\n")
            return

        # Gruppenstruktur zur Gliederung
        groups = {
            "LoD2": ["LOD_Nutzung_original", "LOD_Nutzung_vereinheitlicht"],
            "Basemap (Gebäude)": ["BMAP_Nutzung_original"],
            "Basemap (Siedlungsfläche)": ["BMAP_Siedlungsflaeche_klasse"],
            "OSM": ["OSM_Nutzung_original"],
            "Finale Nutzung": [
                "Final_NWGoderWG",
                "Final_Nutzung_vereinheitlicht",
                "Final_Nutzung_Quelle",
            ],
        }

        for group_name, cols in groups.items():
            cols = [c for c in cols if c in usage_cat_cols]
            if not cols:
                continue
            f.write(f"### Datenquelle: {group_name}\n\n")
            for col in cols:
                s = df[col]
                vc = s.value_counts(dropna=False)
                total = len(s) or 1
                n_non_null = int(s.notna().sum())
                share_non_null = s.notna().mean() * 100.0
                f.write(f"Spalte: {col}\n")
                f.write(f"  Nicht-leere Einträge: {n_non_null} ({share_non_null:5.1f} %)\n")
                f.write("  Verteilung (absolut / %):\n")
                for val, count in vc.items():
                    label = "<NA>" if pd.isna(val) else str(val)
                    share = 100.0 * count / total
                    f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
                f.write("\n")

    # --------------------------------------------------------------
    # Baujahrsstatistik
    # --------------------------------------------------------------
    def _write_section_year_stats(
        self,
        f,
        df: pd.DataFrame,
        year_num_cols: List[str],
        year_cat_cols: List[str],
    ) -> None:
        f.write("--- Baujahrs- und Altersstatistik ---\n\n")

        if not year_num_cols and not year_cat_cols:
            f.write("Keine der erwarteten Baujahrsspalten vorhanden.\n\n")
            return

        # Numerische Baujahre (DIVIS/OBAT/Final_Baujahr_Mitte)
        for col in year_num_cols:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            total = len(s) or 1
            n_non_null = int(s.notna().sum())
            share_non_null = s.notna().mean() * 100.0
            f.write(f"Spalte (Baujahr numerisch): {col}\n")
            f.write(f"  Nicht-leere Einträge: {n_non_null} ({share_non_null:5.1f} %)\n")
            if s.dropna().empty:
                f.write("  Keine numerisch auswertbaren Werte vorhanden.\n\n")
                continue

            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            f.write(f"  Minimum: {desc['min']:.1f}\n")
            f.write(f"  5. Perzentil: {desc['5%']:.1f}\n")
            f.write(f"  25. Perzentil: {desc['25%']:.1f}\n")
            f.write(f"  Median (50%): {desc['50%']:.1f}\n")
            f.write(f"  75. Perzentil: {desc['75%']:.1f}\n")
            f.write(f"  95. Perzentil: {desc['95%']:.1f}\n")
            f.write(f"  Maximum: {desc['max']:.1f}\n")
            f.write(f"  Mittelwert: {desc['mean']:.1f}\n")
            f.write(f"  Standardabweichung: {desc['std']:.1f}\n\n")

        # Kategoriale Baujahresklassen & Quellen
        for col in year_cat_cols:
            if col not in df.columns:
                continue
            s = df[col]
            vc = s.value_counts(dropna=False)
            total = len(s) or 1
            n_non_null = int(s.notna().sum())
            share_non_null = s.notna().mean() * 100.0
            f.write(f"Spalte (Baujahrsklasse/kategorial): {col}\n")
            f.write(f"  Nicht-leere Einträge: {n_non_null} ({share_non_null:5.1f} %)\n")
            f.write("  Verteilung (absolut / %):\n")
            for val, count in vc.items():
                label = "<NA>" if pd.isna(val) else str(val)
                share = 100.0 * count / total
                f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
            f.write("\n")

    # --------------------------------------------------------------
    # Heizsystem-Statistik
    # --------------------------------------------------------------
    def _write_section_heating_stats(
        self,
        f,
        df: pd.DataFrame,
        heating_cat_cols: List[str],
    ) -> None:
        f.write("--- Heizsysteme (Zensus-basiert, heuristisch zugewiesen) ---\n\n")
        if not heating_cat_cols:
            f.write("Keine der erwarteten Heizsystem-Spalten vorhanden.\n\n")
            return

        for col in heating_cat_cols:
            s = df[col]
            vc = s.value_counts(dropna=False)
            total = len(s) or 1
            n_non_null = int(s.notna().sum())
            share_non_null = s.notna().mean() * 100.0
            f.write(f"Spalte: {col}\n")
            f.write(f"  Nicht-leere Einträge: {n_non_null} ({share_non_null:5.1f} %)\n")
            f.write("  Verteilung (absolut / %):\n")
            for val, count in vc.items():
                label = "<NA>" if pd.isna(val) else str(val)
                share = 100.0 * count / total
                f.write(f"    - {label}: {count:7d} ({share:5.1f} %)\n")
            f.write("\n")

    # --------------------------------------------------------------
    # Plots (optional, analog zu AP1CSVStatistics)
    # --------------------------------------------------------------
    def _create_plots(
        self,
        df: pd.DataFrame,
        geom_num_cols: List[str],
        usage_cat_cols: List[str],
        year_num_cols: List[str],
        heating_cat_cols: List[str],
        out_dir: Path,
        id_cols: List[str],
    ) -> None:
        has_num = bool(geom_num_cols or year_num_cols)
        has_cat = bool(usage_cat_cols or heating_cat_cols)
        if not has_num and not has_cat:
            return

        import matplotlib.pyplot as plt

        # Numerische Spalten: Histogramm
        for col in list(geom_num_cols) + list(year_num_cols):
            if col not in df.columns or col in id_cols:
                continue
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(s, bins=30)
            plt.title(f"Histogramm {col}")
            plt.xlabel(col)
            plt.ylabel("Anzahl Gebäude")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{col}_hist.png"
            plt.savefig(out_file, dpi=150)
            plt.close()

        # Kategoriale Spalten: Balkendiagramm (Top 25)
        for col in list(usage_cat_cols) + list(heating_cat_cols):
            if col not in df.columns or col in id_cols:
                continue
            s = df[col].copy()
            if s.dropna().empty:
                continue
            vc = s.fillna("<NA>").value_counts()
            if vc.empty:
                continue
            if len(vc) > 25:
                top = vc.head(25)
                rest_sum = int(vc.iloc[25:].sum())
                top.loc["Weitere"] = rest_sum
                vc = top
            plt.figure(figsize=(10, 6))
            vc.plot(kind="bar")
            plt.title(f"Verteilung {col}")
            plt.ylabel("Anzahl Gebäude")
            plt.xticks(rotation=45, ha="right")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{col}_verteilung.png"
            plt.savefig(out_file, dpi=150)
            plt.close()


class BasicGeometryAnalysisLOD2Shapefile:
    """
    Grundlegende Geometrieanalyse für ein LoD2-Shapefile (Sachsen / EPSG:25833).

    Ziel:
    - Pro Gebäude (ID) ein Record mit Kennwerten für Energiesimulationen:
        * footprint_area_m2         – Grundfläche
        * footprint_perimeter_m     – Umfang
        * bbox_width_m / bbox_height_m / bbox_slenderness
        * compactness_index         – 4πA / P² (1 = Kreis, <1 länglich/zerklüftet)
        * equiv_diameter_m          – Durchmesser eines Kreises mit gleicher Fläche
        * height_mean_m             – gemittelte Gebäudehöhe
        * height_min_m / height_max_m
        * wall_area_est_m2          – Hüllfläche Außenwand (≈ Perimeter * Höhe)
        * roof_area_est_m2          – einfache Dachflächen-Schätzung (≈ Grundfläche)
        * volume_est_m3             – Volumenschätzung (≈ Grundfläche * Höhe)
        * n_geom_parts              – Anzahl Teilgeometrien (MultiPolygon-Komplexität)
        * n_adjacent_buildings      – Anzahl Nachbargebäude mit Berührung
        * shared_boundary_len_m     – Länge gemeinsamer Gebäudetrennflächen (2D)
        * share_wall_to_neighbours  – Anteil der Wandlänge in Kontakt zu Nachbarn
        * storeys_est               – grobe Geschosszahl (Höhe / 3 m)
        * dwelling_est              – grobe Wohnungszahl (Nutzfläche / 60 m²)
        * type_indicator            – heuristischer Gebäudetyp (SFH, MFH, Reihe, etc.)

    Ausgaben:
    - GPKG mit Geometrie + Kennwertspalten
    - CSV mit allen Attributen (ohne Geometrie)
    - TXT mit Statistikübersicht
    - PNG-Verteilungsdiagramme ausgewählter Kennwerte
    """

    _ID_CANDS = [
        "LOD_UNITID", "UNITID", "unitid", "UNIT_ID", "unit_id",
        "GEBID", "GEBIDBY", "ALKISOID",
        "gml_id", "GML_ID", "building_id", "BUILDING_ID",
        "obj_id", "OBJ_ID", "uuid", "UUID",
    ]
    _H_CANDS = [
        "HOEHEGEB", "HoeheGeb", "GebMsHoehe", "GebHoehe",
        "height", "HEIGHT", "measuredheight", "MEASUREDHEIGHT",
        "BMAP_height_m", "LOD_GebHoehe"
    ]

    def __init__(
        self,
        id_col: Optional[str] = None,
        height_col: Optional[str] = None,
        target_epsg: int = 25833,
        verbose: bool = False,
    ) -> None:
        self.id_col = id_col
        self.height_col = height_col
        self.target_epsg = target_epsg
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------
    def run(
        self,
        shp_path: Path | str | None = None,
        out_dir: Path | str | None = None,
        layer_name: str = "lod2_geom_stats",
        lod2_path: Path | str | None = None,
    ) -> Dict[str, Path]:
        """
        Liest ein LoD2-Shapefile, berechnet Gebäude-Kennwerte und gibt
        Pfade zu den wichtigsten Ausgaben zurück.

        - GPKG mit Geometrie + Kennwerten
        - CSV nur mit Attributen (ohne Geometrie)
        - TXT-Report
        - PNG-Plots
        """
        # Alias: lod2_path darf anstelle von shp_path verwendet werden
        if shp_path is None and lod2_path is not None:
            shp_path = lod2_path

        if shp_path is None:
            raise ValueError("Bitte shp_path oder lod2_path angeben.")

        shp_path = Path(shp_path)
        if out_dir is None:
            out_dir = shp_path.parent / "geom_analysis"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- robustes Einlesen mit Fallback auf Fiona ---
        try:
            gdf_raw = gpd.read_file(shp_path)
        except GEOSException as e:
            if self.verbose:
                print("[geom-analysis] Fehler beim Lesen mit Standard-Engine:", e)
                print("[geom-analysis] Versuche Fallback mit engine='fiona' ...")
            gdf_raw = gpd.read_file(shp_path, engine="fiona")

        if gdf_raw.empty:
            raise ValueError(f"LoD2-Shapefile enthält keine Objekte: {shp_path}")

        # CRS vereinheitlichen (für Flächen/Umfänge)
        try:
            if gdf_raw.crs is not None and gdf_raw.crs.to_epsg() != self.target_epsg:
                gdf_raw = gdf_raw.to_crs(self.target_epsg)
        except Exception:
            # lieber gar nicht reprojizieren, als Fehler werfen
            pass

        # ID- und Höhe-Spalten bestimmen
        id_col = self._resolve_id_col(gdf_raw)
        h_col = self._resolve_height_col(gdf_raw)

        # Gebäude aggregieren, Kennwerte berechnen, Typen zuweisen
        gdf_buildings = self._aggregate_by_building(gdf_raw, id_col, h_col)
        gdf_buildings = self._compute_basic_metrics(gdf_buildings)
        gdf_buildings = self._compute_neighbour_metrics(gdf_buildings)
        gdf_buildings = self._add_type_indicators(gdf_buildings)

        # --- Ausgaben schreiben ---
        # 1) Geopackage mit Geometrie
        gpkg_path = out_dir / f"{shp_path.stem}_geomstats.gpkg"
        gdf_buildings.to_file(gpkg_path, layer=layer_name, driver="GPKG")

        # 2) CSV NUR mit Attributen (ohne Geometrie)
        attr_df = gdf_buildings.drop(columns=[gdf_buildings.geometry.name], errors="ignore")
        attr_csv_path = out_dir / f"{shp_path.stem}_geomstats_attributes.csv"
        attr_df.to_csv(attr_csv_path, index=False)

        # 3) Summary-Report als Text
        report_path = out_dir / f"{shp_path.stem}_geomstats.txt"
        self._write_summary_report(gdf_buildings, report_path, shp_path, id_col, h_col)

        # 4) Verteilungsdiagramme
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        self._create_distribution_plots(gdf_buildings, plots_dir, prefix=shp_path.stem)

        if self.verbose:
            print(f"[geom-analysis] Ergebnisse geschrieben nach: {out_dir}")

        return {
            "gpkg": gpkg_path,
            "attributes_csv": attr_csv_path,
            "summary_txt": report_path,
            "plots_dir": plots_dir,
        }

    # ------------------------------------------------------------------
    # interne Helfer
    # ------------------------------------------------------------------
    def _resolve_id_col(self, gdf: gpd.GeoDataFrame) -> str:
        if self.id_col and self.id_col in gdf.columns:
            gdf["LOD_UNITID"] = gdf[self.id_col].astype(str)
            return "LOD_UNITID"
        for c in self._ID_CANDS:
            if c in gdf.columns:
                gdf["LOD_UNITID"] = gdf[c].astype(str)
                return "LOD_UNITID"
        # Fallback: Index
        gdf["LOD_UNITID"] = gdf.index.astype(str)
        return "LOD_UNITID"

    def _resolve_height_col(self, gdf: gpd.GeoDataFrame) -> Optional[str]:
        if self.height_col and self.height_col in gdf.columns:
            return self.height_col
        for c in self._H_CANDS:
            if c in gdf.columns:
                return c
        return None

    def _aggregate_by_building(
        self,
        gdf_raw: gpd.GeoDataFrame,
        id_col: str,
        h_col: Optional[str],
    ) -> gpd.GeoDataFrame:
        cols = [id_col]
        if h_col and h_col in gdf_raw.columns:
            cols.append(h_col)
        cols = list(dict.fromkeys(cols))

        df = gdf_raw[cols + [gdf_raw.geometry.name]].copy()

        def _agg_height(s: pd.Series) -> float | None:
            if h_col is None:
                return None
            v = pd.to_numeric(s, errors="coerce").dropna()
            return float(v.mean()) if not v.empty else None

        grouped = []
        for bid, sub in df.groupby(id_col):
            geom = sub.geometry.unary_union
            height_val = _agg_height(sub[h_col]) if h_col else None
            grouped.append(
                {
                    id_col: bid,
                    "geometry": geom,
                    "height_mean_m": height_val,
                }
            )

        gdf_b = gpd.GeoDataFrame(grouped, geometry="geometry", crs=gdf_raw.crs)
        return gdf_b

    def _compute_basic_metrics(
        self,
        gdf_b: gpd.GeoDataFrame,
        h_col_mean: str = "height_mean_m",
    ) -> gpd.GeoDataFrame:
        gdf = gdf_b.copy()

        # Grundfläche & Umfang
        gdf["footprint_area_m2"] = gdf.geometry.area
        gdf["footprint_perimeter_m"] = gdf.geometry.length

        # BBox-basiert (für "länglichen" Grundriss)
        bounds = gdf.geometry.bounds
        gdf["bbox_width_m"] = bounds["maxx"] - bounds["minx"]
        gdf["bbox_height_m"] = bounds["maxy"] - bounds["miny"]

        def _slenderness(w, h):
            if w <= 0 or h <= 0:
                return None
            return max(w, h) / min(w, h)

        gdf["bbox_slenderness"] = [
            _slenderness(w, h) for w, h in zip(gdf["bbox_width_m"], gdf["bbox_height_m"])
        ]

        # Geometrie-Komplexität: Anzahl Polygonteile
        def _num_parts(geom):
            if geom is None or geom.is_empty:
                return 0
            if geom.geom_type == "Polygon":
                return 1
            if geom.geom_type == "MultiPolygon":
                return len(geom.geoms)
            return 1

        gdf["n_geom_parts"] = gdf.geometry.apply(_num_parts)

        # Kompaktheit & äquivalenter Durchmesser
        A = gdf["footprint_area_m2"].astype(float)
        P = gdf["footprint_perimeter_m"].astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):
            gdf["compactness_index"] = (4.0 * np.pi * A) / (P ** 2)
            gdf["equiv_diameter_m"] = 2.0 * np.sqrt(A / np.pi)

        # Höhenstatistik
        if h_col_mean in gdf.columns:
            h = pd.to_numeric(gdf[h_col_mean], errors="coerce")
        else:
            h = pd.Series(index=gdf.index, dtype=float)

        gdf["height_mean_m"] = h
        gdf["height_min_m"] = h
        gdf["height_max_m"] = h

        # Geschoss-/Wohnungs-Schätzung
        # Annahmen: 3 m pro Geschoss, 60 m² pro Wohnung
        storeys = (h / 3.0).clip(lower=0.0)
        gdf["storeys_est"] = storeys
        usable_floor_area = gdf["footprint_area_m2"] * storeys
        dwellings = usable_floor_area / 60.0
        gdf["dwelling_est"] = dwellings

        # Wand- & Dachflächen (Approximationen)
        if h.notna().any():
            gdf["wall_area_est_m2"] = gdf["footprint_perimeter_m"] * h
            gdf["roof_area_est_m2"] = gdf["footprint_area_m2"]
            gdf["volume_est_m3"] = gdf["footprint_area_m2"] * h
        else:
            gdf["wall_area_est_m2"] = np.nan
            gdf["roof_area_est_m2"] = np.nan
            gdf["volume_est_m3"] = np.nan

        return gdf

    def _compute_neighbour_metrics(
        self,
        gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf["n_adjacent_buildings"] = 0
        gdf["shared_boundary_len_m"] = 0.0
        gdf["share_wall_to_neighbours"] = 0.0

        # einfache Reparatur der Geometrien
        fixed_geoms = []
        for geom in gdf.geometry.values:
            if geom is None:
                fixed_geoms.append(None)
                continue
            try:
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except GEOSException:
                geom = None
            fixed_geoms.append(geom)

        gdf[gdf.geometry.name] = fixed_geoms

        # Spatial Index
        try:
            sindex = gdf.sindex
        except Exception:
            return gdf

        geoms = list(gdf.geometry.values)
        perims = gdf["footprint_perimeter_m"].values

        for idx, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue

            try:
                if not geom.is_valid:
                    try:
                        geom = geom.buffer(0)
                    except GEOSException:
                        continue
                    geoms[idx] = geom
                    gdf.at[gdf.index[idx], gdf.geometry.name] = geom
            except GEOSException:
                continue

            try:
                possible_matches_idx = list(sindex.query(geom, predicate="intersects"))
            except GEOSException:
                continue

            possible_matches_idx = [j for j in possible_matches_idx if j != idx]
            if not possible_matches_idx:
                continue

            neighbours = set()
            shared_len = 0.0

            for j in possible_matches_idx:
                other = geoms[j]
                if other is None or other.is_empty:
                    continue

                try:
                    if not other.is_valid:
                        try:
                            other = other.buffer(0)
                        except GEOSException:
                            continue
                        geoms[j] = other
                        gdf.at[gdf.index[j], gdf.geometry.name] = other
                except GEOSException:
                    continue

                try:
                    if not geom.touches(other) and not geom.intersects(other):
                        continue
                    inter = geom.boundary.intersection(other.boundary)
                except GEOSException:
                    continue

                if not inter.is_empty:
                    neighbours.add(j)
                    try:
                        shared_len += inter.length
                    except GEOSException:
                        pass

            gdf.at[gdf.index[idx], "n_adjacent_buildings"] = len(neighbours)
            gdf.at[gdf.index[idx], "shared_boundary_len_m"] = shared_len

            perim = perims[idx] if idx < len(perims) else None
            if perim and perim > 0:
                gdf.at[gdf.index[idx], "share_wall_to_neighbours"] = shared_len / perim

        return gdf

    def _add_type_indicators(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Heuristische Gebäudetyp-Zuordnung.

        Nutzt:
        - footprint_area_m2
        - height_mean_m / storeys_est
        - share_wall_to_neighbours
        - bbox_slenderness
        - dwelling_est
        """

        gdf = gdf.copy()

        area = gdf["footprint_area_m2"]
        h = gdf["height_mean_m"]
        storeys = gdf["storeys_est"].fillna(0)
        share_wall = gdf["share_wall_to_neighbours"].fillna(0)
        slender = gdf["bbox_slenderness"].fillna(1)
        dwellings = gdf["dwelling_est"].fillna(0)

        type_indicator = []

        for a, hh, st, sw, sl, dw in zip(area, h, storeys, share_wall, slender, dwellings):
            t = "UNKNOWN"

            # 1) Sehr kleine Gebäude (Gartenlauben, Schuppen, Garagen …)
            if (a < 30) and (hh < 3.5) and (sw < 0.2):
                t = "SMALL_OUTBUILDING"

            # 4) Reihenhäuser – länglicher Grundriss, max. 3 Geschosse,
            #    mind. 50 % Wandfläche an Nachbarn
            elif (sl >= 2.5) and (st <= 3.5) and (sw >= 0.5):
                t = "ROW_HOUSE"

            # 3) Doppelhaushälfte – kompakt, ca. 1/4 der Wandfläche an Nachbarn
            elif (a >= 60) and (a <= 200) and (hh >= 5) and (hh <= 11) and (sw >= 0.15) and (sw <= 0.40) and (sl <= 2.0):
                t = "SEMI_DETACHED"

            # 2) Freistehendes Einfamilienhaus – kleine Grundfläche, niedrige Höhe,
            #    kaum Kontakt zu Nachbarn
            elif (a >= 60) and (a <= 160) and (hh >= 5) and (hh <= 11) and (sw < 0.1) and (st <= 2.5):
                t = "SFH_DETACHED"

            # 5–7) Mehrfamilienhäuser anhand geschätzter Wohnungszahl
            else:
                if dw > 1.5:
                    if dw <= 6:
                        t = "MFH_SMALL_UP_TO_6"
                    elif dw <= 20:
                        t = "MFH_MEDIUM_UP_TO_20"
                    else:
                        t = "MFH_LARGE_GT_20"

            type_indicator.append(t)

        gdf["type_indicator"] = type_indicator
        return gdf

    def _write_summary_report(
        self,
        gdf: gpd.GeoDataFrame,
        report_path: Path,
        shp_path: Path,
        id_col: str,
        h_col: Optional[str],
    ) -> None:
        with report_path.open("w", encoding="utf-8") as f:
            f.write("BasicGeometryAnalysis – LoD2-Shapefile\n")
            f.write("=" * 72 + "\n\n")
            f.write(f"Eingabedatei : {shp_path}\n")
            f.write(f"Anzahl Gebäude: {len(gdf)}\n")
            f.write(f"ID-Spalte    : {id_col}\n")
            f.write(f"Höhen-Spalte : {h_col or 'keine (nur 2D-Kennwerte)'}\n\n")

            def _write_desc(title: str, series: pd.Series):
                s = pd.to_numeric(series, errors="coerce").dropna()
                if s.empty:
                    f.write(f"{title}: keine Werte vorhanden.\n\n")
                    return
                desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
                f.write(f"{title}:\n")
                f.write(f"  N         : {int(desc['count'])}\n")
                f.write(f"  Min       : {desc['min']:.3f}\n")
                f.write(f"  5%        : {desc['5%']:.3f}\n")
                f.write(f"  25%       : {desc['25%']:.3f}\n")
                f.write(f"  Median    : {desc['50%']:.3f}\n")
                f.write(f"  75%       : {desc['75%']:.3f}\n")
                f.write(f"  95%       : {desc['95%']:.3f}\n")
                f.write(f"  Max       : {desc['max']:.3f}\n")
                f.write(f"  Mittelwert: {desc['mean']:.3f}\n\n")

            _write_desc("Grundfläche [m²]", gdf["footprint_area_m2"])
            _write_desc("Umfang [m]", gdf["footprint_perimeter_m"])
            if "height_mean_m" in gdf.columns:
                _write_desc("Gebäudehöhe (Mittel) [m]", gdf["height_mean_m"])
            if "wall_area_est_m2" in gdf.columns:
                _write_desc("geschätzte Außenwandfläche [m²]", gdf["wall_area_est_m2"])
            if "roof_area_est_m2" in gdf.columns:
                _write_desc("geschätzte Dachfläche [m²]", gdf["roof_area_est_m2"])
            if "volume_est_m3" in gdf.columns:
                _write_desc("geschätztes Gebäudevolumen [m³]", gdf["volume_est_m3"])
            if "share_wall_to_neighbours" in gdf.columns:
                _write_desc(
                    "Anteil Wandlänge im Kontakt zu Nachbarn [–]",
                    gdf["share_wall_to_neighbours"],
                )
            if "dwelling_est" in gdf.columns:
                _write_desc("geschätzte Wohnungszahl [–]", gdf["dwelling_est"])

            if "type_indicator" in gdf.columns:
                vc = gdf["type_indicator"].value_counts(dropna=False)
                f.write("\nVerteilung Gebäudetyp-Indikator:\n")
                for k, v in vc.items():
                    f.write(f"  {k:25s}: {v:6d}\n")

    def _create_distribution_plots(
        self,
        gdf: gpd.GeoDataFrame,
        out_dir: Path,
        prefix: str = "",
    ) -> None:
        """
        Erzeugt einfache Verteilungsdiagramme (Histogramme / Balkendiagramme)
        für einige zentrale Kennwerte.
        """
        out_dir = Path(out_dir)

        import matplotlib.pyplot as plt  # lokal importieren

        num_cols = [
            "footprint_area_m2",
            "height_mean_m",
            "storeys_est",
            "dwelling_est",
            "share_wall_to_neighbours",
            "n_adjacent_buildings",
            "compactness_index",
            "bbox_slenderness",
        ]
        num_cols = [c for c in num_cols if c in gdf.columns]

        cat_cols = []
        if "type_indicator" in gdf.columns:
            cat_cols.append("type_indicator")

        # numerische Histogramme
        for col in num_cols:
            s = pd.to_numeric(gdf[col], errors="coerce").dropna()
            if s.empty:
                continue
            plt.figure(figsize=(8, 5))
            s.hist(bins=30)
            plt.title(f"Verteilung {col}")
            plt.xlabel(col)
            plt.ylabel("Anzahl Gebäude")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{prefix}_{col}_hist.png"
            plt.savefig(out_file, dpi=150)
            plt.close()

        # kategoriale Verteilungen (Balkendiagramm)
        for col in cat_cols:
            s = gdf[col].fillna("NA").astype(str)
            vc = s.value_counts()
            plt.figure(figsize=(10, 6))
            vc.plot(kind="bar")
            plt.title(f"Verteilung {col}")
            plt.ylabel("Anzahl Gebäude")
            plt.xticks(rotation=45, ha="right")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Tight layout not applied.*")
                plt.tight_layout()
            out_file = out_dir / f"{prefix}_{col}_verteilung.png"
            plt.savefig(out_file, dpi=150)
            plt.close()



# ------------------------------------------------------------
# UnifiedDatasetWriter (Export)
# ------------------------------------------------------------

class UnifiedDatasetWriter:
    """
    Datenklasse f?r unified dataset writer.
    """
    def to_geopackage(self, gdf: gpd.GeoDataFrame, path: Path, layer: str = "buildings") -> Path:
        """
        F?hrt to_geopackage aus.
        
        Args:
            gdf: Beschreibung.
            path: Beschreibung.
            layer: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, layer=layer, driver="GPKG")
        return path

    def to_parquet(self, gdf: gpd.GeoDataFrame, dir: Path, filename: str = "buildings.parquet") -> Path:
        """
        F?hrt to_parquet aus.
        
        Args:
            gdf: Beschreibung.
            dir: Beschreibung.
            filename: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        out = dir / filename
        gdf.to_parquet(out)
        return out

    def to_geojson(self, gdf: gpd.GeoDataFrame, dir: Path, filename: str = "buildings.geojson") -> Path:
        """
        F?hrt to_geojson aus.
        
        Args:
            gdf: Beschreibung.
            dir: Beschreibung.
            filename: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        out = dir / filename
        gdf.to_file(out, driver="GeoJSON")
        return out
