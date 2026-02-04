"""
Schema-Definitionen und Spaltenmodelle.

Dient der Vereinheitlichung von Attributnamen in AP1/AP2 und f?r
Ausgabeformate (GPKG/CSV).
"""
# kwp_bedarfskennwerte/config/schemas.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Hinweis: Für robuste Validierung Pydantic verwenden.
try:  # pydantic v2 bevorzugt; v1-Fallback
    from pydantic import BaseModel, Field, AnyUrl
except Exception:  # pragma: no cover
    # Minimaler Fallback ohne echte Validierung – bitte pydantic installieren!
    class BaseModel:  # type: ignore
        """
        Datenklasse f?r base model.
        """
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
    def Field(default=None, **kwargs):
        """
        F?hrt Field aus.
        
        Args:
            default: Beschreibung.
            **kwargs: Weitere Schl?sselwortargumente.
        
        Returns:
            Beschreibung.
        """
        return default
    AnyUrl = str  # type: ignore

BBox = Tuple[float, float, float, float]

class OverpassSettings(BaseModel):
    """Konfiguration für Overpass/OSM-Zugriffe."""
    url: Optional[AnyUrl] = Field(None, description="Primäre Overpass-API URL")
    mirrors: List[AnyUrl] = Field(default_factory=list, description="Fallback-Overpass-Mirrors")
    timeout_s: int = 120
    tile_size_m: int = 600
    max_retries: int = 3
    sleep_between_s: float = 1.0

class BasemapSettings(BaseModel):
    """Optionale Basemap/XYZ/MVT-Einstellungen."""
    mvt_url_template: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class DataPaths(BaseModel):
    """Zentrale Pfade für Artefakte, Cache und Eingänge."""
    lod2_path: Optional[Path] = None
    cache_dir: Path
    out_dir: Path
    work_dir: Path

class RegionSpec(BaseModel):
    """Eingrenzung des Arbeitsgebiets."""
    bbox_25833: Optional[BBox] = Field(None, description="BBOX in EPSG:25833")
    region_file: Optional[Path] = None
    region_layer: Optional[str] = None

class AppSettings(BaseModel):
    """Hauptkonfiguration, die im gesamten Projekt weitergereicht wird."""
    region: RegionSpec
    data: DataPaths
    overpass: OverpassSettings
    basemap: BasemapSettings = BasemapSettings()

    target_epsg: int = 25833
    verbose: bool = False

    # Beispiel: zusätzliche globale Parameter
    sample_area_fraction: float = 0.1
