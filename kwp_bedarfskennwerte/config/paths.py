"""
Zentrale Pfadkonfiguration f?r Daten, Cache, Outputs und Arbeitsverzeichnisse.

Alle Pfade sind relativ zum Projektroot, um portable Projekt-Setups
(z. B. Notebook, CLI, CI) zu erm?glichen.
"""
# kwp_bedarfskennwerte/config/paths.py
from __future__ import annotations

from pathlib import Path

# Ordnernamen unterhalb von ./data
DATA_DIRNAME = "data"
DIR_ADRESSEN = "Adressen"
DIR_BAUJAHRE_OBAT = "Baujahre_OBAT"
DIR_GEOMETRIE_LOD2 = "Geometrie_LOD2"
DIR_BEDARFSKENNWERTE_IWU = "Bedarfskennwerte_IWU"
DIR_ZENSUS2022 = "Zensus2022"


def rel_data_dir() -> Path:
    """Relatives data-Verzeichnis (gegenueber Working Directory)."""
    return Path(DATA_DIRNAME)


def rel_data_subdir(name: str) -> Path:
    """Relatives Unterverzeichnis von data/."""
    return rel_data_dir() / name


def rel_adressen_dir() -> Path:
    """
    F?hrt rel_adressen_dir aus.
    
    Returns:
        Beschreibung.
    """
    return rel_data_subdir(DIR_ADRESSEN)


def rel_baujahre_obat_dir() -> Path:
    """
    F?hrt rel_baujahre_obat_dir aus.
    
    Returns:
        Beschreibung.
    """
    return rel_data_subdir(DIR_BAUJAHRE_OBAT)


def rel_geometrie_lod2_dir() -> Path:
    """
    F?hrt rel_geometrie_lod2_dir aus.
    
    Returns:
        Beschreibung.
    """
    return rel_data_subdir(DIR_GEOMETRIE_LOD2)


def rel_bedarfskennwerte_iwu_dir() -> Path:
    """
    F?hrt rel_bedarfskennwerte_iwu_dir aus.
    
    Returns:
        Beschreibung.
    """
    return rel_data_subdir(DIR_BEDARFSKENNWERTE_IWU)


def rel_zensus2022_dir() -> Path:
    """
    F?hrt rel_zensus2022_dir aus.
    
    Returns:
        Beschreibung.
    """
    return rel_data_subdir(DIR_ZENSUS2022)
