"""
Runtime-Konfiguration und Kontextobjekte f?r die Pipeline.

H?lt u. a. Pfade, BBOX, CRS/EPSG sowie Kontext f?r tempor?re Artefakte.
"""
# kwp_bedarfskennwerte/config/runtime.py
from __future__ import annotations
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .schemas import AppSettings

# ---------------------------------------------------------------------------
# Ziel: Settings aus (1) Datei (kwp.toml/yaml/json), (2) ENV, (3) CLI-Defaults
# zusammenführen. Validierung erst *nach* dem Merge (via AppSettings).

# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """
    Datenklasse f?r pipeline context.
    """
    settings: AppSettings
    def resource(self, *parts: str) -> Path:
        """
        F?hrt resource aus.
        
        Args:
            *parts: Weitere Positionsargumente.
        
        Returns:
            Beschreibung.
        """
        p = Path(self.settings.data.out_dir, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

def _read_file(p: Path) -> Dict[str, Any]:
    """kleine Loader-Hilfe für json/toml/yaml (falls libs vorhanden)."""
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix.lower() in {".toml", ".tml"}:
        try:
            import tomllib  # py311+
        except Exception:
            import tomli as tomllib  # type: ignore
        return tomllib.loads(p.read_text(encoding="utf-8"))
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # pip install pyyaml
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raise ValueError(f"Unbekanntes Konfigformat: {p.suffix}")

def _load_file_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Lädt Datei-Konfiguration – expliziter Pfad > ENV `KWP_CONFIG` > Standards."""
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    env_cfg = os.environ.get("KWP_CONFIG")
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates += [Path("kwp.json"), Path("kwp.toml"), Path("kwp.yaml"), Path("kwp.yml")]
    for p in candidates:
        if p.exists():
            return _read_file(p)
    return {}

def _load_env_config(prefix: str = "KWP_") -> Dict[str, Any]:
    """ENV → verschachtelt per '__'. Einfache int/float-Casts bleiben erhalten."""
    conf: Dict[str, Any] = {}
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path = k[len(prefix):].lower().split("__")
        cursor = conf
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        # simple Casts
        if v.isdigit():
            val: Any = int(v)
        else:
            try:
                val = float(v)
            except ValueError:
                val = v
        cursor[path[-1]] = val
    return conf

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def build_context(cli_defaults: Dict[str, Any], config_path: Optional[Path] = None) -> PipelineContext:
    """Erstellt Kontext via Datei < ENV < CLI."""
    file_conf = _load_file_config(config_path)
    env_conf = _load_env_config()
    merged = _deep_merge(file_conf, env_conf)
    merged = _deep_merge(merged, cli_defaults)
    settings = AppSettings(**merged)
    return PipelineContext(settings=settings)
