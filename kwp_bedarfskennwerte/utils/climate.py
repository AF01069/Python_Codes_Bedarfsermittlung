"""
kwp_bedarfskennwerte.utils.climate

Klimakorrektur-Hilfsfunktionen.

Im Projekt werden Bedarfskennwerte (IWU) verwendet, die sich auf ein
Referenzklima beziehen (TRY-Referenzstation Potsdam).

Der Deutsche Wetterdienst (DWD) stellt monatliche Klimafaktoren (KF) je
Zustell-Postleitzahl bereit. Laut DWD-Datensatzbeschreibung werden die
Klimafaktoren für gleitende 12-Monats-Zeiträume berechnet als:

    KF = G(TRY, Potsdam) / G(Standort)

Damit gilt für die Übertragung eines referenzklimatischen Heizbedarfs
(Q_ref, z. B. IWU) auf den Standort:

    Q_standort = Q_ref * (G(Standort) / G(TRY,Potsdam)) = Q_ref / KF

Dieses Modul lädt (bei Bedarf) den jeweils aktuellsten
DWD-Klimafaktor-Datensatz (CSV) aus dem OpenData-Verzeichnis des DWD und
stellt Hilfsfunktionen zur Ermittlung des standortbezogenen Faktors bereit.

Robustheit:
- Das DWD-Verzeichnis enthält sowohl ".csv" als auch "_k.csv" Varianten.
  Dieses Modul bevorzugt ".csv" und kann Dezimal-Kommas automatisch parsen.
- Der CSV-Delimiter wird per Sniffer ermittelt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import io
import re
from typing import Dict, Optional, Tuple

import requests


DWD_KF_BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/derived_germany/techn/monthly/"
    "climate_correction_factor/recent/"
)


@dataclass(frozen=True)
class DwdKfDataset:
    """Repräsentiert eine konkrete DWD-Klimafaktor-Datei aus dem recent-Verzeichnis."""
    filename: str
    url: str

    @property
    def period(self) -> str:
        # KF_YYYYMMDD_YYYYMMDD.csv
        """
        F?hrt period aus.
        
        Args:
        
        Returns:
            Beschreibung.
        """
        m = re.match(r"^KF_(\d{8})_(\d{8})\.csv$", self.filename)
        if not m:
            return self.filename
        return f"{m.group(1)}-{m.group(2)}"


def normalize_plz(value) -> Optional[str]:
    """Normalisiert PLZ auf 5-stellig (nur Ziffern)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) == 5:
        return digits
    # Manche Daten enthalten 4-stellige (führende 0 fehlt)
    if len(digits) == 4:
        return "0" + digits
    return None


def _fetch_index_html(timeout_s: int = 30) -> str:
    r = requests.get(DWD_KF_BASE_URL, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def _pick_latest_csv_from_index(index_html: str) -> DwdKfDataset:
    """
    Wählt aus der Directory-Listing-HTML die 'neueste' KF_*.csv (ohne _k.csv).
    Entscheidung: max nach Enddatum im Dateinamen.
    """
    filenames = re.findall(r'href="(KF_\d{8}_\d{8}\.csv)"', index_html)
    if not filenames:
        raise RuntimeError("Keine KF_*.csv Dateien im DWD-Index gefunden.")

    def end_date(fn: str) -> int:
        """
        F?hrt end_date aus.
        
        Args:
            fn: Beschreibung.
        
        Returns:
            Beschreibung.
        """
        m = re.match(r"^KF_(\d{8})_(\d{8})\.csv$", fn)
        return int(m.group(2)) if m else 0

    latest = max(filenames, key=end_date)
    return DwdKfDataset(filename=latest, url=DWD_KF_BASE_URL + latest)


def ensure_dwd_kf_dataset(cache_dir: Path, timeout_s: int = 30) -> Tuple[DwdKfDataset, Path]:
    """
    Stellt sicher, dass der aktuellste DWD-Klimafaktor-Datensatz lokal vorliegt.
    Lädt die Datei in cache_dir/dwd_kf/ herunter (wenn nicht vorhanden).

    Returns: (dataset, local_path)
    """
    cache_dir = Path(cache_dir)
    target_dir = cache_dir / "dwd_kf"
    target_dir.mkdir(parents=True, exist_ok=True)

    index_html = _fetch_index_html(timeout_s=timeout_s)
    ds = _pick_latest_csv_from_index(index_html)
    local_path = target_dir / ds.filename

    if not local_path.exists():
        r = requests.get(ds.url, timeout=timeout_s)
        r.raise_for_status()
        local_path.write_bytes(r.content)

    return ds, local_path


def load_kf_mapping(csv_path: Path) -> Dict[str, float]:
    """
    Lädt eine KF_*.csv in ein Mapping {plz: kf_float}.

    Die Datei enthält typischerweise >8.000 PLZ-Werte. Delimiter wird erkannt.
    Unterstützt Dezimal-Kommas.
    """
    csv_path = Path(csv_path)
    raw = csv_path.read_bytes()

    # robust: versuche utf-8, sonst latin1
    for enc in ("utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    # Sniffer
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
    except Exception:
        # fallback
        dialect = csv.excel
        dialect.delimiter = ";"

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames is None:
        raise RuntimeError(f"Keine Header in Klimafaktor-CSV: {csv_path}")

    # plausible Spalten finden
    fields_lower = {f.lower(): f for f in reader.fieldnames}
    plz_col = None
    for cand in ("plz", "postleitzahl", "zip", "zipcode"):
        if cand in fields_lower:
            plz_col = fields_lower[cand]
            break
    if plz_col is None:
        # häufig: erste Spalte ist PLZ
        plz_col = reader.fieldnames[0]

    # KF-Spalte: oft "KF" oder ähnlich
    kf_col = None
    for cand in ("kf", "klimafaktor", "climate_correction_factor"):
        if cand in fields_lower:
            kf_col = fields_lower[cand]
            break
    if kf_col is None:
        # häufig: zweite Spalte ist KF
        if len(reader.fieldnames) < 2:
            raise RuntimeError(f"Unerwartetes Spaltenlayout in {csv_path}: {reader.fieldnames}")
        kf_col = reader.fieldnames[1]

    mapping: Dict[str, float] = {}
    for row in reader:
        plz = normalize_plz(row.get(plz_col))
        if not plz:
            continue
        v = row.get(kf_col)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        # Dezimal-Komma -> Punkt
        s = s.replace(",", ".")
        try:
            mapping[plz] = float(s)
        except ValueError:
            continue

    if not mapping:
        raise RuntimeError(f"Keine PLZ->KF Werte aus {csv_path} geladen (Spalten: {reader.fieldnames}).")
    return mapping
