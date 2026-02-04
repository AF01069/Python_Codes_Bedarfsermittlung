# Setup

## Voraussetzungen
- Windows
- Python-Umgebung in `venv/`

## Datenstruktur
Siehe `README.md` und `docs/DATASETS.md`.

## Lauf
```powershell
.\venv\Scripts\python.exe main.py
```

## Einzelschritte
```powershell
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap1 --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap1-enrich --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap2 --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap2-climate --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli forecast-2045 --help
```

