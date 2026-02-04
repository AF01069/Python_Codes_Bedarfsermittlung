# KWP-Bedarfskennwerte – SAENA Bedarfsermittlung

Pipeline zur Ableitung von Wärme- und Kältebedarfen auf Gebäudeebene für die
Kommunale Wärmeplanung in Sachsen. Die Methodik orientiert sich am
Studienbericht in `docs/Endbericht_SAENAStudie_BKF.docx`.

## Überblick

Das Projekt stellt eine reproduzierbare Pipeline bereit:

- **AP1**: Geometrische/strukturelle Anreicherung (LoD2, OSM, Basemap)
- **AP1-Enrich**: Zensus 2022, GHS-OBAT, DIVIS, Adressen
- **AP2**: Gebäudetypisierung + Wärme-/TWW-/Kältebedarf
- **AP2-Climate**: Klimakorrektur (DWD Klimafaktoren)
- **Forecast 2045**: Szenariofortschreibung

## Voraussetzungen

- Windows (getestet)
- Python in `venv/` (lokale Umgebung)
- Eingangsdatensätze in `data/` (siehe Abschnitt Datenstruktur)

## Datenstruktur (erwartet)

```
data/
  Adressen/
    hk_sn_adressen_20250918.txt
  Geometrie_LOD2/
    lod2_33496_5676_2_sn.shp
  Baujahre_OBAT/
    GHS_OBAT_GPKG_DEU_E2020_R2024A_V1_0.gpkg
  Bedarfskennwerte_IWU/
    IWU_Bedarfskennwerte_combined_flat.csv
  Zensus2022/
    zensus_100m.gpkg  (wird bei Bedarf erzeugt)
```

## Quickstart (Beispiel)

Ein kompletter Lauf über `main.py` startet alle Schritte nacheinander:

```powershell
.\venv\Scripts\python.exe main.py
```

Einzelläufe (CLI):

```powershell
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap1 --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap1-enrich --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap2 --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli ap2-climate --help
.\venv\Scripts\python.exe -m kwp_bedarfskennwerte.cli forecast-2045 --help
```

## Outputs

Outputs werden standardmäßig unter `out/` geschrieben:

- `out/ap1/` – AP1 Ergebnisse, QA, Vergleichsstatistiken
- `out/ap2/` – Typisierung, Wärme-/Kältebedarfe, Forecast-2045

## Lizenz

Noch festzulegen. Bitte `LICENSE` ergänzen.

## Datennutzung

Siehe `docs/DATASETS.md` für Datenquellen, Verwendungszwecke und Lizenzen.

