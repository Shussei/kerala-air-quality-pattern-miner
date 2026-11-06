# DATA_SOURCES.md

Kerala Air Quality Pattern Miner — data-source inventory.

**Retrieval date:** 2026-10-30 (all checks performed this date)

## How to read this document

Each source lists verified facts (confirmed by live request or direct
observation) and unresolved questions. Nothing here is assumed; automated
ingestion is optional and credential/network guarded. CSV upload always works
offline.

---

## 1. OpenAQ Platform — air quality

| Field | Value |
|-------|-------|
| Name | OpenAQ Platform (air-quality measurements) |
| URL | https://api.openaq.org/v3/locations ; https://docs.openaq.org/ |
| Retrieval date | 2026-10-30 |
| Access method | REST API v3, `GET /v3/locations`, `GET /v3/sensors/{id}/measurements/daily` |
| Schema | location id, name, city, coordinates (lat/lon WGS84), sensors/parameters, datetime, value, unit, license |
| Units | µg/m³ for PM2.5/PM10/etc. (varies by parameter) |
| Location coverage | **NOT available for Kerala.** Live query with a valid API key returned 4,000+ India locations across 4 pagination pages (offsets 0–4000); zero locations fell within the Kerala bounding box (lat 8.0–12.5°N, lon 74.0–77.5°E). |
| Temporal coverage | Varies by sensor; hourly and daily aggregations available |
| License / terms | OpenAQ terms of service; attribution required |
| API requirements | **API key required** (`X-API-Key` header). Key confirmed working (authentication successful). Read from `OPENAQ_API_KEY` env var. Never hardcoded, logged, or committed. |
| Limitations | OpenAQ does not appear to operate stations in Kerala. Automated OpenAQ ingestion is not useful for this project without Kerala coverage. |

### Verification performed

`GET https://api.openaq.org/v3/locations?country=IN&limit=1000` with a valid API key
returned HTTP 200 with 1,000 results per page. Pagination through offsets 0, 1000,
2000, 3000 (4,000+ locations) found **zero** stations with coordinates in Kerala.

### Conclusion

OpenAQ API access works, but Kerala coverage is absent. Do not rely on OpenAQ for
Kerala air-quality data. Use the CPCB CSV files (see Source 4) instead.

---

## 2. NASA POWER — meteorology / weather

| Field | Value |
|-------|-------|
| Name | NASA POWER (Solar & Meteorological data from MERRA-2 model) |
| URL | https://power.larc.nasa.gov/docs/services/api/temporal/daily/ |
| Retrieval date | 2026-10-30 |
| Access method | REST API `temporal/daily/`, point-based (latitude/longitude), daily temporal resolution |
| Schema | YEAR, MO, DY, T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS10M |
| Units | T2M/T2M_MAX/T2M_MIN: °C; RH2M: %; PRECTOTCORR: mm/day; WS10M: m/s |
| Location coverage | Global grid. Downloaded point: **latitude 9.9312, longitude 76.2673** (Kerala, near Kochi/Cochin) |
| Temporal coverage | **2019-01-01 through 2025-12-31** (daily, in Local Solar Time) |
| License / terms | NASA POWER terms of use (public, free) |
| API requirements | None required for public access |
| Limitations | CSV file has a 13-line text header block before the column header; must skip 14 rows on load. Date is split across YEAR/MO/DY columns (must combine into a single datetime). Elevation 47.32 m. Missing data coded as -999. |

### Verification performed

Downloaded file `POWER_Point_Daily_20190101_20251231_009d93N_076d27E_LST.csv`
contains 7,833 data rows (2019–2025) at Kerala coordinates 9.9312°N, 76.2673°E.
Parameters: T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS10M.

---

## 3. Kerala State Pollution Control Board (KSPCB)

| Field | Value |
|-------|-------|
| Name | KSPCB Continuous Ambient Air Quality Monitoring |
| URL | https://kspcb.kerala.gov.in/samp ; https://www.kspcbipam.kerala.gov.in/Dashboard/airQuality |
| Retrieval date | 2026-10-30 |
| Access method | Web dashboard; dated AQI PDF widget observed. No confirmed public bulk API. |
| Schema | Station, SO2, CO, PM2.5, O3, PM10, NH3, NOx, NO, NO2, AQI, Remarks |
| Units | µg/m³ for gases/PM; CO in mg/m³ (per KSPCB PDF) |
| Location coverage | Kerala stations observed in one dated PDF widget. Full station list unverified. |
| Temporal coverage | Single-date snapshot observed (PDF dated 28-08-2026). Historical bulk coverage unverified. |
| License / terms | KSPCB website terms |
| API requirements | None confirmed; no public bulk API identified |
| Limitations | Dashboard not confirmed to offer downloadable bulk dataset or public API. Use CSV upload (Source 4) for historical analysis. |

---

## 4. CPCB / KSPCB daily air-quality CSV files (downloaded, REAL data)

| Field | Value |
|-------|-------|
| Name | CPCB daily ambient air quality — Kerala |
| Files | `data/raw/cpcb_dly_aq_kerala-2008.csv`, `-2009.csv`, `-2010.csv`, `-2011.csv`, `-2015.csv` |
| Retrieval date | 2026-10-30 |
| Access method | User-downloaded CSV files (local upload) |
| Schema (2008/2009) | Sampling Date, State, City/Town/Village/Area, Location of Monitoring Station, Type of Location, SO2, NO2, RSPM/PM10, SPM |
| Schema (2010/2011) | Stn Code, Sampling Date, State, City/Town/Village/Area, Location of Monitoring Station, Agency, Type of Location, SO2, NO2, RSPM/PM10, SPM |
| Schema (2015) | Stn Code, Sampling Date, State, City/Town/Village/Area, Location of Monitoring Station, Agency, Type of Location, SO2, NO2, RSPM/PM10, PM 2.5 |
| Units | µg/m³ (SO2, NO2, RSPM/PM10, SPM, PM 2.5) |
| Location coverage | Kerala stations: Eloor (Kochi, industrial), SMV School & HiTech Chackai & Sasthamangalam (Thiruvananthapuram), District Office Alappuzha, etc. |
| Temporal coverage | 2008, 2009, 2010, 2011, 2015 (intermittent sampling dates, not continuous daily) |
| License / terms | CPCB / KSPCB public data |
| Limitations | Date format varies (M/D/YYYY and DD-MM-YY). Missing values appear as NaN. No latitude/longitude in files (station names must be matched by name or coordinates added manually). Different years have different columns. |

### Verified stations in the downloaded files

- **Eloor, Kochi** (Industrial Area) — 2008, 2009, 2015
- **SMV School, Thiruvananthapuram** — 2010
- **HiTech Chackai, Thiruvananthapuram** (Industrial Area) — 2010
- **Sasthamangalam, Thiruvananthapuram** — 2010
- **District Office, Alissery Road, Alappuzha** — 2010

---

## 5. User-provided CSV (reliable fallback)

Same as before — upload via the Streamlit UI.

---

## Recommended integration using downloaded files

| Dataset | File | Role | Join key |
|---------|------|------|----------|
| Air quality | `data/raw/cpcb_dly_aq_kerala-2010.csv` (or 2008/2009/2011/2015) | Air-quality observations | `Sampling Date` + `Location of Monitoring Station` |
| Weather | `data/raw/POWER_Point_Daily_20190101_20251231_009d93N_076d27E_LST.csv` | Weather observations | Combined `YEAR+MO+DY` + location name (e.g. "Kochi") |

**Important:** The CPCB data (2008–2015) and NASA POWER data (2019–2025) **do not overlap in time**. Integration will produce zero matched rows unless you use data from overlapping years. For a working demonstration, either:
1. Use only one dataset for EDA/clustering/Apriori, or
2. Find/download weather data covering 2008–2015 to match the CPCB air-quality years.

Column mapping for CPCB (Data Import page):
- datetime → `Sampling Date`
- location → `Location of Monitoring Station`
- (no lat/lon in file — use location name matching)

Column mapping for NASA POWER:
- Skip leading rows → `14`
- Date columns → Year: `YEAR`, Month: `MO`, Day: `DY` (combined into `datetime`)
- location → add a constant location name, or map to an existing column
- Numeric weather variables: `T2M`, `RH2M`, `PRECTOTCORR`, `WS10M`
