# Punjab Remote Sensing Risk Atlas Dashboard

A ready-to-run Streamlit dashboard for the **Punjab Remote Sensing Risk Atlas**.

## Public dashboard title

**Punjab Remote Sensing Risk Atlas**

## What it shows

The dashboard displays satellite-based public layers:

1. Environmental Stress Hotspots
2. Drought-Prone Areas
3. Water Storage Stress
4. Crop and Vegetation Stress
5. Heat Stress Hotspots
6. Rainfall Deficit Zones
7. Soil Moisture Stress
8. District Risk Scorecard

The backend analysis period is **2018–2024**, but public layer names are kept simple.

## Folder structure

```text
Punjab_RS_Risk_Atlas_Dashboard_Final/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── rasters/
│   ├── tables/
│   └── vectors/
│
└── assets/
```

## Place these exported GEE files

Put raster files in:

```text
data/rasters/
```

Required raster files:

```text
Punjab_Environmental_Stress_Hotspots.tif
Punjab_Drought_Prone_Areas.tif
Punjab_Water_Storage_Stress.tif
Punjab_Crop_Vegetation_Stress.tif
Punjab_Heat_Stress_Hotspots.tif
Punjab_Rainfall_Deficit_Zones.tif
Punjab_Soil_Moisture_Stress.tif
```

Put the scorecard CSV in:

```text
data/tables/
```

Required table:

```text
Punjab_District_Risk_Scorecard.csv
```

Put district boundary GeoJSON in:

```text
data/vectors/
```

Required vector:

```text
Punjab_Districts.geojson
```

## Run locally

Install requirements:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Upload this folder to GitHub.
2. Go to Streamlit Cloud.
3. Select the GitHub repository.
4. Set main file as:

```text
app.py
```

5. Deploy.

## Notes

Water Storage Stress is based on GRACE/GRACE-FO total water storage anomaly. It is not exact groundwater depth and should be used for broad planning and risk communication.
