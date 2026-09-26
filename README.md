# ConvLSTM and Temporal Vision Transformer for decade-ahead forecasting of categorical land cover maps

PyTorch and Python code that trains a convolutional LSTM (ConvLSTM) and a Temporal Vision
Transformer (ViT) to forecast annual categorical land cover maps ten years ahead, validates
them by a temporal hindcast against persistence, CA–Markov and Random Forest baselines,
and reproduces all statistics, tables and figures of the accompanying article
(citation to be added on publication).

## Method in brief

- **Input:** 31 annual land cover maps (250 m, 1990–2020) grouped into seven classes and
  one-hot encoded; each sample stacks the maps of five consecutive years (t−4 … t).
- **Target:** the class map of year t + 10. Training uses t = 1994–2000 only (targets up to 2010);
  the 2020 map is hindcast from 2006–2010 and the 2030 map is forecast from 2016–2020.
- **Tiles:** 64 × 64 cells; 20 % of spatial blocks (4 × 4 tiles) are held out for early stopping.
- **ConvLSTM:** two ConvLSTM layers (3 × 3 kernels, 64 channels) and a 1 × 1 convolution.
- **Temporal ViT:** 8 × 8-cell tokens per year, learned spatial and temporal position embeddings,
  4 Transformer encoder layers (embedding 128, 4 heads); attention rollout for interpretation.
- **Loss:** class-weighted cross-entropy; AdamW; early stopping (patience 6, at most 60 epochs).
- **Baselines:** persistence; CA–Markov (Markov demand + neighbourhood allocation);
  Random Forest on the five input years and the 3 × 3 neighbourhood composition.
- **Metrics (area-weighted):** overall accuracy, kappa, quantity and allocation disagreement,
  per-class F1 and figure of merit of change.

## Contents

| Path | Purpose |
|---|---|
| `scripts/01_class_areas.py` | Annual class areas of the Level-1 and Level-2 maps inside the national boundary |
| `scripts/02_transitions.py` | Transition matrices, gross and net change per decade and per province |
| `scripts/03_hindcast_baselines.py` | Persistence, CA–Markov and Random Forest hindcast of 2020 with all metrics |
| `scripts/04_train_dl.py` | ConvLSTM and Temporal ViT: training, 2020 hindcast, 2030 forecast, metrics, attention |
| `scripts/05_compare_30m_250m.py` | Class areas of the 30 m and 250 m Level-1 maps (resampling bias) |
| `scripts/fig01_national_parks.py` | Figure 1: national parks (WDPA) |
| `scripts/fig02_study_area.sh` | Figure 2: study area with SRTM shaded relief (GMT) |
| `scripts/fig05_workflow.py` | Figure 5: workflow diagram |
| `scripts/fig06_area_timeseries.py` | Figure 6: annual area of the class groups |
| `scripts/fig07_transitions.py` | Figure 7: forest change map and transition matrix |
| `scripts/fig08_hindcast_2020.py` | Figure 8: hindcast maps and change errors of all models |
| `scripts/fig09_confusion.py` | Figure 9: confusion matrices and per-class F1 |
| `scripts/fig10_trajectory_2030.py` | Figure 10: class areas 1990–2030 and change per decade |
| `scripts/fig11_forecast_2030.py` | Figure 11: 2030 forecasts and model agreement |
| `scripts/fig12_attention.py` | Figure 12: attention rollout of the Temporal ViT |
| `models/` | Trained weights (`convlstm_seed0.pt`, `vit_seed0.pt`; PyTorch state dicts) |
| `results/hindcast/` | Hindcast metrics, per-class F1, confusion matrices, training logs, attention shares |
| `results/areas/` | Class areas, transition matrices, gross/net change, 30 m vs 250 m comparison |
| `results/figure_data/` | Values plotted in Figures 6 and 9–12 |

Figures 3 and 4 (annual maps) have no separate script.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export JAXA_DIR=/path/to/JAXA_HRLULC        # searched recursively for VLUCD_L2_250m_YYYY.tif
export HDX_DIR=/path/to/admin_boundaries    # contains vnm_admin0.shp and vnm_admin1.shp
export PRED_DIR=./hindcast                  # model outputs
export FIG_OUT=./figures

python3 scripts/01_class_areas.py
python3 scripts/02_transitions.py
python3 scripts/03_hindcast_baselines.py
python3 scripts/04_train_dl.py --model convlstm --seeds 1
python3 scripts/04_train_dl.py --model vit --seeds 1
python3 scripts/05_compare_30m_250m.py --years 1990 2010 2020
python3 scripts/fig08_hindcast_2020.py      # and the other figure scripts
```

`01_class_areas.py` and `02_transitions.py` write their tables next to the scripts; `fig06_area_timeseries.py`
reads `class_areas_km2.csv` from `FIG_IN` (e.g. `FIG_IN=results/areas` to use the published copy).
`04_train_dl.py` uses CUDA or the Apple GPU (MPS) when available (`--device cpu` forces the CPU);
`--smoke` runs a one-minute test on a small window. To reuse the published weights instead of
retraining, copy `models/*.pt` into `PRED_DIR` (needed by `fig12_attention.py`).
`fig02_study_area.sh` needs GMT ≥ 6.4 and GDAL. Figures use the font Nimbus Sans
(URW base-35), falling back to Helvetica.

## Data sources and licences

| Data | Source |
|---|---|
| Land cover maps | JAXA EORC, High-Resolution Land Use and Land Cover Map of Vietnam 1990–2020, version 21.09 (Level 1 and Level 2; 30 m and 250 m), https://www.eorc.jaxa.jp/ALOS/en/dataset/lulc/lulc_vnm_v2109_e.htm — free registration; not redistributed here |
| Administrative boundaries | OCHA, Viet Nam – Subnational Administrative Boundaries (COD-AB), https://data.humdata.org/dataset/cod-ab-vnm (CC BY-IGO) |
| Protected areas | UNEP-WCMC and IUCN, Protected Planet (WDPA), https://www.protectedplanet.net — downloaded by the script; not redistributed |
| Relief | SRTM 3″ via the GMT remote dataset `@earth_relief_03s` |

## Licence

Code and trained weights: MIT (see `LICENSE`). Input data: the terms of the original providers.
