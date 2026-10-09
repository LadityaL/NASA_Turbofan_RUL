# Turbofan Engine Remaining Useful Life Prediction

Predictive maintenance with **XGBoost**, **PyTorch LSTM**, and **SHAP**, using the NASA C-MAPSS FD001 benchmark.

**Author:** Aditya Kumar  
**Programme:** MSc Data Science, Queen Mary University of London  
**Supervisor:** Vasileios Klimis

## Overview

This project estimates the number of operating cycles remaining before a turbofan engine fails. It compares a feature-engineered gradient boosting model with a compact recurrent neural network using the same engine-level validation split and official test engines.

The work covers exploratory analysis, causal temporal feature engineering, sequence modelling, model evaluation, and SHAP explanations.

## Recorded results

Results below come from the submitted dissertation and saved notebook outputs; they have not been independently reproduced during repository preparation.

| Model | Test RMSE (cycles) ↓ | Test MAE (cycles) ↓ | R² ↑ | NASA score ↓ |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 18.4838 | 13.4419 | 0.8022 | 630.95 |
| LSTM: mean metrics across 3 seeds | 16.0610 | 11.5121 | 0.8501 | 403.38 |
| LSTM: averaged predictions from 3 seeds | 14.9067 | 10.8442 | 0.8713 | 308.7696 |

The mean LSTM metrics improve RMSE by **13.11%** and MAE by **14.36%** compared with XGBoost. The averaged-prediction row is a separate ensemble result, not the mean of the individual seed metrics. The engine-level comparison of averaged LSTM predictions favours LSTM for 61 engines and XGBoost for 39.

Recorded training times were approximately **0.784 seconds for XGBoost** and **40.34 seconds per LSTM seed**. These timings depend on hardware and software.

The NASA score penalises overestimated remaining life more heavily than underestimated remaining life; lower is better.

## Dataset

The included FD001 files contain 100 run-to-failure training engines, 100 truncated test engines, and one ground-truth RUL value per test engine. Each observation records an engine ID, operating cycle, three operating settings, and 21 sensor measurements. FD001 represents one operating condition and one high-pressure-compressor fault mode.

| File | Purpose |
| --- | --- |
| `train_FD001.txt` | Training engine trajectories |
| `test_FD001.txt` | Test engine trajectories |
| `RUL_FD001.txt` | Remaining life at each test engine's final observed cycle |

Dataset attribution: NASA C-MAPSS, described by Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. See the dissertation bibliography for the project's references. Dataset rights remain with their original source.

## Methodology

- **Validation:** complete engines are split into 80 training and 20 validation engines with random state 42; both models use the same engine IDs.
- **Targets:** training RUL is capped at 125 cycles. Official test RUL values are used for final evaluation.
- **XGBoost:** 93 features, including current inputs, first differences, and causal rolling statistics over 5 and 10 cycles. Early stopping uses validation data.
- **LSTM:** 30-cycle sequences with 18 variables per cycle; one recurrent layer with 64 hidden units, dropout 0.2, and a linear regression head. Scaling is fitted only on training engines. Short test histories are padded by repeating their first observation.
- **Training:** Adam optimiser, maximum 60 epochs, early stopping patience 8, and seeds 42, 7, and 123. The supplied LSTM notebook explicitly runs on CPU.
- **Interpretability:** SHAP explains the final XGBoost model's official test predictions.

## Repository contents and run order

| Order | Notebook | Purpose |
| ---: | --- | --- |
| 1 | [01_EDA_FINAL.ipynb](01_EDA_FINAL.ipynb) | Data quality checks, lifetime analysis, and sensor visualisation |
| 2 | [02_XGBoost_FINAL.ipynb](02_XGBoost_FINAL.ipynb) | Feature engineering, model training, and evaluation |
| 3 | [03_LSTM_FINAL.ipynb](03_LSTM_FINAL.ipynb) | Sequence preparation and training across three seeds |
| 4 | [04_Model_Comparison_FINAL.ipynb](04_Model_Comparison_FINAL.ipynb) | Compare saved predictions, metrics, errors, and training times |
| 5 | [05_SHAP_XGBoost_FINAL.ipynb](05_SHAP_XGBoost_FINAL.ipynb) | Global and local XGBoost explanations |

Other files: `requirements.txt`, the three FD001 data files, and `.gitignore`.

## Getting started

Download or clone the repository, open a terminal in its root folder, and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch Jupyter:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open the notebooks and run all cells in the order listed above. Keep the notebook working directory at the repository root so the data and generated outputs resolve correctly.

## Generated outputs

Running the notebooks creates `xgboost_outputs/`, `lstm_outputs/`, `comparison_outputs/`, and `shap_outputs/`. These contain metrics, predictions, plots, model checkpoints, and experiment configuration files.

The supplied archive includes executed notebooks but **does not include the generated output directories or standalone trained model files**. Run the training notebooks before model comparison or SHAP. The comparison notebook reads existing results rather than retraining models.

## Reproducibility and limitations

The original requirements are unpinned, so identical results across package versions are not guaranteed. Saved notebook outputs record the submitted experiment; retraining can produce different scores and timings. The repository preparation checked notebook JSON, Python syntax, and dataset dimensions without retraining the models.

This is a controlled study of simulated FD001 degradation. It does not establish performance on real aircraft, other fault modes, or changing operating conditions. Future work could evaluate FD002–FD004, multiple engine splits, hyperparameter tuning, and uncertainty calibration.

No new software licence has been assigned to the author's work.
