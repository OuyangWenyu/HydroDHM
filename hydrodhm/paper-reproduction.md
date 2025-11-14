# Reproduction of HydrodHM Experiments in Paper "A Python Framework for Differentiable Hydrological Modeling and Research Workflow Automation"

This document provides instructions to reproduce the experiments presented in the paper "A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins" using the HydroDHM framework.

## Prerequisites

Please ensure you have completed the Installation, Configuration and Download CAMELS Dataset steps as outlined in the [HydroDHM README](../README.md).

## Single Basin LSTM

```bash
python hydrodhm/run_lstm/12025000.py

or

uv run hydrodhm/run_lstm/12025000.py
```

The results will be saved in the directory `results/lstm_results/12025000`.

## Single Basin XAJ

```bash
# calibrate and run XAJ model
python hydrodhm/run_xaj/calibrate_xaj.py --config hydrodhm/run_xaj/12025000.yaml

or 

uv run hydrodhm/run_xaj/calibrate_xaj.py --config hydrodhm/run_xaj/12025000.yaml

# evaluate XAJ model
python hydrodhm/run_xaj/evaluate_xaj_unified.py --exp xaj_results/12025000/xaj_SCE_UA

or

uv run hydrodhm/run_xaj/evaluate_xaj_unified.py --exp xaj_results/12025000/xaj_SCE_UA
```

The results will be saved in the directory `results/xaj_results/12025000`.

## Single Basin DplXaj

```bash
python hydrodhm/run_dplxaj/12025000.py

or

uv run hydrodhm/run_dplxaj/12025000.py
```

The results will be saved in the directory `results/dplxaj_results/12025000`.

## Single Basin DplnnXaj

```bash
python hydrodhm/run_dplnnxaj/12025000.py

or

uv run hydrodhm/run_dplnnxaj/12025000.py
```

The results will be saved in the directory `results/dplnnxaj_results/12025000`.

## Multi-Basin LSTM

```bash
python hydrodhm/run_lstm/camels_all.py

or

uv run hydrodhm/run_lstm/camels_all.py
```

The results will be saved in the directory `results/lstm_results/camels_all`.