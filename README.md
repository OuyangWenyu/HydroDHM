# HydroDHM - Differentiable Hydrological Model

A PyTorch-based differentiable hydrological modeling framework for data-scarce basins, featuring the Xin'anjiang (XAJ) model with automatic calibration and evaluation tools.

This repository contains code for the paper "A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins" ([Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2024.132471)).

## Features

**Physics-Based Models:**
- Command-line tools for XAJ model calibration, evaluation, and visualization
- Multiple optimization algorithms (SCE-UA, GA, scipy)

**Deep Learning Models:**
- LSTM neural networks for streamflow prediction
- DPL-XAJ: Hybrid physics-constrained deep learning model
- Training scripts with CAMELS dataset support

**Common Features:**
- Support for CAMELS datasets and custom hydrological data
- Comprehensive evaluation metrics and publication-quality visualizations
- Built on [hydromodel](https://github.com/OuyangWenyu/hydromodel) and [torchhydro](https://github.com/OuyangWenyu/torchhydro)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/HydroDHM.git
cd HydroDHM

# Install uv (if not already installed)
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Configuration

Create `hydro_setting.yml` in your home directory:

```yaml
local_data_path:
  datasets-origin: 'D:\data'  # Update with your data path
  cache: 'D:\data\.cache'
```

### 3. Prepare Configuration File

Choose a config template based on your data:

**For CAMELS datasets:**
```bash
cd hydrodhm/run_xaj
cp config_camels.yaml my_config.yaml
```

**For custom datasets:**
```bash
cd hydrodhm/run_xaj
cp config_custom.yaml my_config.yaml
```
For detailed examples, refer to `config_camels.yaml` and `config_custom.yaml`.

Edit `my_config.yaml` to set your basin IDs, time periods, and parameters.

### 4. Run XAJ Model Workflow

**Step 1: Calibrate the model**
```bash
python calibrate_xaj_unified.py --config my_config.yaml
```

**Step 2: Evaluate on test period**
```bash
python evaluate_xaj_unified.py --exp your_experiment_name --eval-period test
```

**Step 3: Generate visualizations**
```bash
python visualize_unified.py --eval-dir results/your_experiment_name/evaluation_test
```

## Deep Learning Models

HydroDHM also provides deep learning models for streamflow prediction using [torchhydro](https://github.com/OuyangWenyu/torchhydro).

### LSTM Model

Train a standard LSTM neural network:

```bash
cd hydrodhm/run_lstm

# Train LSTM model
python lstm_camels_example.py 

```

### DPL-XAJ Model

Train a hybrid physics-ML model (LSTM + XAJ):

```bash
cd hydrodhm/run_lstm

# Train DPL-XAJ model
python dpl_xaj_example.py
```

### Model Comparison

| Model | Type | Pros | Use Case |
|-------|------|------|----------|
| **XAJ** | Physics-based | Interpretable, fewer parameters | Physical understanding, calibration |
| **LSTM** | Pure deep learning | Fast, data-driven | Large datasets, pure prediction |
| **DPL-XAJ** | Hybrid physics-ML | Combines physics & data | Best of both worlds |

See `hydrodhm/run_lstm/README.md` for more details.


## Documentation

For detailed information:

- **Physics-Based Models (XAJ)**:
  - [hydromodel package](https://github.com/OuyangWenyu/hydromodel) - Model implementations and API
  - Data format requirements for custom datasets
  - Parameter ranges and physical meanings

- **Deep Learning Models (LSTM, DPL-XAJ)**:
  - [torchhydro package](https://github.com/OuyangWenyu/torchhydro) - Deep learning framework
  - [torchhydro documentation](https://OuyangWenyu.github.io/torchhydro) - Full API reference
  - Model architectures and training strategies

- **Datasets**:
  - [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) - CAMELS and other datasets (auto-downloaded)


## License

Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
