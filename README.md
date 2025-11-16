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

## Reproducing Experiments

Please follow the following first 3 steps before reproducing the experiments in the paper. For detailed instructions, please refer to the [Paper Reproduction Guide](hydrodhm/paper-reproduction.md).

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
uv sync --all-extras
```

### 2. Configuration

Create `hydro_setting.yml` in your home directory:

> Typical locations for the home directory:
> - On Windows: `C:\Users\<YourUsername>`
> - On Linux/macOS: `/home/<yourusername>` or `/Users/<yourusername>`

`hydro_setting.yml` must contain the following paths:

```yaml
local_data_path:
  datasets-origin: 'D:\data'  # The base directory for all hydro datasets
  cache: 'D:\data\.cache'     # cache directory    
```
  > Note: The CAMELS_US dataset folder must be named `CAMELS_US` (uppercase), and should be placed at `datasets-origin/CAMELS_US` (e.g., `D:\data\CAMELS_US`).
  > Do NOT use lowercase or other variations in the folder name, or the data tools may fail to recognize the dataset.

### 3. Download CAMELS Dataset

**Option A: Automatic Download (Recommended)**

Use our download tool to automatically fetch CAMELS-US data:

```bash
# activate the virtual environment or open another terminal
source .venv/bin/activate  # macOS/Linux

# List available CAMELS datasets
python hydrodhm/data_tools/download_camels.py --list

# If it still mentions missing module, please use uv run
uv run hydrodhm/data_tools/download_camels.py --list

# Download CAMELS-US (uses path from hydro_setting.yml)
python hydrodhm/data_tools/download_camels.py camels_us

# Download to a specific path (It will automatically create `CAMELS_US` dataset folder)
python hydrodhm/data_tools/download_camels.py camels_us --data-path path/to/data/camels

# After downloading, convert the data into a cached NC format file (uses path from hydro_setting.yml)
python hydrodhm/data_tools/download_camels.py camels_us --data-path path/to/data/camels --build-cache 
```

**Note:** First download takes 1-3 hours (~15GB). Data is cached as NetCDF files for instant future access.

**Option B: Automatic Download on First Run**

Skip this step and data will be automatically downloaded when you first run the model. However, this may interrupt your workflow.

**Option C: Manual Download**

Due to network issues, automatic download may not be possible. If automatic download fails, you can manually download the dataset from the official source and process it using our tools. For detailed step-by-step instructions, see the [Manual Download Guide](hydrodhm/data_tools/README.md#manual-download-guide) in the data tools documentation.

### 4. Prepare Configuration File

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

### 5. Run XAJ Model Workflow

>**Training time note**:
The SCE-UA calibration process is computationally intensive.
For a typical 30-year training period, running 1000 iterations for a single basin takes around 20 minutes on a standard CPU workstation.

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

### Debugging and Parameter Tuning

For detailed guidance on configuring and debugging deep learning models, see:

📖 **[LSTM Configuration and Debugging](docs/usage.md#lstm-configuration-details-and-debugging)** | **[DPL-XAJ Configuration and Debugging](docs/usage.md#dpl-xaj-configuration-details-and-debugging)**

These guides cover:
- Complete configuration examples with annotations
- Common errors and step-by-step solutions
- Parameter tuning strategies
- Dimension mismatch debugging
- Memory optimization tips
- Configuration checklists


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
