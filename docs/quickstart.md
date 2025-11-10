# Quick Start Guide

Get started with HydroDHM in three simple steps! This guide will walk you through installation, configuration, and running your first hydrological model.

## Overview

HydroDHM provides three types of models for streamflow prediction:

| Model Type | Description | Best For | Requires GPU |
|------------|-------------|----------|--------------|
| **XAJ** | Physics-based model (Xin'anjiang) | Physical interpretability, small datasets | No |
| **LSTM** | Deep learning neural network | Large datasets, pure prediction | Optional |
| **DPL-XAJ** | Hybrid physics-ML model | Best of both worlds | Recommended |

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or higher
- Git (for cloning the repository)
- At least 20GB free disk space (for CAMELS dataset)
- (Optional) NVIDIA GPU with CUDA for deep learning models

## Step 1: Installation

### Install uv (Package Manager)

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

=== "Windows"
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Clone and Install HydroDHM

```bash
# Clone the repository
git clone https://github.com/OuyangWenyu/HydroDHM.git
cd HydroDHM

# Create virtual environment and install dependencies
uv venv
# Activate virtual environment:
# - Windows: .venv\Scripts\activate
# - macOS/Linux: source .venv/bin/activate
uv sync
```

## Step 2: Configuration

### Create Global Configuration File

Create a file named `hydro_setting.yml` in your home directory:

=== "Windows"
    Location: `C:\Users\YourUsername\hydro_setting.yml`

    ```yaml
    local_data_path:
      datasets-origin: 'D:\data'  # Update with your data path
      cache: 'D:\data\.cache'
    ```

=== "macOS/Linux"
    Location: `~/hydro_setting.yml`

    ```yaml
    local_data_path:
      datasets-origin: '/home/username/data'  # Update with your data path
      cache: '/home/username/data/.cache'
    ```

### Data Download

**Important**: On first run, HydroDHM will automatically download the CAMELS dataset via [hydrodataset](https://github.com/OuyangWenyu/hydrodataset). This can take 1-3 hours depending on your internet connection.

To manually download datasets beforehand, see the [hydrodataset documentation](https://github.com/OuyangWenyu/hydrodataset#quick-start).

## Step 3: Choose Your Model and Run

### Option A: XAJ Model (Physics-Based)

Best for beginners and understanding hydrological processes.

#### 1. Prepare Configuration

```bash
cd hydrodhm/run_xaj
cp config_camels.yaml my_config.yaml
```

Edit `my_config.yaml` to customize basins and periods:

```yaml
data:
  dataset: "camels_us"
  basin_ids:
    - "01013500"  # Add your basin IDs
    - "01022500"
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  experiment_name: "my_first_xaj"

model:
  name: "xaj_mz"

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    rep: 100  # Start with 100 for quick testing
```

#### 2. Run Calibration

```bash
python calibrate_xaj_unified.py --config my_config.yaml
```

This will:
- Download CAMELS data (first time only)
- Calibrate XAJ model parameters using SCE-UA algorithm
- Save results to `results/my_first_xaj/`

#### 3. Evaluate on Test Period

```bash
python evaluate_xaj_unified.py --exp my_first_xaj --eval-period test
```

#### 4. Generate Visualizations

```bash
python visualize_unified.py --eval-dir results/my_first_xaj/evaluation_test
```

Results will be saved as PNG files in the evaluation directory.

### Option B: LSTM Model (Deep Learning)

Best for high-accuracy predictions with large datasets.

#### 1. Navigate to LSTM Directory

```bash
cd hydrodhm/run_lstm
```

#### 2. Train LSTM Model

```bash
python lstm_camels_example.py
```

This script will:
- Load CAMELS-US data for 10 basins
- Train a standard LSTM model
- Save model checkpoints to `results/lstm_camels/`

**Configuration**: Edit `lstm_camels_example.py` to customize:
- Basin IDs (`gage_id` parameter)
- Training epochs (`train_epoch` parameter)
- Input variables (`var_t` parameter)
- GPU usage (`ctx` parameter: `[0]` for GPU, `[-1]` for CPU)

### Option C: DPL-XAJ Model (Hybrid)

Combines physical constraints with deep learning.

#### 1. Navigate to LSTM Directory

```bash
cd hydrodhm/run_lstm
```

#### 2. Train DPL-XAJ Model

```bash
python dpl_xaj_example.py
```

This hybrid model:
- Uses LSTM to predict XAJ model parameters
- Runs XAJ model with predicted parameters
- Combines data-driven learning with physical constraints

## Understanding the Results

### XAJ Results Structure

```
results/my_first_xaj/
├── calibration_config.yaml          # Configuration used
├── calibration_results.json         # Best parameters found
├── 01013500_sceua.csv              # Calibration history
├── 01022500_sceua.csv              # Calibration history
└── evaluation_test/
    ├── basins_metrics.csv          # NSE, KGE, RMSE, etc.
    ├── basins_denorm_params.csv    # Calibrated parameters
    ├── xaj_evaluation_results.nc    # Simulated streamflow
    └── figures/
        ├── timeseries_01013500.png # Time series plot
        ├── scatter_01013500.png    # Scatter plot
        ├── fdc_01013500.png        # Flow duration curve
        └── monthly_01013500.png    # Monthly comparison
```

### LSTM Results Structure

```
results/lstm_camels/
├── 09_November_202509_47PM_model.pth   # Trained model weights
├── 09_November_202509_47PM.json        # Training configuration
├── metric_streamflow.csv               # Evaluation metrics
├── epoch2flow_pred.nc                  # Predicted streamflow
├── epoch2flow_obs.nc                   # Observed streamflow
└── dapengscaler_stat.json             # Data normalization stats
```

## Next Steps

### Learn More

- **Detailed Tutorials**: See [Getting Started Guide](getting-started/quickstart.md)
- **XAJ Model**: Learn about [XAJ model structure and parameters](models/xaj/introduction.md)
- **Deep Learning**: Explore [LSTM and DPL-XAJ models](models/deep-learning/introduction.md)
- **API Reference**: Check the [API documentation](api/xaj.md)

### Advanced Usage

- **Custom Datasets**: Learn to use your own data (see Usage Guide below)
- **Multi-Basin Calibration**: Scale up to hundreds of basins
- **Hyperparameter Tuning**: Optimize model performance
- **Cross-Validation**: Implement k-fold validation

## Common Issues and Solutions

### Issue: "hydro_setting.yml not found"

**Solution**: Create the file in your home directory with the correct path settings.

```bash
# Check home directory location
echo $HOME  # Linux/macOS
echo %USERPROFILE%  # Windows
```

### Issue: "Dataset download is too slow"

**Solution**:
1. Download datasets manually from [hydrodataset sources](https://github.com/OuyangWenyu/hydrodataset#supported-datasets)
2. Place in the `datasets-origin` directory specified in `hydro_setting.yml`
3. The library will detect existing data and skip download

### Issue: "CUDA out of memory" (Deep Learning)

**Solution**:
- Reduce `batch_size` in the configuration
- Use CPU by setting `ctx=[-1]`
- Train fewer basins at once

### Issue: "SCE-UA calibration is very slow"

**Solution**:
- Reduce `rep` parameter (e.g., from 5000 to 100 for testing)
- Reduce `ngs` parameter (e.g., from 100 to 20)
- Try faster optimizers: `scipy` or `GA`

## Getting Help

- **Documentation**: [Full documentation](https://OuyangWenyu.github.io/HydroDHM)
- **Issues**: [GitHub Issues](https://github.com/OuyangWenyu/HydroDHM/issues)
- **Dependencies**:
  - [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) - Data access
  - [hydromodel](https://github.com/OuyangWenyu/hydromodel) - XAJ model
  - [torchhydro](https://github.com/OuyangWenyu/torchhydro) - Deep learning models

## Quick Reference Commands

```bash
# XAJ Model Workflow
cd hydrodhm/run_xaj
python calibrate_xaj_unified.py --config config_camels.yaml
python evaluate_xaj_unified.py --exp experiment_name --eval-period test
python visualize_unified.py --eval-dir results/experiment_name/evaluation_test

# LSTM Model
cd hydrodhm/run_lstm
python lstm_camels_example.py

# DPL-XAJ Model
cd hydrodhm/run_lstm
python dpl_xaj_example.py
```

---

For detailed usage instructions, see the [Usage Guide](usage.md).
