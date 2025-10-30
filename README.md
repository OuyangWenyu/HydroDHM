<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-12 15:44:30
 * @LastEditTime: 2025-10-29
 * @LastEditors: Wenyu Ouyang
 * @Description: README FOR HYDRODHM
 * @FilePath: \HydroDHM\README.md
 * Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
-->
# A Differentiable Hydrological Model in Data-Scarce Basins

This repository contains the code for the paper "A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins" ([Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2024.132471)). The code builds upon the [torchhydro](https://github.com/OuyangWenyu/torchhydro) and the [hydromodel](https://github.com/OuyangWenyu/hydromodel) packages. The former is a PyTorch-based hydrological modeling framework, while the latter provides implementations of traditional hydrological models, including the Xin'anjiang model, using Numpy.

The experiments conducted with these models can be found in the `run_xaj`, `streamflow_prediction` and `data-limited_analysis` directories. The `calculate_and_plot` directory contains the scripts used to generate the results and figures.

The model is trained and evaluated on the CAMELS dataset as well as on several basins located upstream of the Three Gorges Reservoir Area, in or near Sichuan Province, China.

For the CAMELS dataset, we will make the data available in the future. However, data sharing for the Three Gorges Reservoir Area may be subject to certain policy restrictions.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Running XAJ Calibration](#running-xaj-calibration)
- [Project Structure](#project-structure)

## Environment Setup

This project now uses `uv` for faster and more reliable dependency management, replacing the previous conda environment.

### 1. Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd HydroDHM
uv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
uv sync
```

This will install all required dependencies including PyTorch with CUDA 11.8 support.

**Note:** If you need a different CUDA version, modify the `pyproject.toml` file before running `uv sync`. See [UV_SETUP.md](UV_SETUP.md) for more details.

## Data Preparation

### Downloading CAMELS Dataset

This project uses the `hydrodataset` package to download and manage hydrological datasets.

#### 1. Install hydrodataset

The package should already be installed as part of the dependencies. If not:

```bash
uv pip install hydrodataset
```

#### 2. Download CAMELS-US Dataset

Use Python to download the dataset:

```python
from hydrodataset import Camels

# Initialize CAMELS dataset downloader
camels = Camels(data_path="path/to/your/data/directory")

# Download the dataset (this may take a while)
camels.download_data()
```

Or use the command-line interface:

```bash
python -c "from hydrodataset import Camels; Camels(data_path='D:/data/camels_us').download_data()"
```

**Recommended data directory structure:**
```
D:/data/
└── CAMELS_US/
    ├── basin_timeseries_v1p2_metForcing_obsFlow/
    ├── camels_attributes_v2.0/
    └── ...
```

#### 3. Verify Download

Check that the data has been downloaded correctly:

```python
from hydrodataset import Camels

camels = Camels(data_path="D:/data/CAMELS_US")
basin_ids = camels.read_object_ids()  # Get list of basin IDs
print(f"Found {len(basin_ids)} basins")
```

### Using Your Own Data

If you have your own hydrological data, organize it according to the `selfmadehydrodataset` format. See [hydromodel documentation](https://github.com/OuyangWenyu/hydromodel) for the required data structure.

## Configuration

### Setting up hydro_setting.yml

The `hydro_setting.yml` file configures data paths and other global settings for hydromodel packages.

#### 1. Location

Create this file in your **home directory** (same level as your user folder):

**Windows:**
```
C:\Users\YourUsername\hydro_setting.yml
```

**macOS/Linux:**
```
~/hydro_setting.yml
```

Alternatively, you can place it in the project directory and set the `HYDRO_SETTING_FILE` environment variable to point to it.

#### 2. Create the Configuration File

**Windows:**
```powershell
# Copy the example file to your home directory
cp hydro_setting.example.yml $env:USERPROFILE\hydro_setting.yml

# Edit the file
notepad $env:USERPROFILE\hydro_setting.yml
```

**macOS/Linux:**
```bash
# Copy the example file to your home directory
cp hydro_setting.example.yml ~/hydro_setting.yml

# Edit the file
nano ~/hydro_setting.yml
```

#### 3. Configuration Content

Edit the `hydro_setting.yml` file with your data paths. The file follows this structure:

```yaml
# Local data paths (Required)
local_data_path:
  # Root directory for all data
  root: 'D:\data'

  # Original datasets directory
  datasets-origin: 'D:\data'

  # Interim/processed datasets directory
  datasets-interim: 'D:\data'

  # Original basin data directory
  basins-origin: 'D:\data'

  # Interim/processed basin data directory
  basins-interim: 'D:\data'

  # Cache directory
  cache: 'C:\Users\YourUsername\.cache'

# MinIO configuration (Optional - leave empty if not using)
minio:
  server_url: ''
  client_endpoint: ''
  access_key: ''
  secret: ''

# PostgreSQL configuration (Optional - leave empty if not using)
postgres:
  server_url: ''
  port: 5432
  username: ''
  password: ''
  database: ''

# Dataset-specific paths (Optional - can override local_data_path settings)
# camels_us: 'D:\project\data\CAMELS_US'
# selfmadehydrodataset: 'D:\project\data\FD_sources'
```

**Important paths to configure:**
- `local_data_path.root`: Main directory for all hydrological data
- `local_data_path.datasets-origin`: Where downloaded raw datasets are stored
- `local_data_path.cache`: Cache directory for temporary files
- You can also directly specify dataset paths like `camels_us` or `selfmadehydrodataset` to override the default structure

**Path format notes:**
- Windows: Use single quotes with single backslashes: `'D:\path\to\data'` or forward slashes: `'D:/path/to/data'`
- Linux/macOS: Use single quotes: `'/home/user/data'`

#### 4. Verify Configuration

Test if the configuration is correctly loaded:

```python
from hydromodel import SETTING

print("Configuration loaded successfully!")
print("Root data path:", SETTING.get("local_data_path", {}).get("root", "Not configured"))
print("Cache directory:", SETTING.get("local_data_path", {}).get("cache", "Not configured"))

# If you specified dataset-specific paths:
# print("CAMELS-US path:", SETTING.get("camels_us", "Not configured"))
```

## Running XAJ Calibration

The project includes both legacy and modern calibration scripts for the XAJ model.

### Using the New Unified Calibration Script (Recommended)

#### 1. Prepare Configuration File

Copy and modify the example configuration:

```bash
cd hydrodhm/run_xaj
cp config_example.yaml my_calibration_config.yaml
```

Edit `my_calibration_config.yaml` to set your parameters:

```yaml
data:
  dataset: "camels_us"  # or "selfmadehydrodataset"
  path: "D:/path/to/data"     # or full path to your data
  basin_ids: ["01013500"]  # Basin ID(s) to calibrate
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  output_dir: "D:/results/hydro/XAJ"
  experiment_name: "camels_xaj_test"

model:
  name: "xaj_mz"
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
    time_interval_hours: 24

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    random_seed: 1234
    rep: 1000      # Increase for production runs
    ngs: 1000
    kstop: 50
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE"]
```

#### 2. Validate Configuration (Optional but Recommended)

Before running the full calibration, validate your configuration:

```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml --dry-run
```

This will check:
- Data paths are valid
- Basin IDs exist
- Time periods are properly formatted
- All required parameters are present

#### 3. Run Calibration

```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml
```

**For quick testing** (fewer iterations):
```bash
# Temporarily override parameters for quick test
python calibrate_xaj_unified.py --config my_calibration_config.yaml --experiment-name quick_test
```

#### 4. Monitor Progress

The calibration process will display:
- Current iteration number
- Best objective function value found
- Parameter values

Progress is also saved to the output directory.

#### 5. Check Results

Results are saved in the directory specified by `output_dir/experiment_name`:

```
D:/results/hydro/XAJ/camels_xaj_test/
├── calibration_config.yaml      # Configuration used
├── calibrated_params.json       # Best parameters found
├── calibration_history.csv      # Optimization history
└── plots/                       # Visualization (if enabled)
```

### Using the Legacy Script

For compatibility with older workflows, you can still use the legacy script:

```bash
python calibrate_xaj.py \
    --data_type camels \
    --data_dir camels_us \
    --basin_id 01013500 \
    --calibrate_period 1990-10-01 2000-09-30 \
    --test_period 2000-10-01 2010-09-30 \
    --exp test_calibration
```

See `calibrate_xaj.py` for all available command-line arguments.

### Multi-Basin Calibration

To calibrate multiple basins, list them in the configuration:

```yaml
data:
  basin_ids:
    - "01013500"
    - "01022500"
    - "01030500"
```

The script will calibrate each basin independently.

### Cross-Validation

Enable k-fold cross-validation in the configuration:

```yaml
data:
  cv_fold: 5  # 5-fold cross-validation
```

## Model Evaluation

After calibration is complete, you can evaluate the model performance on different time periods using the new unified evaluation interface.

### Using the Unified Evaluation Script (Recommended)

The new `evaluate_xaj_unified.py` script provides a simplified and powerful evaluation interface:

#### 1. Basic Evaluation (Test Period)

Evaluate on the test period (default):

```bash
cd hydrodhm/run_xaj
python evaluate_xaj_unified.py --exp expchangdian_61561
```

**Parameters:**
- `--exp`: Experiment name (subdirectory containing calibration results)
- `--result-dir`: Root directory where results are stored (default: `./results`)
- `--eval-period`: Which period to evaluate (`train`, `test`, or `custom`)

#### 2. Evaluate Different Periods

**Evaluate training period:**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 --eval-period train
```

**Evaluate test period (default):**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 --eval-period test
```

**Evaluate custom period:**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 \
    --eval-period custom --custom-period 2020-01-01 2021-12-31
```

#### 3. What the Script Does

The evaluation script will:
- Load the `calibration_config.yaml` from your calibration results
- Automatically find and load calibrated parameters
- Use the unified `evaluate()` API from hydromodel
- Calculate comprehensive performance metrics
- Save results in multiple formats

#### 4. Check Evaluation Results

Results are saved in subdirectories within your experiment folder:

```
results/expchangdian_61561/
├── calibration_config.yaml           # Original calibration config
├── param_range.yaml                  # Parameter ranges (if saved)
├── <basin_id>_sceua.csv             # SCE-UA calibration results
├── evaluation_test/                  # Test period evaluation results
│   ├── basins_metrics.csv            # Performance metrics (NSE, KGE, RMSE, etc.)
│   ├── basins_norm_params.csv        # Normalized parameters [0,1]
│   ├── basins_denorm_params.csv      # Physical parameter values
│   ├── xaj_mz_evaluation_results.nc  # Simulation results (NetCDF)
│   └── evaluation_info.yaml          # Evaluation metadata
├── evaluation_train/                 # Training period evaluation results
│   └── ...
└── evaluation_custom_2020-01-01_2021-12-31/  # Custom period results
    └── ...
```

**Key output files:**
- `basins_metrics.csv`: Performance metrics for each basin (NSE, KGE, RMSE, PBIAS, etc.)
- `basins_denorm_params.csv`: Calibrated parameters in physical units
- `xaj_mz_evaluation_results.nc`: Full simulation results with time series data
- `evaluation_info.yaml`: Metadata about the evaluation run

#### 5. Understanding the Metrics

The evaluation automatically calculates multiple metrics:

| Metric | Description | Range | Best Value |
|--------|-------------|-------|------------|
| NSE | Nash-Sutcliffe Efficiency | (-∞, 1] | 1.0 |
| KGE | Kling-Gupta Efficiency | (-∞, 1] | 1.0 |
| RMSE | Root Mean Square Error | [0, ∞) | 0.0 |
| PBIAS | Percent Bias | (-∞, ∞) | 0.0 |

All metrics are printed to console and saved to CSV for easy analysis.

### Using the Legacy Evaluation Script

For compatibility with older calibration results that don't use the unified config format:

```bash
python evaluate_xaj.py --exp expchangdian_61561 --result-dir results
```

**Note:** The legacy script requires the old configuration format and has limited features compared to the unified script.


## Project Structure

```
HydroDHM/
├── hydrodhm/
│   ├── run_xaj/                  # XAJ model calibration
│   │   ├── calibrate_xaj.py              # Legacy calibration script
│   │   ├── calibrate_xaj_unified.py      # New unified calibration script
│   │   ├── config_example.yaml           # Example configuration
│   │   ├── evaluate_xaj.py               # Legacy evaluation script
│   │   ├── evaluate_xaj_unified.py       # New unified evaluation script
│   │   └── visualize.py                  # Visualization tools
│   ├── streamflow_prediction/    # Neural network models
│   ├── data-limited_analysis/    # Data-limited basin experiments
│   └── calculate_and_plot/       # Result analysis and visualization
├── pyproject.toml                # Project dependencies (uv)
├── .python-version               # Python version (3.11)
├── UV_SETUP.md                   # UV environment setup guide
├── README.md                     # This file (English)
└── README_zh.md                  # Chinese version
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ouyang2024differentiable,
  title={A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins},
  author={Ouyang, Wenyu and others},
  journal={Journal of Hydrology},
  year={2024},
  doi={10.1016/j.jhydrol.2024.132471}
}
```

## License

Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.

## Troubleshooting

### Common Issues

**1. "hydromodel not found" error:**
```bash
uv pip install hydromodel hydrodataset
```

**2. PyTorch CUDA not available:**
- Check NVIDIA driver is installed
- Verify CUDA version matches (default: 11.8)
- See `pyproject.toml` to change CUDA version

**3. Data download fails:**
- Check internet connection
- Ensure sufficient disk space
- Try downloading manually from [CAMELS website](https://ral.ucar.edu/solutions/products/camels)

**4. "Basin ID not found" error:**
- Verify basin ID exists in dataset
- Check data path in `hydro_setting.yml`
- Ensure data is properly downloaded

For more help, please open an issue on the GitHub repository.
