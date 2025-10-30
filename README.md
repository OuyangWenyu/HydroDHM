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

### Using CAMELS Dataset

This project uses the `hydrodataset` package to access hydrological datasets. The package automatically downloads and caches data using the [AquaFetch](https://github.com/hyex-research/AquaFetch) backend.

#### 1. Install hydrodataset

The package should already be installed as part of the dependencies. If not:

```bash
uv pip install hydrodataset
```

#### 2. Configure Data Path

Make sure your `hydro_setting.yml` (in your home directory) has the data path configured:

```yaml
local_data_path:
  datasets-origin: 'D:\\data'  # Update with your path
  cache: 'D:\\data\\.cache'
```

#### 3. Download CAMELS Data

**Option A: Using Download Script (Recommended for beginners)**

We provide a convenient download script in `hydrodhm/data_tools/` for easy dataset management:

```bash
cd hydrodhm/data_tools

# List all available CAMELS datasets
python download_camels.py --list

# Download CAMELS-US (uses path from hydro_setting.yml)
python download_camels.py camels_us

# Or specify a custom path
python download_camels.py camels_us --data-path D:/data/camels

# Download other datasets
python download_camels.py camels_gb
python download_camels.py camels_aus
```

The script will:
- Automatically download data via hydrodataset's AquaFetch backend
- Convert to standardized NetCDF format
- Verify the download by showing basin counts
- Provide helpful guidance on next steps

See `hydrodhm/data_tools/DOWNLOAD_GUIDE.md` for detailed instructions.

**Option B: Using Python Directly**

The `hydrodataset` package handles data downloading automatically. Simply initialize the dataset class and the data will be downloaded and cached on first access:

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Get the data path from your hydro_setting.yml
data_path = SETTING["local_data_path"]["datasets-origin"]

# Initialize dataset - downloads automatically if not present
# This will fetch raw data via AquaFetch and cache it as .nc files
ds = CamelsUs(data_path)

# Access basin IDs
basin_ids = ds.read_object_ids()
print(f"Found {len(basin_ids)} basins")

# Read time-series data (streamflow, precipitation, etc.)
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)

# Read static attributes
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
```

**How it works:**
1. **First Access**: When you initialize a dataset class for the first time, `hydrodataset` uses AquaFetch to download the raw data
2. **Automatic Caching**: The data is processed into standardized NetCDF (`.nc`) format and cached in your configured data directory
3. **Fast Subsequent Access**: All future data requests read directly from the fast `.nc` cache files

**Important notes:**
- First-time download may take 30 minutes to several hours depending on the dataset
- CAMELS-US dataset is approximately 10-20 GB
- Make sure you have sufficient disk space
- A stable internet connection is recommended for initial download
- Cached `.nc` files are stored in `{data_path}/{dataset_name}/` directories

#### 4. Available CAMELS Datasets

The `hydrodataset` package supports multiple CAMELS datasets:

- `camels_us` - United States (671 basins)
- `camels_aus` - Australia (222 basins)
- `camels_gb` - Great Britain (671 basins)
- `camels_br` - Brazil (897 basins)
- `camels_ch` - Switzerland (331 basins)
- `camels_cl` - Chile (516 basins)
- `camels_de` - Germany (1555 basins)
- `camels_dk` - Denmark (304 basins)
- `camels_fr` - France (662 basins)
- `camels_nz` - New Zealand (343 basins)
- `camels_se` - Sweden (54 basins)

To use a different dataset, simply import and initialize the appropriate class:

```python
from hydrodataset.camels_gb import CamelsGb
from hydrodataset.camels_aus import CamelsAus

# Use CAMELS-GB
gb_ds = CamelsGb(data_path)

# Use CAMELS-AUS
aus_ds = CamelsAus(data_path)
```

**Standardized Variable Names:**

All CAMELS datasets use standardized variable names for consistency:
- `streamflow` - Observed streamflow
- `precipitation` - Precipitation data
- `temperature_max` / `temperature_min` - Temperature data
- `area` - Basin area
- `p_mean` - Mean annual precipitation
- And more...

This allows you to use the same code across different CAMELS datasets without modification.

For more details, see the [hydrodataset documentation](https://OuyangWenyu.github.io/hydrodataset).

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
  root: 'D:\\data'

  # Original datasets directory
  datasets-origin: 'D:\\data'

  # Interim/processed datasets directory
  datasets-interim: 'D:\\data'

  # Original basin data directory
  basins-origin: 'D:\\data'

  # Interim/processed basin data directory
  basins-interim: 'D:\\data'

  # Cache directory
  cache: 'C:\\Users\YourUsername\\.cache'

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
├── calibration_config.yaml      # Configuration used (if --save-config)
├── param_range.yaml             # Parameter ranges (if --save-config)
└── <basin_id>_sceua.csv        # SCE-UA optimization history for each basin
```

**Note:** Use the `--save-config` flag to save configuration files:
```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml --save-config
```

**Key output files:**
- `<basin_id>_sceua.csv`: Complete optimization history with all iterations and parameter values
  - The row with the minimum `like1` value contains the best parameters
- `calibration_config.yaml`: Full configuration for reproducibility and evaluation
- `param_range.yaml`: Parameter ranges (needed for evaluation to denormalize parameters)

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

## Visualization

After evaluation is complete, you can generate publication-quality figures using the unified visualization script.

### Using the Unified Visualization Script

The `visualize_unified.py` script creates comprehensive visualizations from evaluation results:

#### 1. Basic Usage (Generate All Plots)

```bash
cd hydrodhm/run_xaj
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test
```

This will generate all available plot types in the `figures/` subdirectory.

#### 2. Generate Specific Plot Types

**Time series only:**
```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --plot-types timeseries
```

**Multiple specific types:**
```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --plot-types timeseries scatter fdc
```

**Available plot types:**
- `timeseries` - Time series with precipitation (dual-axis plot)
- `scatter` - Observed vs simulated scatter plot with density coloring
- `fdc` - Flow duration curve (log scale)
- `monthly` - Monthly average comparison with error bars
- `metrics` - Multi-basin metrics comparison (only for multiple basins)
- `all` - Generate all plot types (default)

#### 3. Visualize Specific Basins

For multi-basin calibrations, select specific basins to plot:

```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --basins 01013500 01022500
```

#### 4. Custom Output Directory

Save plots to a different location:

```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --output-dir D:/my_figures
```

#### 5. Generated Figures

The script generates high-quality figures (300 DPI) suitable for publication:

```
results/your_exp_name/evaluation_test/figures/
├── timeseries_01013500.png      # Time series with precipitation
├── scatter_01013500.png          # Scatter plot with 1:1 line
├── fdc_01013500.png             # Flow duration curve
├── monthly_01013500.png         # Monthly averages
└── metrics_comparison.png       # Multi-basin comparison (if applicable)
```

**Features of generated plots:**
- **300 DPI resolution** - Ready for publication
- **Automatic metrics calculation** - NSE, KGE, RMSE, PBIAS displayed on each plot
- **Professional formatting** - Consistent color schemes and fonts
- **Period detection** - Automatically labels plots as "Training Period" or "Test Period"

#### 6. Example Workflow

Complete workflow from calibration to visualization:

```bash
# 1. Calibrate model
python calibrate_xaj_unified.py --config my_config.yaml --save-config

# 2. Evaluate on test period
python evaluate_xaj_unified.py --exp my_experiment --eval-period test

# 3. Generate all visualizations
python visualize_unified.py --eval-dir results/my_experiment/evaluation_test

# 4. Check the figures
ls results/my_experiment/evaluation_test/figures/
```

#### 7. Interpreting Results

**Time Series Plot:**
- Top panel: Precipitation bars (inverted axis)
- Bottom panel: Observed (blue solid) vs Simulated (red dashed) streamflow
- Metrics box: Quick performance assessment

**Scatter Plot:**
- Heat map shows point density (log scale)
- Black dashed line: 1:1 perfect fit
- Points above line: Model overestimates
- Points below line: Model underestimates

**Flow Duration Curve:**
- Log scale Y-axis
- Shows flow distribution across entire period
- Good match indicates correct flow regime representation

**Monthly Comparison:**
- Bar chart with error bars (standard deviation)
- Identifies seasonal biases
- Useful for water resource planning

### Tips for Better Visualization

1. **For papers/presentations**: Use default settings (300 DPI, all plot types)
2. **For quick checks**: Generate only `timeseries` and `scatter` plots
3. **For multi-basin studies**: Include the `metrics` comparison plot
4. **Save config files**: Always use `--save-config` during calibration for reproducibility


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
│   │   ├── visualize.py                  # Legacy visualization tools
│   │   └── visualize_unified.py          # New unified visualization tools
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
