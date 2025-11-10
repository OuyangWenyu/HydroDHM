# Usage Guide

This comprehensive guide covers everything you need to know about using HydroDHM, from data preparation to running models and understanding code structure.

## Table of Contents

- [Project Structure](#project-structure)
- [Data Management](#data-management)
- [XAJ Model Usage](#xaj-model-usage)
- [LSTM Model Usage](#lstm-model-usage)
- [DPL-XAJ Model Usage](#dpl-xaj-model-usage)
- [Working with Custom Data](#working-with-custom-data)
- [Advanced Topics](#advanced-topics)

## Project Structure

Understanding the project structure will help you navigate and use HydroDHM effectively:

```
HydroDHM/
├── hydrodhm/                          # Main package directory
│   ├── run_xaj/                       # XAJ model scripts
│   │   ├── calibrate_xaj_unified.py   # Calibration script
│   │   ├── evaluate_xaj_unified.py    # Evaluation script
│   │   ├── visualize_unified.py       # Visualization script
│   │   ├── config_camels.yaml         # CAMELS config template
│   │   └── config_custom.yaml         # Custom data config template
│   │
│   ├── run_lstm/                      # Deep learning models
│   │   ├── lstm_camels_example.py     # LSTM example
│   │   ├── dpl_xaj_example.py         # DPL-XAJ example
│   │   └── results/                   # Training results
│   │
│   ├── streamflow_prediction/         # Core prediction modules
│   │   ├── module.py                  # LSTM module definitions
│   │   └── parallel.py                # Parallel training utilities
│   │
│   ├── calculate_and_plot/            # Analysis utilities
│   │   ├── calculate_results_metrics.py
│   │   ├── plot_streamflow_timeseries.py
│   │   └── plot_losses.py
│   │
│   ├── utils/                         # Utility functions
│   │   ├── results_utils.py           # Results processing
│   │   └── results_plot_utils.py      # Plotting utilities
│   │
│   └── results/                       # Default output directory
│       └── xaj_SCE_UA/                # Example results
│
├── docs/                              # Documentation
├── pyproject.toml                     # Package configuration
└── README.md                          # Project README
```

### Key Components

- **run_xaj/**: All scripts and configurations for physics-based XAJ model
- **run_lstm/**: Deep learning model examples and training scripts
- **streamflow_prediction/**: Core neural network architectures
- **calculate_and_plot/**: Post-processing and visualization tools
- **results/**: Output directory for all experiments

## Data Management

HydroDHM uses [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) for data access, providing a unified interface to multiple hydrological datasets.

### Supported Datasets

HydroDHM works with any dataset supported by hydrodataset, including:

- **CAMELS Family**: CAMELS-US, CAMELS-AUS, CAMELS-GB, CAMELS-BR, etc.
- **Large-scale datasets**: Caravan, HYSETS, LamaH-CE
- **Regional datasets**: EStream, GRDC-Caravan
- **Custom datasets**: Your own hydrological data

### Data Download Process

#### Automatic Download (Recommended for Beginners)

On first run, HydroDHM automatically downloads data via hydrodataset:

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Data path from your hydro_setting.yml
data_path = SETTING["local_data_path"]["datasets-origin"]

# Initialize dataset (downloads automatically if needed)
ds = CamelsUs(data_path)
```

**Download Times** (approximate):
- CAMELS-US: 1-3 hours (~15GB)
- CAMELS-AUS: 30-60 minutes (~2GB)
- CAMELS-GB: 15-30 minutes (~250MB)
- Caravan: 3-6 hours (~25GB)

#### Manual Download (Recommended for Large Datasets)

For better control and faster setup:

1. **Visit hydrodataset repository**: [https://github.com/OuyangWenyu/hydrodataset](https://github.com/OuyangWenyu/hydrodataset)

2. **Find dataset links**: Check the [Supported Datasets](https://github.com/OuyangWenyu/hydrodataset#supported-datasets) table

3. **Download and place files**:
   ```bash
   # Example for CAMELS-US
   # Download from: https://zenodo.org/records/15529996
   # Extract to: D:\data\camels_us\  (Windows)
   #         or: /home/username/data/camels_us/  (Linux)
   ```

4. **Verify structure**: Ensure data is in the correct format
   ```bash
   datasets-origin/
   └── camels_us/
       ├── basin_mean_forcing/
       ├── basin_timeseries_v1p2_metForcing_obsFlow/
       └── camels_attributes_v2.0/
   ```

### Data Access Examples

#### Read Basin IDs

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

data_path = SETTING["local_data_path"]["datasets-origin"]
ds = CamelsUs(data_path)

# Get all basin IDs
basin_ids = ds.read_object_ids()
print(f"Total basins: {len(basin_ids)}")
print(f"First 5 basins: {basin_ids[:5]}")
```

#### Read Time Series Data

```python
# Read streamflow and precipitation for specific basins
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=["01013500", "01022500"],
    t_range=["1990-01-01", "2000-12-31"],
    var_lst=["streamflow", "precipitation", "potential_evapotranspiration"]
)

print(ts_data)
# Output: xarray.Dataset with dimensions (basin, time, variable)
```

#### Read Basin Attributes

```python
# Read static attributes
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=["01013500", "01022500"],
    var_lst=["area", "p_mean", "pet_mean", "aridity"]
)

print(attr_data)
```

### Standardized Variable Names

Hydrodataset uses standardized variable names across all datasets:

| Standardized Name | Description | Units |
|-------------------|-------------|-------|
| `streamflow` | River discharge | m³/s |
| `precipitation` | Rainfall | mm/day |
| `temperature_max` | Maximum temperature | °C |
| `temperature_min` | Minimum temperature | °C |
| `potential_evapotranspiration` | PET | mm/day |
| `area` | Basin area | km² |
| `p_mean` | Mean annual precipitation | mm/year |
| `pet_mean` | Mean annual PET | mm/year |

**Example**: Use the same code for different datasets:

```python
# CAMELS-US
us_ds = CamelsUs(data_path)
us_flow = us_ds.read_ts_xrdataset(
    gage_id_lst=["01013500"],
    var_lst=["streamflow"],
    t_range=["1990-01-01", "2000-12-31"]
)

# CAMELS-AUS (same API!)
aus_ds = CamelsAus(data_path)
aus_flow = aus_ds.read_ts_xrdataset(
    gage_id_lst=["224219A"],
    var_lst=["streamflow"],
    t_range=["1990-01-01", "2000-12-31"]
)
```

## XAJ Model Usage

The XAJ (Xin'anjiang) model is a conceptual hydrological model widely used for rainfall-runoff simulation. HydroDHM implements XAJ with automatic calibration via [hydromodel](https://github.com/OuyangWenyu/hydromodel).

### Model Structure

The XAJ model consists of three main modules:

1. **Evapotranspiration Module**: Three-layer evaporation model
   - Upper layer (EU)
   - Lower layer (EL)
   - Deep layer (ED)

2. **Runoff Generation Module**:
   - Tension water capacity curve (parameter B)
   - Free water storage (parameters IM, SM, EX, KI, KG)

3. **Flow Routing Module**:
   - Surface runoff routing (Rs → Qs)
   - Interflow routing (Ri → Qi)
   - Groundwater routing (Rg → Qg)

### XAJ Parameters

| Parameter | Description | Typical Range | Unit |
|-----------|-------------|---------------|------|
| K | Evapotranspiration coefficient | 0.5 - 1.5 | - |
| B | Tension water distribution | 0.1 - 0.5 | - |
| IM | Impervious area fraction | 0.0 - 0.1 | - |
| UM | Upper layer tension water capacity | 5 - 20 | mm |
| LM | Lower layer tension water capacity | 60 - 90 | mm |
| DM | Deep layer tension water capacity | 20 - 60 | mm |
| C | Deep evapotranspiration coefficient | 0.1 - 0.2 | - |
| SM | Free water capacity | 10 - 50 | mm |
| EX | Free water distribution | 1.0 - 2.0 | - |
| KI | Interflow outflow coefficient | 0.2 - 0.7 | - |
| KG | Groundwater outflow coefficient | 0.2 - 0.7 | - |
| CS | Channel recession constant | 0.2 - 0.7 | - |
| L | Lag time | 0 - 5 | hours |
| CI | Interflow recession constant | 0.5 - 0.9 | - |
| CG | Groundwater recession constant | 0.95 - 0.998 | - |

### Configuration File Explained

Here's a detailed breakdown of the XAJ configuration:

```yaml
data:
  # Dataset type: "camels_us", "camels_aus", "camels_gb", etc.
  dataset: "camels_us"

  # Path to data (from hydro_setting.yml)
  path: "D:\\data"

  # Basin IDs to calibrate
  basin_ids:
    - "01013500"
    - "01022500"

  # Training and testing periods
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]

  # Warmup period (days) - model initialization
  warmup_length: 365

  # Required input variables (CAMELS standardized names)
  variables:
    - "precipitation"
    - "potential_evapotranspiration"
    - "streamflow"  # For calibration/evaluation

  # Output settings
  output_dir: "results"
  experiment_name: "xaj_camels_example"

model:
  # Model variant: "xaj" or "xaj_mz"
  # "xaj": Traditional XAJ with CS and L routing
  # "xaj_mz": XAJ with mizuRoute-style UH routing
  name: "xaj_mz"

  params:
    source_type: "sources"      # Runoff source configuration
    source_book: "HF"           # Parameter book: "HF" or "EH"
    kernel_size: 15             # Unit hydrograph kernel size
    time_interval_hours: 24     # Time step (24 = daily)

training:
  # Calibration algorithm: "SCE_UA", "GA", "scipy"
  algorithm: "SCE_UA"

  # Loss function: "RMSE", "NSE", "KGE"
  loss: "RMSE"

  # SCE-UA specific settings
  SCE_UA:
    random_seed: 1234
    rep: 5000        # Max function evaluations
    ngs: 100         # Number of complexes
    kstop: 50        # Convergence loops
    peps: 0.1        # Convergence threshold (%)
    pcento: 0.1      # Population convergence fraction

evaluation:
  # Metrics to compute
  metrics:
    - "NSE"    # Nash-Sutcliffe Efficiency
    - "KGE"    # Kling-Gupta Efficiency
    - "RMSE"   # Root Mean Square Error
    - "PBIAS"  # Percent Bias
```

### Running XAJ Calibration

#### Basic Usage

```bash
cd hydrodhm/run_xaj
python calibrate_xaj_unified.py --config config_camels.yaml
```

#### Command-Line Options

```bash
# Specify custom output directory
python calibrate_xaj_unified.py \
    --config config_camels.yaml \
    --output-dir /path/to/results

# Override experiment name
python calibrate_xaj_unified.py \
    --config config_camels.yaml \
    --experiment-name my_custom_experiment

# Dry run (validate configuration)
python calibrate_xaj_unified.py \
    --config config_camels.yaml \
    --dry-run
```

#### Monitoring Calibration Progress

Calibration can take several hours. Monitor progress:

```bash
# Check SCE-UA output files
tail -f results/xaj_experiment/01013500_sceua.csv

# The file shows:
# - iteration number
# - current best objective function value
# - parameter values
```

### Evaluating XAJ Results

After calibration, evaluate model performance:

```bash
# Evaluate on test period
python evaluate_xaj_unified.py \
    --exp xaj_experiment \
    --eval-period test

# Evaluate on training period
python evaluate_xaj_unified.py \
    --exp xaj_experiment \
    --eval-period train

# Evaluate on custom period
python evaluate_xaj_unified.py \
    --exp xaj_experiment \
    --eval-period custom \
    --start-time "2015-01-01" \
    --end-time "2020-12-31"
```

### Visualizing XAJ Results

```bash
# Generate all plots
python visualize_unified.py \
    --eval-dir results/xaj_experiment/evaluation_test

# Specify output directory for plots
python visualize_unified.py \
    --eval-dir results/xaj_experiment/evaluation_test \
    --output-dir custom_figures
```

Generated plots include:
- **Time series plots**: Observed vs. simulated streamflow
- **Scatter plots**: 1:1 comparison
- **Flow duration curves**: Ranking of flow magnitudes
- **Monthly aggregations**: Seasonal performance

### Understanding XAJ Results

#### Calibration Results

```
results/xaj_experiment/
├── calibration_config.yaml        # Configuration used
├── calibration_results.json       # Best parameters
├── 01013500_sceua.csv            # Optimization history
└── param_range.yaml              # Parameter bounds
```

**calibration_results.json**:
```json
{
  "01013500": {
    "params": {
      "K": 1.12,
      "B": 0.35,
      "IM": 0.02,
      // ... other parameters
    },
    "objective_value": 15.3,  # RMSE value
    "success": true
  }
}
```

#### Evaluation Results

```
results/xaj_experiment/evaluation_test/
├── basins_metrics.csv           # Performance metrics
├── basins_denorm_params.csv     # Calibrated parameters
├── basins_norm_params.csv       # Normalized parameters
├── xaj_evaluation_results.nc    # Simulated streamflow (NetCDF)
└── figures/                     # Visualization plots
```

**basins_metrics.csv**:
```csv
basin_id,NSE,KGE,RMSE,PBIAS
01013500,0.78,0.81,12.5,-3.2
01022500,0.82,0.84,18.3,2.1
```

## LSTM Model Usage

HydroDHM implements LSTM models for streamflow prediction using [torchhydro](https://github.com/OuyangWenyu/torchhydro), a PyTorch-based deep learning framework for hydrology.

### LSTM Architecture

The LSTM model in HydroDHM:
- **Input**: Time series of meteorological forcings (precipitation, temperature, etc.)
- **Hidden layers**: LSTM cells with configurable hidden states
- **Output**: Predicted streamflow

### Running LSTM Training

#### Basic Example

```bash
cd hydrodhm/run_lstm
python lstm_camels_example.py
```

#### Customizing LSTM Configuration

Edit `lstm_camels_example.py`:

```python
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

args = cmd(
    # Output directory
    sub=os.path.join("results", "my_lstm_experiment"),

    # Data source
    source_cfgs={"source_name": "camels_us", "source_path": source_path},

    # GPU/CPU setting
    ctx=[0],  # [0] for GPU 0, [-1] for CPU, [0,1] for multi-GPU

    # Model architecture
    model_name="CpuLSTM",  # or "CudnnLSTM" for GPU
    model_hyperparam={
        "n_input_features": 23,      # Number of input features
        "n_output_features": 1,      # Number of outputs (streamflow)
        "n_hidden_states": 256,      # LSTM hidden size
    },

    # Training basins
    gage_id=[
        "01013500",
        "01022500",
        # Add more basin IDs
    ],

    # Training parameters
    batch_size=256,
    train_epoch=50,           # Number of training epochs
    save_epoch=10,            # Save model every N epochs

    # Sequence configuration
    forecast_length=270,      # Sequence length (days)
    warmup_length=365,        # Warmup period

    # Input/output variables
    var_t=[
        "precipitation",
        "temperature_max",
        "temperature_min",
        "potential_evapotranspiration",
        # ... more meteorological variables
    ],
    var_out=["streamflow"],

    # Time periods
    train_period=["2000-10-01", "2010-09-30"],
    valid_period=["2010-10-01", "2012-09-30"],
    test_period=["2012-10-01", "2015-09-30"],

    # Optimization
    loss_func="RMSESum",
    opt="Adam",
    lr_scheduler={0: 1e-3, 10: 5e-4, 20: 1e-4},  # Learning rate schedule
)

# Run training
config_data = default_config_file()
update_cfg(config_data, args)
train_and_evaluate(config_data)
```

### Key LSTM Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `n_hidden_states` | LSTM hidden size | 64, 128, 256 |
| `forecast_length` | Sequence length | 90, 270, 365 |
| `batch_size` | Training batch size | 64, 128, 256 |
| `train_epoch` | Training epochs | 20, 50, 100 |
| `learning_rate` | Initial learning rate | 1e-3, 5e-4 |

### LSTM Results

```
results/my_lstm_experiment/
├── 09_November_202509_47PM_model.pth   # Model weights
├── 09_November_202509_47PM.json        # Configuration
├── model_Ep50.pth                      # Epoch 50 checkpoint
├── metric_streamflow.csv               # Evaluation metrics
├── epoch50flow_pred.nc                 # Predictions
├── epoch50flow_obs.nc                  # Observations
└── dapengscaler_stat.json             # Normalization stats
```

### LSTM Configuration Details and Debugging

#### Understanding `n_input_features`

**What it is**: Total number of features fed into LSTM at each timestep

**Formula**:
```
n_input_features = len(var_t) + len(constant_attributes)
```

**For CAMELS-US with attributes**:
```python
# Example from lstm_camels_example.py
var_t=[
    StandardVariable.PRECIPITATION,           # 1
    StandardVariable.DAYLIGHT_DURATION,       # 2
    StandardVariable.SOLAR_RADIATION,         # 3
    StandardVariable.TEMPERATURE_MAX,         # 4
    StandardVariable.TEMPERATURE_MIN,         # 5
    StandardVariable.VAPOR_PRESSURE,          # 6
]  # 6 time series variables

# + basin attributes (automatically concatenated)
# CAMELS-US has ~17 static attributes
# Total: 6 + 17 = 23 input features
```

**How to verify your dataset's feature count**:
```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

ds = CamelsUs(SETTING['local_data_path']['datasets-origin'])
n_time_series = len(var_t)  # Count your variables
n_attributes = len(ds.available_static_features)  # e.g., 17 for CAMELS-US
n_input_features = n_time_series + n_attributes  # e.g., 23
```

#### Model Selection: CpuLSTM vs CudnnLSTM

| Model | Use When | Performance | Memory |
|-------|----------|-------------|--------|
| **CpuLSTM** | `ctx=[-1]` (CPU mode) | Slower | Low |
| **CudnnLSTM** | `ctx=[0]` (GPU mode) | 5-10x faster | Higher |

**Example configuration**:
```python
# For GPU
ctx=[0],
model_name="CudnnLSTM",
batch_size=64,

# For CPU
ctx=[-1],
model_name="CpuLSTM",
batch_size=8,  # Keep smaller for CPU
```

#### Sequence Length (`forecast_length`)

**Trade-offs**:

| Length | Memory | Training Time | Pattern Capture |
|--------|--------|---------------|-----------------|
| 90 days | Low | Fast | Short-term |
| 270 days | Medium | Medium | Seasonal |
| 365 days | High | Slow | Full annual cycle |

**Recommendations**:
- **Quick testing**: 90 days
- **Standard training**: 270 days
- **Seasonal basins**: 365 days

#### Batch Size and GPU Memory

**GPU memory guide**:

| GPU VRAM | Max batch_size | forecast_length | n_hidden_states |
|----------|----------------|-----------------|-----------------|
| 6 GB     | 8-16           | 270             | 256             |
| 8 GB     | 16-32          | 270             | 256             |
| 12 GB    | 32-64          | 270             | 256             |
| 24 GB    | 64-128         | 270             | 256             |

**If you encounter CUDA OOM**:
1. Reduce `batch_size` (e.g., 64 → 32 → 16)
2. Reduce `forecast_length` (e.g., 270 → 180 → 90)
3. Reduce `n_hidden_states` (e.g., 256 → 128)
4. Switch to CPU: `ctx=[-1]`, `model_name="CpuLSTM"`

### Common LSTM Errors and Solutions

#### Error 1: Input Feature Dimension Mismatch

**Error message**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2160x6 and 23x256)
                                                             ↑      ↑↑
                                                          Actual  Expected
```

**Cause**: `n_input_features` doesn't match actual input dimensions

**Solution**:
```python
# Step 1: Count your time series variables
var_t = [...]  # Your variable list
n_time_series = len(var_t)

# Step 2: Check if attributes are concatenated (default for CAMELS)
from hydrodataset.camels_us import CamelsUs
ds = CamelsUs(data_path)
n_attributes = len(ds.available_static_features)  # Usually 17 for CAMELS-US

# Step 3: Calculate total
n_input_features = n_time_series + n_attributes

# Step 4: Update configuration
model_hyperparam={
    "n_input_features": n_input_features,  # Update this value
    "n_output_features": 1,
    "n_hidden_states": 256,
}
```

#### Error 2: CUDA Out of Memory

**Error message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions (try in order)**:

**Option 1: Reduce batch size**
```python
batch_size=16,  # Reduce from 64
```

**Option 2: Reduce sequence length**
```python
forecast_length=90,  # Reduce from 270
```

**Option 3: Reduce hidden states**
```python
model_hyperparam={
    "n_hidden_states": 128,  # Reduce from 256
}
```

**Option 4: Use CPU**
```python
ctx=[-1],
model_name="CpuLSTM",
batch_size=8,
```

#### Error 3: Training Loss Becomes NaN

**Error message**:
```
Epoch 5: Training loss = nan
```

**Possible causes and solutions**:

**Cause 1: Learning rate too high**
```python
# Solution: Reduce learning rate
opt="Adam",
lr_scheduler={0: 1e-4, 10: 5e-5, 20: 1e-5},  # Start smaller
```

**Cause 2: Gradient explosion**
```python
# Solution: Add gradient clipping
gradient_clip_value=1.0,
```

**Cause 3: Data normalization issues**
```python
# Solution: Check data quality
import numpy as np

# Check for NaN or Inf values in your data
print(f"NaN in data: {np.isnan(data).sum()}")
print(f"Inf in data: {np.isinf(data).sum()}")

# Ensure scaler is properly configured
scaler="DapengScaler",
```

#### Error 4: Model Not Improving

**Symptoms**: Loss stays constant or decreases very slowly

**Diagnostic checklist**:

1. **Check learning rate**:
   ```python
   # Try higher initial learning rate
   lr_scheduler={0: 1e-3, 10: 5e-4, 20: 1e-4},
   ```

2. **Increase model capacity**:
   ```python
   model_hyperparam={
       "n_hidden_states": 512,  # Increase from 256
   }
   ```

3. **Check data quality**:
   - Verify correct train/validation/test splits
   - Ensure no data leakage
   - Check for sufficient variability in training data

4. **Train longer**:
   ```python
   train_epoch=100,  # Increase from 50
   ```

5. **Monitor validation metrics**:
   - If training loss decreases but validation doesn't → overfitting
   - If both stay high → model capacity or data issues

### LSTM Configuration Checklist

Before running, verify:

- [ ] `n_input_features` = `len(var_t)` + number of basin attributes
- [ ] `model_name` matches `ctx` setting (CpuLSTM for CPU, CudnnLSTM for GPU)
- [ ] `batch_size` fits in available GPU/CPU memory
- [ ] `forecast_length` is appropriate for your analysis (90-365 days)
- [ ] Learning rate schedule is configured
- [ ] Train/validation/test periods don't overlap
- [ ] Data path in `hydro_setting.yml` is correct
- [ ] All required variables are available in your dataset

### LSTM Performance Tuning Tips

**Start with baseline configuration**:
```python
n_hidden_states=256
batch_size=16
forecast_length=270
learning_rate=1e-3
```

**Tune one parameter at a time**:

1. **Hidden states**: Try 128, 256, 512 → monitor validation NSE
2. **Batch size**: Try 8, 16, 32, 64 → balance speed and stability
3. **Sequence length**: Try 90, 180, 270, 365 → monitor performance vs. time
4. **Learning rate**: Try different schedules → avoid NaN and slow convergence

**Monitor training with**:
```bash
# View training logs
tail -f results/lstm_camels/train_log.txt

# Check GPU memory usage
nvidia-smi -l 1  # Update every second
```

## DPL-XAJ Model Usage

DPL-XAJ (Differentiable Parameter Learning - XAJ) is a hybrid physics-ML model that combines LSTM with the XAJ hydrological model.

### DPL-XAJ Architecture

```
Meteorological     ┌─────────────┐
Forcings      ────>│    LSTM     │───> XAJ Parameters
(P, PET)           └─────────────┘     (15 parameters)
                                              │
                                              ▼
                                       ┌─────────────┐
                                       │  XAJ Model  │───> Streamflow
                                       └─────────────┘
```

The LSTM learns to predict optimal XAJ parameters dynamically based on meteorological conditions.

### Running DPL-XAJ

```bash
cd hydrodhm/run_lstm
python dpl_xaj_example.py
```

### DPL-XAJ Configuration

Key differences from standard LSTM:

```python
args = cmd(
    # DPL-specific model
    model_name="DplAttrXaj",

    model_hyperparam={
        "n_input_features": 19,
        "n_output_features": 15,      # 15 XAJ parameters
        "n_hidden_states": 256,
        "kernel_size": 15,            # XAJ routing kernel
        "warmup_length": 30,          # XAJ warmup
        "param_limit_func": "clamp",  # Keep params in valid range
    },

    # Special dataset for DPL
    dataset="DplDataset",

    # Multi-output loss function
    loss_func="MultiOutLoss",
    loss_param={
        "loss_funcs": "RMSESum",
        "item_weight": [1, 0],  # Weight for [streamflow, other outputs]
    },

    # Input/output variables
    var_t=["precipitation", "potential_evapotranspiration"],
    var_out=["streamflow", "evapotranspiration"],

    # Optimizer (Adadelta often works well for DPL)
    opt="Adadelta",
)
```

### DPL-XAJ Advantages

- **Physical constraints**: Respects hydrological principles
- **Interpretability**: XAJ parameters have physical meaning
- **Data efficiency**: Combines data-driven and physics-based approaches
- **Robustness**: Better extrapolation than pure deep learning

### DPL-XAJ Configuration Details and Debugging

#### Understanding DPL-XAJ Data Flow

**Important**: DplAttrXaj has a **different** data flow than standard LSTM:

```
Standard LSTM:
Time Series → LSTM → Streamflow

DPL-XAJ:
Basin Attributes (static) → ANN → XAJ Parameters (15 values)
                                          ↓
Time Series (P, PET) → XAJ Physical Model → Streamflow + ET
```

**Key differences**:
- **ANN input**: Static basin attributes (NOT time series)
- **ANN output**: 15 XAJ parameters
- **XAJ input**: Time series data (P and PET)

#### Critical Configuration: `n_input_features`

**What it is**: Number of basin **static attributes** (NOT time series variables!)

**How to determine for CAMELS-US**:
```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

ds = CamelsUs(SETTING['local_data_path']['datasets-origin'])
n_attributes = len(ds.available_static_features)
print(f"CAMELS-US has {n_attributes} attributes")  # Should be 19
```

**Common values**:
- CAMELS-US: **19** attributes
- CAMELS-AUS: 17 attributes
- CAMELS-GB: 20 attributes

**Configuration**:
```python
model_hyperparam={
    "n_input_features": 19,      # ⚠️ Must match your dataset!
    "n_output_features": 15,     # ✅ Fixed: 15 XAJ parameters
    "n_hidden_states": 256,
    "kernel_size": 15,
    "warmup_length": 30,
    "param_limit_func": "clamp",
},
```

#### Critical Configuration: `constant_only=True`

**REQUIRED**: Must set `constant_only=True` for attribute-based DPL models

**What it does**:
```python
# With constant_only=True (CORRECT for DplAttrXaj):
z_train = torch.from_numpy(self.c[basin, :]).float()  # Uses attributes

# Without constant_only=True (WRONG):
z_train = xc_norm.float()  # Uses time series → dimension error!
```

**Full configuration**:
```python
dpl_args = cmd(
    model_name="DplAttrXaj",
    dataset="DplDataset",
    constant_only=True,  # ⚠️ CRITICAL! Do not omit!
    ...
)
```

#### XAJ Parameters (15 total)

The ANN learns to predict these 15 parameters:

| # | Parameter | Description | Typical Range |
|---|-----------|-------------|---------------|
| 1 | K | Evapotranspiration coefficient | 0.5 - 1.5 |
| 2 | B | Tension water distribution | 0.1 - 0.5 |
| 3 | IM | Impervious area fraction | 0.0 - 0.1 |
| 4 | UM | Upper layer capacity | 5 - 20 mm |
| 5 | LM | Lower layer capacity | 60 - 90 mm |
| 6 | DM | Deep layer capacity | 20 - 60 mm |
| 7 | C | Deep ET coefficient | 0.1 - 0.2 |
| 8 | SM | Free water capacity | 10 - 50 mm |
| 9 | EX | Free water distribution | 1.0 - 2.0 |
| 10 | KI | Interflow outflow coef. | 0.2 - 0.7 |
| 11 | KG | Groundwater outflow coef. | 0.2 - 0.7 |
| 12 | A | Recession constant | Varies |
| 13 | Theta | Time constant | Varies |
| 14 | CI | Interflow recession | 0.5 - 0.9 |
| 15 | CG | Groundwater recession | 0.95 - 0.998 |

**Fixed value**: Always `n_output_features=15` for XAJ

#### Input Variables (`var_t`)

**For XAJ physical model**, only need 2 variables:

```python
var_t=[
    StandardVariable.PRECIPITATION,              # Required
    StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,  # Required
]
```

**Note**: These are inputs to **XAJ**, not the ANN!
- **ANN inputs**: Basin attributes (from `constant_only=True`)
- **XAJ inputs**: P and PET time series

### Common DPL-XAJ Errors and Solutions

#### Error 1: ANN Input Dimension Mismatch

**Error message**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (6000x19 and 17x256)
                                                             ↑↑      ↑↑
                                                          Actual  Expected
```

**Cause**: `n_input_features` doesn't match dataset's attribute count

**Solution**:
```python
# Step 1: Check your dataset's attributes
from hydrodataset.camels_us import CamelsUs
ds = CamelsUs(data_path)
n_attrs = len(ds.available_static_features)
print(f"Dataset has {n_attrs} attributes")

# Step 2: Update configuration
model_hyperparam={
    "n_input_features": 19,  # ← Must match n_attrs
    "n_output_features": 15,  # ← Always 15 for XAJ
    ...
}
```

#### Error 2: Missing `constant_only=True`

**Error message**:
```
IndexError: index 5 is out of bounds for dimension 1 with size 5
```

or

```
RuntimeError: The size of tensor a (50) must match the size of tensor b (15)
```

**Cause**: Dataset is using time series instead of attributes for ANN input

**Solution**:
```python
dpl_args = cmd(
    model_name="DplAttrXaj",
    dataset="DplDataset",
    constant_only=True,  # ← Add this line!
    ...
)
```

#### Error 3: Wrong Number of Time Series Variables

**Error**: XAJ model fails or gives poor results

**Cause**: Using wrong variables for XAJ input

**Correct configuration**:
```python
# For XAJ, ONLY need these 2:
var_t=[
    StandardVariable.PRECIPITATION,
    StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
]

# DO NOT add other variables like temperature, etc.
# They are not used by XAJ physical model
```

#### Error 4: NaN Values During Training

**Error message**:
```
ValueError: Error: NaN values detected. Check your data firstly!!!
```

**Possible causes and solutions**:

**Cause 1: Warmup period too short**
```python
# Solution: Increase warmup
warmup_length=365,  # Use 1 year instead of 30 days
```

**Cause 2: Learning rate too high**
```python
# Solution: Use Adadelta (more stable)
opt="Adadelta",  # Recommended for DPL models
```

**Cause 3: Data quality issues**
```python
# Solution: Check for NaN in data
import numpy as np
prcp_data = ds.read_ts_xrdataset(...)
print(f"NaN count: {np.isnan(prcp_data).sum()}")
```

**Cause 4: Parameter constraints**
```python
# Solution: Use clamp instead of sigmoid
param_limit_func="clamp",  # More stable than "sigmoid"
```

### DPL-XAJ Complete Configuration Example

```python
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
import os

def main():
    source_path = SETTING["local_data_path"]["datasets-origin"]

    dpl_args = cmd(
        # Output directory
        sub=os.path.join("results", "dpl_xaj_camels"),

        # Data source
        source_cfgs={"source_name": "camels_us", "source_path": source_path},

        # GPU/CPU
        ctx=[0],  # [0] for GPU, [-1] for CPU

        # Model configuration
        model_name="DplAttrXaj",
        model_hyperparam={
            "n_input_features": 19,      # ⚠️ CAMELS-US attribute count
            "n_output_features": 15,     # ✅ Fixed: 15 XAJ parameters
            "n_hidden_states": 256,      # ANN hidden layer size
            "kernel_size": 15,           # XAJ routing kernel
            "warmup_length": 30,         # XAJ warmup period
            "param_limit_func": "clamp", # Parameter constraint method
        },

        # Dataset configuration
        dataset="DplDataset",
        constant_only=True,  # ⚠️ CRITICAL: Use attributes for ANN

        # Loss function
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],  # [streamflow_weight, ET_weight]
            "limit_part": [1],
        },

        # Data normalization
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": ["streamflow"],
            "gamma_norm_cols": [
                StandardVariable.PRECIPITATION,
                StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
            ],
            "pbm_norm": True,  # ✅ Required for physics-based models
        },

        # Basins
        gage_id=[
            "01013500",
            "01022500",
            "01030500",
            "01031500",
            "01047000",
        ],

        # Training configuration
        batch_size=50,
        train_epoch=50,
        forecast_length=60,
        warmup_length=30,

        # Time periods
        train_period=["1985-10-01", "1995-09-30"],
        test_period=["2000-10-01", "2010-09-30"],
        valid_period=None,

        # Input variables (for XAJ physical model)
        var_t=[
            StandardVariable.PRECIPITATION,
            StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
        ],

        # Output variables
        var_out=[StandardVariable.STREAMFLOW, StandardVariable.EVAPOTRANSPIRATION],
        n_output=2,

        # Optimizer
        opt="Adadelta",  # Recommended for DPL

        # Model loading
        model_loader={"load_way": "specified", "test_epoch": 50},

        # Tensor layout
        which_first_tensor="sequence",
    )

    config_data = default_config_file()
    update_cfg(config_data, dpl_args)
    train_and_evaluate(config_data)

if __name__ == "__main__":
    main()
```

### DPL-XAJ Configuration Checklist

Before running, verify:

- [ ] `n_input_features` matches dataset attribute count (19 for CAMELS-US)
- [ ] `constant_only=True` is set in configuration
- [ ] `n_output_features=15` (for XAJ parameters)
- [ ] `var_t` includes **only** Precipitation and PET
- [ ] `pbm_norm=True` in scaler_params
- [ ] `warmup_length` ≥ 30 days (365 recommended for stability)
- [ ] `opt="Adadelta"` (more stable than Adam for DPL)
- [ ] Data path is correctly configured in `hydro_setting.yml`
- [ ] Dataset has sufficient data (at least several years)

### DPL-XAJ vs Standard LSTM Comparison

| Aspect | Standard LSTM | DPL-XAJ |
|--------|--------------|----------|
| **ANN Input** | Time series + attributes | Attributes only |
| **ANN Output** | Streamflow directly | XAJ parameters (15) |
| **Time Series Input** | Multiple variables | P and PET only |
| **`constant_only`** | False (default) | **True** (required) |
| **`n_input_features`** | Time vars + attributes | Attributes only |
| **Physical Model** | None | XAJ |
| **Interpretability** | Black box | Physical parameters |
| **Data Requirements** | Large datasets | Smaller datasets OK |
| **Training Stability** | May need tuning | More stable |
| **Best Use Case** | Pure prediction | Physical understanding |

### DPL-XAJ Performance Tips

1. **Start with longer warmup**: Use 365 days for stable initial states
2. **Use Adadelta optimizer**: More stable than Adam for DPL models
3. **Check parameter ranges**: Use `param_limit_func="clamp"` to keep params valid
4. **Monitor XAJ outputs**: Check both streamflow and ET predictions
5. **Compare with XAJ-only**: First calibrate pure XAJ to get baseline

**Debugging workflow**:
```bash
# Step 1: Verify data
python -c "from hydrodataset.camels_us import CamelsUs; ..."

# Step 2: Test with minimal config
# - Use 1-2 basins
# - Set train_epoch=2
# - Check for errors

# Step 3: Scale up gradually
# - Add more basins
# - Increase epochs
# - Monitor validation metrics
```

## Working with Custom Data

HydroDHM supports custom hydrological datasets. Here's how to prepare and use your own data.

### Data Format Requirements

#### For XAJ Model

Create a configuration file `config_custom.yaml`:

```yaml
data:
  dataset: "selfmadehydrodataset"
  path: "/path/to/your/data"
  basin_ids: ["basin_001", "basin_002"]
  train_period: ["2010-01-01", "2015-12-31"]
  test_period: ["2016-01-01", "2020-12-31"]

  # Custom data must have these columns
  variables:
    - "precipitation"                 # mm/day
    - "potential_evapotranspiration"  # mm/day
    - "streamflow"                    # m³/s

model:
  name: "xaj_mz"
  params:
    # Important: provide basin area for unit conversion
    basin_area:
      basin_001: 1250.5  # km²
      basin_002: 890.3   # km²

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
```

#### Directory Structure

```
your_data_directory/
├── basin_attributes.csv
├── basin_basin_001.csv
└── basin_basin_002.csv
```

**basin_attributes.csv**:
```csv
id,name,area(km^2)
basin_001,My Basin 1,1250.5
basin_002,My Basin 2,890.3
```

**basin_basin_001.csv**:
```csv
time,prcp(mm/day),pet(mm/day),flow(m^3/s)
2010-01-01,5.2,2.1,45.3
2010-01-02,0.0,2.3,42.1
2010-01-03,12.5,1.9,55.7
...
```

### Preparing Custom Data

Use the `prepare_data.py` script:

```bash
cd scripts
python prepare_data.py --origin_data_dir /path/to/your/raw/data
```

This script will:
1. Validate data format
2. Convert units if needed
3. Create properly formatted files
4. Generate necessary index files

### Using Custom Data with XAJ

```bash
cd hydrodhm/run_xaj
python calibrate_xaj_unified.py --config config_custom.yaml
```

### Using Custom Data with LSTM

For custom data with torchhydro, you need to:

1. Create a custom dataset class
2. Implement required methods
3. Register with torchhydro

See [torchhydro documentation](https://OuyangWenyu.github.io/torchhydro) for detailed instructions.

## Advanced Topics

### Multi-Basin Calibration

Calibrate multiple basins in parallel:

```yaml
# config_multi.yaml
data:
  basin_ids:
    - "01013500"
    - "01022500"
    - "01030500"
    # ... up to hundreds of basins
```

The calibration script automatically parallelizes basin processing.

### Cross-Validation

Implement k-fold cross-validation:

```yaml
data:
  cv_fold: 5  # 5-fold cross-validation

training:
  algorithm: "SCE_UA"
  # Calibration will run 5 times with different train/test splits
```

### Hyperparameter Tuning

#### XAJ Algorithm Parameters

```yaml
training:
  algorithm: "SCE_UA"
  SCE_UA:
    rep: 10000      # Increase for better convergence
    ngs: 200        # More complexes = better global search
    kstop: 100      # Longer convergence check
```

#### LSTM Hyperparameters

```python
# Grid search example
hidden_sizes = [64, 128, 256]
batch_sizes = [64, 128, 256]
learning_rates = [1e-3, 5e-4, 1e-4]

for h in hidden_sizes:
    for b in batch_sizes:
        for lr in learning_rates:
            args = cmd(
                model_hyperparam={"n_hidden_states": h},
                batch_size=b,
                lr_scheduler={0: lr},
                sub=f"results/lstm_h{h}_b{b}_lr{lr}"
            )
            # Train and evaluate
```

### Parameter Transfer

Use calibrated parameters from one basin for another:

```python
import json

# Load calibrated parameters
with open('results/exp1/calibration_results.json') as f:
    params = json.load(f)

source_params = params['01013500']['params']

# Use for another basin
# Modify config or script to use these initial parameters
```

### Ensemble Modeling

Combine multiple models:

```python
# 1. Train multiple LSTM models with different architectures
# 2. Train XAJ model
# 3. Combine predictions

lstm_pred = load_lstm_predictions()
xaj_pred = load_xaj_predictions()
dpl_pred = load_dpl_predictions()

# Simple average ensemble
ensemble_pred = (lstm_pred + xaj_pred + dpl_pred) / 3

# Weighted ensemble based on performance
weights = [0.4, 0.3, 0.3]  # Based on validation NSE
ensemble_pred = sum(w * p for w, p in zip(weights, [lstm_pred, xaj_pred, dpl_pred]))
```

### Saving and Loading Models

#### XAJ Model

```python
# Parameters are automatically saved in:
# results/experiment_name/calibration_results.json

# Load and use:
import json
with open('results/exp1/calibration_results.json') as f:
    params = json.load(f)

# Apply to new data using evaluate_xaj_unified.py
```

#### LSTM Model

```python
import torch

# Save model
torch.save(model.state_dict(), 'my_model.pth')

# Load model
model = CpuLSTM(**model_params)
model.load_state_dict(torch.load('my_model.pth'))
model.eval()

# Make predictions
predictions = model(input_data)
```

## Troubleshooting

### Common Issues

**1. Memory errors with LSTM**
```python
# Reduce batch size
batch_size=64  # instead of 256

# Reduce sequence length
forecast_length=90  # instead of 270

# Use CPU instead of GPU (uses less memory)
ctx=[-1]
```

**2. XAJ calibration not converging**
```yaml
# Increase iterations
SCE_UA:
  rep: 10000  # instead of 5000

# Adjust parameter ranges
# Edit param_range.yaml to tighten bounds
```

**3. Poor model performance**
- Check data quality (missing values, outliers)
- Ensure adequate warmup period (>365 days)
- Try different loss functions (NSE, KGE)
- Check basin area and unit conversions

## Performance Benchmarks

Typical performance on CAMELS-US basins:

| Model | Median NSE | Median KGE | Training Time (10 basins) |
|-------|------------|------------|---------------------------|
| XAJ | 0.65-0.75 | 0.70-0.80 | 2-4 hours |
| LSTM | 0.75-0.85 | 0.75-0.85 | 1-2 hours (GPU) |
| DPL-XAJ | 0.70-0.80 | 0.72-0.82 | 2-3 hours (GPU) |

Performance varies by:
- Basin characteristics (size, climate, land use)
- Data quality and length
- Model configuration
- Calibration/training effort

## Best Practices

1. **Start Small**: Test with 1-2 basins before scaling up
2. **Check Data**: Visualize input/output data before modeling
3. **Use Warmup**: Always include adequate warmup period
4. **Monitor Progress**: Check calibration/training logs regularly
5. **Validate Results**: Evaluate on independent test period
6. **Document Experiments**: Keep track of configurations and results
7. **Version Control**: Use git to track code changes
8. **Backup Results**: Save successful experiments

## Further Resources

- **hydrodataset**: [Documentation](https://OuyangWenyu.github.io/hydrodataset)
- **hydromodel**: [GitHub](https://github.com/OuyangWenyu/hydromodel)
- **torchhydro**: [Documentation](https://OuyangWenyu.github.io/torchhydro)
- **Paper**: [A Differentiable, Physics-Based Hydrological Model](https://doi.org/10.1016/j.jhydrol.2024.132471)

---

For quick examples, see the [Quick Start Guide](quickstart.md).
