# Deep Learning Models for Streamflow Prediction

This document provides detailed information about the deep learning models available in HydroDHM.

## Overview

HydroDHM provides two types of deep learning models:

1. **LSTM (Long Short-Term Memory)**: Pure data-driven neural network
2. **DPL-XAJ (Differentiable Parameter Learning + XAJ)**: Hybrid physics-constrained model

Both models are built on the [torchhydro](https://github.com/OuyangWenyu/torchhydro) framework.

## Model Architectures

### LSTM Model

**Architecture:**
```
Input (Forcings + Attributes)
    ↓
LSTM Layers (256 hidden units)
    ↓
Dense Layer
    ↓
Output (Streamflow)
```

**Key Features:**
- Sequence-to-sequence learning
- Handles temporal dependencies
- Can incorporate static basin attributes
- Fast training on GPU

**When to Use:**
- Large datasets (hundreds of basins, years of data)
- Pure prediction tasks
- When interpretability is not critical

### DPL-XAJ Model

**Architecture:**
```
Input (Forcings + Attributes)
    ↓
LSTM Layers (256 hidden units)
    ↓
Parameter Network (15 XAJ parameters)
    ↓
XAJ Hydrological Model (Physics-based)
    ↓
Output (Streamflow + ET)
```

**Key Features:**
- Neural network learns XAJ parameters dynamically
- Physics-based constraints from XAJ model
- Interpretable parameters
- Requires warmup period for hydrological states

**When to Use:**
- Need physical interpretability
- Limited data scenarios
- Want to understand hydrological processes
- Combine data-driven and physics-based approaches

## Configuration Files

### LSTM Configuration

```yaml
data:
  dataset: "camels_us"
  basin_ids: ["01013500", "01022500"]
  train_period: ["2000-10-01", "2010-09-30"]
  test_period: ["2012-10-01", "2015-09-30"]

  forcing_vars:
    - "precipitation"
    - "temperature_max"
    - "temperature_min"
    - "daylight_duration"
    - "solar_radiation"
    - "vapor_pressure"

  target_vars:
    - "streamflow"

model:
  name: "CpuLSTM"  # or "CudnnLSTM" for GPU
  params:
    n_input_features: 23    # Forcings + attributes
    n_output_features: 1     # Streamflow only
    n_hidden_states: 256     # LSTM hidden size
    seq_length: 270          # Lookback window (days)

training:
  use_gpu: false
  epochs: 50
  batch_size: 256
  optimizer: "Adam"
  loss_func: "RMSESum"
  lr_scheduler:
    0: 0.001
    10: 0.0005
    20: 0.0001
```

### DPL-XAJ Configuration

```yaml
data:
  dataset: "camels_us"
  basin_ids: ["01013500", "01022500"]
  train_period: ["1985-10-01", "1995-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]

  forcing_vars:
    - "precipitation"
    - "potential_evapotranspiration"

  target_vars:
    - "streamflow"
    - "evapotranspiration"  # Auxiliary output

model:
  name: "DplAttrXaj"
  params:
    n_input_features: 17     # Forcings + attributes
    n_output_features: 15    # XAJ parameters
    n_hidden_states: 256
    kernel_size: 15          # XAJ routing kernel
    seq_length: 60           # Forecast horizon
    warmup_length: 30        # Hydrological warmup
    param_limit_func: "clamp"

training:
  use_gpu: true  # Strongly recommended
  epochs: 50
  batch_size: 50  # Smaller than LSTM
  optimizer: "Adadelta"
  loss_func: "RMSESum"
  loss_weights: [1, 0]  # [streamflow, ET]
```

## Training Workflow

### Step 1: Prepare Data

Ensure `hydro_setting.yml` is configured:

```yaml
local_data_path:
  datasets-origin: 'D:/data'
  cache: 'D:/data/.cache'
```

### Step 2: Edit Configuration

Choose and modify a config template:

```bash
cd hydrodhm/run_lstm
cp config_lstm_camels.yaml my_lstm_config.yaml
# Edit my_lstm_config.yaml
```

### Step 3: Train Model

```bash
# LSTM
python train_lstm.py --config my_lstm_config.yaml

# DPL-XAJ
python train_dpl_xaj.py --config my_dpl_config.yaml
```

### Step 4: Monitor Training

Training progress is saved in the output directory:

```
results/
└── my_experiment/
    ├── train_log.txt          # Training logs
    ├── model_epoch_10.pth     # Model checkpoints
    ├── model_epoch_20.pth
    └── best_model.pth         # Best model
```

## Hyperparameter Tuning

### Important Hyperparameters

**LSTM:**
- `n_hidden_states` (128-512): Larger = more capacity, slower training
- `seq_length` (60-365): Longer = more context, more memory
- `batch_size` (64-512): Larger = faster, more memory
- `learning_rate` (0.0001-0.01): Start with 0.001

**DPL-XAJ:**
- `warmup_length` (30-365): Longer = better physics initialization
- `kernel_size` (5-30): Controls routing time scale
- `batch_size` (20-100): Smaller due to memory requirements
- `loss_weights`: Balance between streamflow and ET

### Recommended Settings

**Small dataset (<10 basins):**
```yaml
n_hidden_states: 128
batch_size: 64
epochs: 100
```

**Medium dataset (10-100 basins):**
```yaml
n_hidden_states: 256
batch_size: 256
epochs: 50
```

**Large dataset (>100 basins):**
```yaml
n_hidden_states: 512
batch_size: 512
epochs: 30
```

## Performance Comparison

Based on CAMELS-US experiments:

| Model | NSE (median) | Training Time | Interpretability |
|-------|--------------|---------------|------------------|
| XAJ (calibrated) | 0.65 | Fast (minutes) | High |
| LSTM | 0.75 | Medium (hours) | Low |
| DPL-XAJ | 0.73 | Slow (days) | High |

**Key Insights:**
- LSTM achieves best raw performance
- DPL-XAJ offers physics interpretability with near-LSTM performance
- XAJ is fastest but needs careful calibration

## Advanced Usage

### Multi-GPU Training

For large-scale experiments:

```bash
# Use multiple GPUs
python train_lstm.py --config config.yaml --gpu-id 0 1 2 3
```

### Transfer Learning

Use pretrained model on new basins:

```yaml
training:
  pretrained_model: "results/pretrained/best_model.pth"
  freeze_layers: ["lstm"]  # Only train output layer
```

### Custom Loss Functions

Combine multiple objectives:

```yaml
training:
  loss_func: "MultiOutLoss"
  loss_weights: [0.7, 0.3]  # [NSE, KGE]
```

## Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size`
- Reduce `seq_length`
- Reduce `n_hidden_states`
- Use gradient accumulation

**2. NaN Loss**
```
Loss becomes NaN during training
```
**Solutions:**
- Reduce learning rate
- Check for missing data in inputs
- Use gradient clipping
- Verify data normalization

**3. Slow Convergence**
```
Model not improving after many epochs
```
**Solutions:**
- Increase learning rate
- Check data quality
- Add more training data
- Try different optimizer (Adam vs Adadelta)

**4. Poor Generalization**
```
Good training metrics, poor test metrics
```
**Solutions:**
- Reduce model complexity (fewer hidden units)
- Add dropout
- Use more training data
- Check for data leakage

## Best Practices

### Data Preprocessing

1. **Remove bad basins**: Check for data quality issues
2. **Normalize inputs**: Use DapengScaler for hydrology
3. **Handle missing data**: Interpolate or mask
4. **Split data carefully**: Avoid temporal leakage

### Training Strategy

1. **Start small**: Test on 1-2 basins first
2. **Use validation set**: Monitor overfitting
3. **Save checkpoints**: Don't lose progress
4. **Log everything**: Track hyperparameters and metrics

### Model Selection

1. **LSTM**: Best for pure prediction, large datasets
2. **DPL-XAJ**: Best for interpretability, physics constraints
3. **XAJ**: Best for traditional hydrology, fast calibration

## References

- [torchhydro Documentation](https://OuyangWenyu.github.io/torchhydro)
- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
- [LSTM for Streamflow](https://hess.copernicus.org/articles/22/6005/2018/)
- [Differentiable Models](https://doi.org/10.1016/j.jhydrol.2024.132471)

## Getting Help

- Open an issue on [GitHub](https://github.com/OuyangWenyu/HydroDHM/issues)
- Check [torchhydro examples](https://github.com/OuyangWenyu/torchhydro/tree/main/examples)
- Read the [hydromodel documentation](https://github.com/OuyangWenyu/hydromodel)
