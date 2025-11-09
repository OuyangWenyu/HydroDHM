# Configuration

Learn how to configure HydroDHM for your specific needs.

## Configuration Files

HydroDHM uses two types of configuration:

1. **Global Configuration**: `hydro_setting.yml` (in home directory)
2. **Model Configuration**: YAML files for specific experiments

## Global Configuration

### Location

=== "Windows"
    ```
    C:\Users\YourUsername\hydro_setting.yml
    ```

=== "macOS/Linux"
    ```
    ~/hydro_setting.yml
    ```

### Structure

```yaml
local_data_path:
  # Root directory for all data
  root: 'D:\data'

  # Original datasets
  datasets-origin: 'D:\data'

  # Processed datasets
  datasets-interim: 'D:\data'

  # Cache directory
  cache: 'D:\data\.cache'

# Optional: Specific dataset paths
camels_us: 'D:\data\camels_us'
selfmadehydrodataset: 'D:\data\my_basins'
```

## XAJ Model Configuration

### Minimal Configuration

```yaml
data:
  dataset: "camels_us"
  basin_ids: ["01013500"]
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  output_dir: "results"
  experiment_name: "my_experiment"

model:
  name: "xaj_mz"

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    rep: 5000
    ngs: 100

evaluation:
  metrics: ["NSE", "KGE", "RMSE"]
```

### Full Configuration

See [config_camels.yaml](https://github.com/OuyangWenyu/HydroDHM/blob/main/hydrodhm/run_xaj/config_camels.yaml) for all options.

## Deep Learning Configuration

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

  target_vars:
    - "streamflow"

model:
  name: "CpuLSTM"
  params:
    n_hidden_states: 256
    seq_length: 270

training:
  use_gpu: false
  epochs: 50
  batch_size: 256
  optimizer: "Adam"
```

### DPL-XAJ Configuration

```yaml
data:
  dataset: "camels_us"
  forcing_vars:
    - "precipitation"
    - "potential_evapotranspiration"
  target_vars:
    - "streamflow"
    - "evapotranspiration"

model:
  name: "DplAttrXaj"
  params:
    n_output_features: 15  # XAJ parameters
    warmup_length: 30

training:
  use_gpu: true
  batch_size: 50
  optimizer: "Adadelta"
```

## Environment Variables

You can override the config file location:

```bash
export HYDRO_SETTING_FILE=/path/to/custom/hydro_setting.yml
```

## Command-Line Overrides

Most parameters can be overridden via command line:

```bash
# XAJ
python calibrate_xaj_unified.py \
    --config config.yaml \
    --output-dir custom_results \
    --experiment-name my_exp

# LSTM
python train_lstm.py \
    --config config.yaml \
    --epochs 100 \
    --batch-size 512 \
    --gpu-id 0 1
```

## Configuration Validation

Validate your configuration before running:

```bash
# XAJ dry run
python calibrate_xaj_unified.py --config config.yaml --dry-run

# Check configuration programmatically
python -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    print('✓ Configuration valid')
    print(f'Basins: {config[\"data\"][\"basin_ids\"]}')
"
```

## Best Practices

!!! tip "Start Small"
    Test with 1-2 basins and short periods first

!!! warning "Path Format"
    - Windows: Use `'D:\path'` or `'D:/path'`
    - Linux/macOS: Use `'/path/to/data'`

!!! info "Backup Configs"
    Save successful configurations for reproducibility

## Next Steps

- [XAJ Calibration Tutorial](../tutorials/xaj-calibration.md)
- [LSTM Training Tutorial](../tutorials/lstm-training.md)
- [Advanced Configuration](../advanced/hyperparameter-tuning.md)
