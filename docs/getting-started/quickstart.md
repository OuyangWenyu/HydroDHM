# Quick Start

Get up and running with HydroDHM in minutes!

## Choose Your Model

HydroDHM offers three types of models:

=== "XAJ (Physics)"

    Best for: Physical interpretability, traditional hydrology

    ```bash
    cd hydrodhm/run_xaj
    python calibrate_xaj_unified.py --config config_camels.yaml
    ```

=== "LSTM (Deep Learning)"

    Best for: High accuracy, large datasets

    ```bash
    cd hydrodhm/run_lstm
    python train_lstm.py --config config_lstm_camels.yaml
    ```

=== "DPL-XAJ (Hybrid)"

    Best for: Combining physics and data

    ```bash
    cd hydrodhm/run_lstm
    python train_dpl_xaj.py --config config_dpl_camels.yaml
    ```

## XAJ Model Example

### 1. Prepare Configuration

```bash
cd hydrodhm/run_xaj
cp config_camels.yaml my_config.yaml
```

Edit `my_config.yaml`:

```yaml
data:
  dataset: "camels_us"
  basin_ids: ["01013500"]
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]

model:
  name: "xaj_mz"

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
```

### 2. Run Calibration

```bash
python calibrate_xaj_unified.py --config my_config.yaml
```

### 3. Evaluate Results

```bash
python evaluate_xaj_unified.py --exp xaj_experiment --eval-period test
```

### 4. Visualize

```bash
python visualize_unified.py --eval-dir results/xaj_experiment/evaluation_test
```

## LSTM Model Example

### 1. Prepare Configuration

```bash
cd hydrodhm/run_lstm
cp config_lstm_camels.yaml my_lstm.yaml
```

### 2. Train Model

```bash
# Quick test (5 epochs)
python train_lstm.py --config my_lstm.yaml --epochs 5

# Full training
python train_lstm.py --config my_lstm.yaml --epochs 50
```

### 3. Results

Training results are saved in:
```
results/lstm_experiment/
├── train_log.txt
├── model_epoch_10.pth
└── best_model.pth
```

## What's Next?

- Learn more about [XAJ Model](../models/xaj/introduction.md)
- Explore [Deep Learning Models](../models/deep-learning/introduction.md)
- Read detailed [Tutorials](../tutorials/xaj-calibration.md)
- Check [API Reference](../api/xaj.md)

## Common Issues

!!! warning "Data Not Found"
    Make sure `hydro_setting.yml` is configured correctly in your home directory.

!!! tip "GPU Training"
    For deep learning models, GPU is highly recommended. Check installation with:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

!!! info "First Run"
    CAMELS dataset will be automatically downloaded on first run (~10-20 GB).
