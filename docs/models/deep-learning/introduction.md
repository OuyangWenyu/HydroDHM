# Deep Learning Models Introduction

HydroDHM provides two deep learning models for streamflow prediction:

1. **LSTM**: Pure data-driven neural network
2. **DPL-XAJ**: Hybrid physics-constrained model

Both are built on [torchhydro](https://github.com/OuyangWenyu/torchhydro).

## Why Deep Learning for Hydrology?

### Advantages

✓ **High Accuracy**: State-of-the-art performance on benchmark datasets
✓ **Temporal Dependencies**: Excellent at capturing time series patterns
✓ **Scalability**: Train once on many basins
✓ **Transfer Learning**: Apply to ungauged basins

### Challenges

✗ **Data Hungry**: Requires substantial training data
✗ **Black Box**: Limited interpretability (pure LSTM)
✗ **Computational**: Needs GPU for reasonable speed
✗ **Overfitting Risk**: Can memorize instead of generalize

## Model Comparison

### LSTM

**Pure Data-Driven**

```
Inputs → LSTM Layers → Dense → Streamflow
```

**Pros**:
- Highest raw accuracy
- Fast training
- No hydrological knowledge needed

**Cons**:
- Black box
- Requires large datasets
- May violate physical constraints

### DPL-XAJ

**Physics-Constrained**

```
Inputs → LSTM → XAJ Parameters → XAJ Model → Streamflow
```

**Pros**:
- Physically interpretable
- Works with less data
- Respects hydrological constraints
- Dynamic parameter learning

**Cons**:
- Slower training
- More complex setup
- Requires GPU

## When to Use Each

| Scenario | Recommended Model |
|----------|-------------------|
| Large dataset (>100 basins, >20 years) | LSTM |
| Need physical interpretation | DPL-XAJ |
| Limited data (<10 years) | DPL-XAJ or XAJ |
| Pure prediction task | LSTM |
| Research on hybrid modeling | DPL-XAJ |
| Operational forecasting | LSTM (if data available) |

## Getting Started

### Quick Start: LSTM

```bash
cd hydrodhm/run_lstm
python train_lstm.py --config config_lstm_camels.yaml --epochs 5
```

### Quick Start: DPL-XAJ

```bash
cd hydrodhm/run_lstm
python train_dpl_xaj.py --config config_dpl_camels.yaml --epochs 5
```

## Key Concepts

### Sequence Length

How many past time steps the model looks at:

- **LSTM**: 270 days typical
- **DPL-XAJ**: 60 days typical (+ warmup)

### Warmup Period

For DPL-XAJ, initialize hydrological states:

- **Typical**: 30-365 days
- **Purpose**: Let XAJ model "spin up"

### Batch Size

Number of samples processed together:

- **LSTM**: 256-512 typical
- **DPL-XAJ**: 50-100 (memory intensive)

### Hidden Units

LSTM layer size:

- **Small models**: 128
- **Medium**: 256
- **Large**: 512

## Performance Examples

Based on CAMELS-US (671 basins):

**LSTM Results**:
- Median NSE: 0.75
- Training time: ~12 hours (V100 GPU)
- Inference: < 1 second per basin

**DPL-XAJ Results**:
- Median NSE: 0.73
- Training time: ~48 hours (V100 GPU)
- Inference: < 2 seconds per basin

## Next Steps

- [LSTM Model Details](lstm.md)
- [DPL-XAJ Model Details](dpl-xaj.md)
- [Training Guide](training.md)
- [LSTM Tutorial](../../tutorials/lstm-training.md)

## References

- Kratzert et al. (2018): [Rainfall-Runoff modelling using LSTM](https://hess.copernicus.org/articles/22/6005/2018/)
- Feng et al. (2022): [Differentiable, Learnable, Regionalized Process-Based Models](https://doi.org/10.1029/2021WR030784)
- Our Paper: [Differentiable Hydrological Model](https://doi.org/10.1016/j.jhydrol.2024.132471)
