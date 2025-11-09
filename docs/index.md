# HydroDHM

**Differentiable Hydrological Model for Data-Scarce Basins**

[![License](https://img.shields.io/badge/License-BSD-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://OuyangWenyu.github.io/HydroDHM)

HydroDHM is a PyTorch-based differentiable hydrological modeling framework designed for data-scarce basins. It provides both physics-based models (XAJ) and deep learning models (LSTM, DPL-XAJ) with command-line tools for easy deployment.

## Features

### 🏔️ Physics-Based Models
- **XAJ Model**: Xin'anjiang hydrological model with automatic calibration
- **Multiple Algorithms**: SCE-UA, GA, scipy optimization
- **Command-Line Tools**: No coding required

### 🧠 Deep Learning Models
- **LSTM**: Pure data-driven neural networks
- **DPL-XAJ**: Hybrid physics-constrained deep learning
- **Automatic Training**: Simple YAML configuration

### 📊 Comprehensive Analysis
- Multiple evaluation metrics (NSE, KGE, RMSE, PBIAS)
- Publication-quality visualizations
- Multi-basin analysis support

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/OuyangWenyu/HydroDHM.git
cd HydroDHM

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Run XAJ Model

```bash
cd hydrodhm/run_xaj
python calibrate_xaj_unified.py --config config_camels.yaml
python evaluate_xaj_unified.py --exp your_experiment
python visualize_unified.py --eval-dir results/your_experiment/evaluation_test
```

### Train LSTM Model

```bash
cd hydrodhm/run_lstm
python train_lstm.py --config config_lstm_camels.yaml
```

## Model Comparison

| Model | Type | Pros | Best For |
|-------|------|------|----------|
| **XAJ** | Physics-based | Interpretable, fewer parameters | Physical understanding |
| **LSTM** | Deep learning | High accuracy, data-driven | Large datasets |
| **DPL-XAJ** | Hybrid | Physics + ML | Best of both worlds |

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [XAJ Model Documentation](models/xaj/introduction.md)
- [Deep Learning Models](models/deep-learning/introduction.md)
- [API Reference](api/xaj.md)

## Citation

If you use HydroDHM in your research, please cite:

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

## Links

- [GitHub Repository](https://github.com/OuyangWenyu/HydroDHM)
- [hydromodel Package](https://github.com/OuyangWenyu/hydromodel)
- [torchhydro Package](https://github.com/OuyangWenyu/torchhydro)
- [hydrodataset Package](https://github.com/OuyangWenyu/hydrodataset)
