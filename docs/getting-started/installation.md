# Installation

This guide will help you install HydroDHM and its dependencies.

## Requirements

- Python 3.11 or later
- Git

## Quick Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

=== "Windows"

    ```powershell
    # Install uv
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # Clone and install
    git clone https://github.com/OuyangWenyu/HydroDHM.git
    cd HydroDHM
    uv venv
    .venv\Scripts\activate
    uv sync
    ```

=== "macOS/Linux"

    ```bash
    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Clone and install
    git clone https://github.com/OuyangWenyu/HydroDHM.git
    cd HydroDHM
    uv venv
    source .venv/bin/activate
    uv sync
    ```

### Using pip

```bash
git clone https://github.com/OuyangWenyu/HydroDHM.git
cd HydroDHM
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Configuration

Create `hydro_setting.yml` in your home directory:

=== "Windows"

    File location: `C:\Users\YourUsername\hydro_setting.yml`

    ```yaml
    local_data_path:
      datasets-origin: 'D:\data'
      cache: 'D:\data\.cache'
    ```

=== "macOS/Linux"

    File location: `~/hydro_setting.yml`

    ```yaml
    local_data_path:
      datasets-origin: '/home/user/data'
      cache: '/home/user/data/.cache'
    ```

## Verify Installation

```python
from torchhydro import SETTING
from hydromodel import SETTING as HYDRO_SETTING

print("✓ HydroDHM installed successfully!")
print(f"Data path: {SETTING['local_data_path']['datasets-origin']}")
```

## GPU Support

For deep learning models, GPU acceleration is recommended:

```bash
# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration Details](configuration.md)
