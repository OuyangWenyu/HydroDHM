# Data Tools

This directory contains utilities for managing and downloading hydrological datasets.

## Available Tools

### download_camels.py

A command-line tool for downloading CAMELS datasets using the `hydrodataset` package.

**Features:**
- Automatic download via AquaFetch backend
- Support for 11 CAMELS datasets worldwide
- Automatic caching in NetCDF format
- User-friendly error messages
- Progress tracking

**Usage:**
```bash
# List available datasets
python download_camels.py --list

# Download CAMELS-US (uses path from hydro_setting.yml)
python download_camels.py camels_us

# Download to specific path
python download_camels.py camels_us --data-path D:/data/camels

# Force re-download
python download_camels.py camels_us --force
```

**Supported Datasets:**
- camels_us (United States, 671 basins)
- camels_aus (Australia, 222 basins)
- camels_br (Brazil, 897 basins)
- camels_ch (Switzerland, 331 basins)
- camels_cl (Chile, 516 basins)
- camels_gb (Great Britain, 671 basins)
- camels_de (Germany, 1555 basins)
- camels_dk (Denmark, 304 basins)
- camels_fr (France, 662 basins)
- camels_nz (New Zealand, 343 basins)
- camels_se (Sweden, 54 basins)

For detailed instructions, see [DOWNLOAD_GUIDE.md](DOWNLOAD_GUIDE.md).

## Requirements

- Python 3.9+
- hydrodataset package (`pip install hydrodataset`)
- Configured `hydro_setting.yml` (see project README)

## Quick Start

1. **Configure data path** in `~/hydro_setting.yml`:
   ```yaml
   local_data_path:
     datasets-origin: 'D:/data'
   ```

2. **Download a dataset**:
   ```bash
   cd hydrodhm/data_tools
   python download_camels.py camels_us
   ```

3. **Use in calibration**:
   ```bash
   cd ../run_xaj
   python calibrate_xaj_unified.py --config my_config.yaml
   ```

## Next Steps

After downloading data:
- Configure your calibration in `hydrodhm/run_xaj/config_example.yaml`
- Run model calibration with `calibrate_xaj_unified.py`
- Evaluate results with `evaluate_xaj_unified.py`
- Generate visualizations with `visualize_unified.py`

## Troubleshooting

**Common Issues:**

1. **HTTP 503 errors**: Server temporarily unavailable, wait and retry
2. **Timeout errors**: Use wired connection or retry during off-peak hours
3. **Permission errors**: Check write permissions for data directory
4. **Disk space**: Ensure 10-20 GB available for CAMELS-US

For more help, see [DOWNLOAD_GUIDE.md](DOWNLOAD_GUIDE.md) or open an issue on GitHub.
