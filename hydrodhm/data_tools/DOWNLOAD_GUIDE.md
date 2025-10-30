# CAMELS Dataset Download Guide

Quick guide to downloading CAMELS datasets for hydrological modeling using the `download_camels.py` script.

## Quick Start

### 1. Install hydrodataset

```bash
pip install hydrodataset
# or
uv pip install hydrodataset
```

### 2. Configure Data Path (Optional)

Create `hydro_setting.yml` in your home directory:

**Windows:** `C:\Users\YourUsername\hydro_setting.yml`
**Linux/Mac:** `~/hydro_setting.yml`

```yaml
local_data_path:
  datasets-origin: 'D:\data'  # Update with your path
  cache: 'D:\data\.cache'
```

If not configured, you can specify path with `--data-path` option.

### 3. List Available Datasets

```bash
python download_camels.py --list
```

This shows all 11 available CAMELS datasets from different regions.

### 4. Download a Dataset

```bash
# Using path from hydro_setting.yml
python download_camels.py camels_us

# Or specify path directly
python download_camels.py camels_us --data-path D:/data/camels
```

## Available Datasets

| Dataset | Region | Basins | Size |
|---------|--------|--------|------|
| camels_us | United States | 671 | ~10-20 GB |
| camels_aus | Australia | 222 | ~5-10 GB |
| camels_gb | Great Britain | 671 | ~10-15 GB |
| camels_br | Brazil | 897 | ~15-20 GB |
| camels_ch | Switzerland | 331 | ~5-10 GB |
| camels_cl | Chile | 516 | ~10-15 GB |
| camels_de | Germany | 1555 | ~20-30 GB |
| camels_dk | Denmark | 304 | ~5-10 GB |
| camels_fr | France | 662 | ~10-15 GB |
| camels_nz | New Zealand | 343 | ~5-10 GB |
| camels_se | Sweden | 54 | ~2-5 GB |

## Usage Examples

### Download Multiple Datasets

```bash
python download_camels.py camels_us --data-path D:/data/camels
python download_camels.py camels_gb --data-path D:/data/camels
python download_camels.py camels_aus --data-path D:/data/camels
```

### Force Re-download

```bash
python download_camels.py camels_us --force
```

### Check Existing Download

```bash
# Run without --force to check if data exists
python download_camels.py camels_us
```

If data exists, it will show basin count and sample basin IDs without re-downloading.

## What Happens During Download?

1. **Automatic Fetching**: Script uses hydrodataset's AquaFetch backend to fetch raw data from official sources
2. **Format Conversion**: Data is automatically converted to standardized NetCDF (.nc) format
3. **Caching**: Cached files are stored in `{data_path}/{dataset_name}/` for fast future access
4. **Verification**: Script verifies download by reading basin IDs

## After Download

### Use in Calibration

Create a configuration file (`calibration_config.yaml`):

```yaml
data:
  dataset: "camels_us"
  path: "D:/data/camels"
  basin_ids: ["01013500", "01022500"]
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  output_dir: "results"
  experiment_name: "xaj_camels"

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
    rep: 10000
    ngs: 100
    kstop: 50
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE", "PBIAS"]
```

### Run Calibration

```bash
cd ../run_xaj
python calibrate_xaj_unified.py --config calibration_config.yaml
```

### Use from Python

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# Get data path
data_path = SETTING["local_data_path"]["datasets-origin"]

# Initialize dataset (automatically downloads if not present)
ds = CamelsUs(data_path)

# Read basin IDs
basin_ids = ds.read_object_ids()
print(f"Found {len(basin_ids)} basins")

# Read time series data
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)

# Read attributes
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
```

## Troubleshooting

### Error: "hydrodataset package not found"

```bash
pip install hydrodataset
```

### Error: "Could not find data path in hydro_setting.yml"

Either:
1. Create `hydro_setting.yml` in home directory (see step 2)
2. Use `--data-path` option:
   ```bash
   python download_camels.py camels_us --data-path D:/data/camels
   ```

### Download Interrupted

Simply run the same command again. The script will resume or verify existing data.

### Insufficient Disk Space

CAMELS datasets are large (10-20 GB for CAMELS-US). Ensure you have:
- At least 30 GB free for CAMELS-US
- At least 50 GB free for multiple datasets
- SSD recommended for faster I/O

### Slow Download

First-time downloads are slow (30 minutes to several hours) because:
- Large data volumes (10-20 GB)
- Data processing and conversion to NetCDF
- Network speed varies

**Tips:**
- Use stable wired connection
- Download during off-peak hours
- Be patient - subsequent access is very fast

### Connection Errors

1. Check internet connection
2. Verify you can access source URLs in browser (see `--list` output)
3. Check firewall settings
4. Try again later (source servers may be busy)

## Data Organization

After download, data is organized as:

```
D:/data/camels/
└── camels_us/
    ├── attributes.nc           # Basin attributes
    └── timeseries/
        ├── 01013500_lump.nc    # Time series for each basin
        ├── 01022500_lump.nc
        └── ...
```

NetCDF files contain:
- **Standardized variable names**: `streamflow`, `precipitation`, `temperature_max`, etc.
- **Metadata**: units, descriptions, time information
- **Fast access**: Optimized for xarray/pandas workflows

## Additional Resources

- **hydrodataset Documentation**: https://OuyangWenyu.github.io/hydrodataset
- **HydroDHM README**: See main README.md for complete workflow
- **CAMELS-US Source**: https://ral.ucar.edu/solutions/products/camels
- **Issue Tracker**: https://github.com/OuyangWenyu/HydroDHM/issues

## Need Help?

1. Check this guide
2. Read HydroDHM README.md
3. Check hydrodataset documentation
4. Open an issue on GitHub

Happy modeling! 🌊
