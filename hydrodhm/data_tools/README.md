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

# Convert the data into a cached NC format file 
python hydrodhm/data_tools/download_camels.py camels_us  --build-cache 
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


## Requirements

- Python 3.11+
- hydrodataset package (`pip install hydrodataset`)
- Configured `hydro_setting.yml` (see project README)


## Manual Download Guide

If automatic download fails due to network issues, you can manually download CAMELS datasets and process them using this tool. Follow these steps:

### Step 1: Download the Dataset

1. Visit the [CAMELS-US dataset on Zenodo](https://zenodo.org/records/15529996)
2. Download the compressed archive file(s) to your local machine. Once you have the .zip files in your target folder, you can:

   - auto-extract (will extract all ZIP files in the directory only):
     ```bash
     python hydrodhm/data_tools/download_camels.py --extract-only --data-path /path/to/CAMELS_US
 
     ```
   - Or manually extract the archives using a terminal command such as:
     ```bash
     unzip basin_set_full_res.zip -d basin_set_full_res
     ```
   Replace `basin_set_full_res.zip` and directory as appropriate for each archive file.
3. Make sure that after extraction, the folder name is exactly `CAMELS_US` (uppercase), and the folder structure matches the example below:

Example of the correct `CAMELS_US` directory structure (with one level of subfolders shown):

```
CAMELS_US/
├── basin_set_full_res/
│   ├── HCDN_nhru_final_671.shp
│   └── ... (other files)
├── basin_timeseries_v1p2_metForcing_obsFlow/
│   ├── basin_dataset_public/
│   └── basin_dataset_public_v1p2/
├── basin_timeseries_v1p2_modelOutput_nldas/
│   └── model_output_nldas/
├── basin_timeseries_v1p2_modelOutput_daymet/
│   └── model_output_daymet/
├── basin_timeseries_v1p2_modelOutput_maurer/
│   └── model_output_maurer/
├── camels_attributes_v2.0.xlsx
├── camels_attributes_v2.0.pdf
├── camels_clim.txt
├── camels_geol.txt
├── camels_hydro.txt
├── camels_name.txt
├── camels_soil.txt
├── camels_topo.txt
├── camels_vege.txt
├── readme.txt
```

### Step 2: Set the Data Directory Path

1. Configure the dataset path in your `hydro_setting.yml` file (located in your home directory), for example:

> - On Windows: `C:\Users\<YourUsername>`
> - On Linux/macOS: `/home/<yourusername>` or `/Users/<yourusername>`

`hydro_setting.yml` file contain:

   ```yaml
   local_data_path:
     datasets-origin: 'D:\data'      # The base directory for all hydro datasets
     cache: 'D:\data\.cache'        # cache directory
   ```

   > Note: The CAMELS_US dataset folder must be named `CAMELS_US` (uppercase), and should be placed at `datasets-origin/CAMELS_US` (e.g., `D:\data\CAMELS_US`).
   > Do NOT use lowercase or other variations in the folder name, or the data tools may fail to recognize the dataset.

### Step 3: Run Download Script to Verify and Process

After extracting the data, run the download script to verify the data 

```bash
# Navigate to the data tools directory
cd hydrodhm/data_tools

# Run the download script (it will detect existing data and process it)
python download_camels.py camels_us --build-cache 
```

The script will:
- Detect the manually downloaded data
- Verify the data structure
- Convert the data to NetCDF format for faster access
- Cache the processed data for future use

**Note**: The script will skip downloading if data already exists, but will process and cache it for optimal performance.

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
