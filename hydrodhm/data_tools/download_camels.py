"""
Author: Wenyu Ouyang
Date: 2025-10-30
LastEditTime: 2025-10-30
LastEditors: Wenyu Ouyang
Description: CAMELS dataset download tool based on hydrodataset
FilePath: \HydroDHM\hydrodhm\data_tools\download_camels.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.

This script provides a simple interface to download CAMELS datasets using
the hydrodataset package, which automatically fetches data via AquaFetch
backend and caches it in NetCDF format.

Usage:
    # List available datasets
    python download_camels.py --list

    # Download CAMELS-US (uses path from hydro_setting.yml)
    python download_camels.py camels_us

    # Download to specific path
    python download_camels.py camels_us --data-path D:/data/camels
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from hydrodataset import SETTING
    import importlib
except ImportError:
    print("\n❌ Error: hydrodataset package not found.")
    print("📦 Please install it with:")
    print("   pip install hydrodataset")
    print("   or")
    print("   uv pip install hydrodataset\n")
    sys.exit(1)


# Mapping of CAMELS dataset names to their classes
CAMELS_DATASETS = {
    "camels_us": {
        "module": "hydrodataset.camels_us",
        "class": "CamelsUs",
        "description": "CAMELS-US (United States) - 671 basins",
        "url": "https://ral.ucar.edu/solutions/products/camels",
    },
    "camels_aus": {
        "module": "hydrodataset.camels_aus",
        "class": "CamelsAus",
        "description": "CAMELS-AUS (Australia) - 222 basins",
        "url": "https://doi.org/10.4225/49/5B8D8DC67FF9C",
    },
    "camels_br": {
        "module": "hydrodataset.camels_br",
        "class": "CamelsBr",
        "description": "CAMELS-BR (Brazil) - 897 basins",
        "url": "https://doi.org/10.5281/zenodo.3964745",
    },
    "camels_ch": {
        "module": "hydrodataset.camels_ch",
        "class": "CamelsCh",
        "description": "CAMELS-CH (Switzerland) - 331 basins",
        "url": "https://doi.org/10.5194/essd-15-5755-2023",
    },
    "camels_cl": {
        "module": "hydrodataset.camels_cl",
        "class": "CamelsCl",
        "description": "CAMELS-CL (Chile) - 516 basins",
        "url": "http://camels.cr2.cl/",
    },
    "camels_gb": {
        "module": "hydrodataset.camels_gb",
        "class": "CamelsGb",
        "description": "CAMELS-GB (Great Britain) - 671 basins",
        "url": "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9",
    },
    "camels_de": {
        "module": "hydrodataset.camels_de",
        "class": "CamelsDe",
        "description": "CAMELS-DE (Germany) - 1555 basins",
        "url": "https://doi.org/10.5194/essd-16-5625-2024",
    },
    "camels_dk": {
        "module": "hydrodataset.camels_dk",
        "class": "CamelsDk",
        "description": "CAMELS-DK (Denmark) - 304 basins",
        "url": "https://doi.org/10.5194/essd-13-5671-2021",
    },
    "camels_fr": {
        "module": "hydrodataset.camels_fr",
        "class": "CamelsFr",
        "description": "CAMELS-FR (France) - 662 basins",
        "url": "https://doi.org/10.5281/zenodo.5153381",
    },
    "camels_nz": {
        "module": "hydrodataset.camels_nz",
        "class": "CamelsNz",
        "description": "CAMELS-NZ (New Zealand) - 343 basins",
        "url": "https://doi.org/10.5281/zenodo.4751009",
    },
    "camels_se": {
        "module": "hydrodataset.camels_se",
        "class": "CamelsSe",
        "description": "CAMELS-SE (Sweden) - 54 basins",
        "url": "https://doi.org/10.5194/essd-13-5743-2021",
    },
}


def list_datasets():
    """List all available CAMELS datasets."""
    print("\n" + "=" * 80)
    print("Available CAMELS Datasets")
    print("=" * 80)

    for dataset_name, info in CAMELS_DATASETS.items():
        print(f"\n📊 {dataset_name}")
        print(f"   Description: {info['description']}")
        print(f"   URL: {info['url']}")

    print("\n" + "=" * 80)
    print(f"Total: {len(CAMELS_DATASETS)} datasets available")
    print("=" * 80 + "\n")


def download_dataset(dataset_name: str, data_path: str = None, force: bool = False):
    """
    Download a CAMELS dataset using hydrodataset's automatic download.

    Parameters
    ----------
    dataset_name : str
        Name of the CAMELS dataset (e.g., 'camels_us')
    data_path : str, optional
        Path where to download the data. If None, uses path from hydro_setting.yml
    force : bool
        Force re-download even if data already exists

    Returns
    -------
    int
        0 if successful, 1 if error
    """
    if dataset_name not in CAMELS_DATASETS:
        print(f"\n❌ Error: Unknown dataset '{dataset_name}'")
        print("💡 Use --list to see available datasets\n")
        return 1

    dataset_info = CAMELS_DATASETS[dataset_name]

    # Determine data path
    if data_path is None:
        try:
            data_path = SETTING["local_data_path"]["datasets-origin"]
            print(f"\n📁 Using data path from hydro_setting.yml:")
            print(f"   {data_path}")
        except (KeyError, TypeError):
            print("\n❌ Error: Could not find data path in hydro_setting.yml")
            print("\n💡 Please either:")
            print("   1. Configure hydro_setting.yml (see README)")
            print("      Location: C:\\Users\\YourUsername\\hydro_setting.yml (Windows)")
            print("      Location: ~/hydro_setting.yml (Linux/Mac)")
            print("   2. Specify data path with --data-path option\n")
            return 1

    # Create data path if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Import dataset class
    try:
        print(f"\n📦 Initializing {dataset_name}...")
        module = importlib.import_module(dataset_info["module"])
        dataset_class = getattr(module, dataset_info["class"])
    except ImportError as e:
        print(f"\n❌ Error: Could not import {dataset_info['class']}")
        print(f"   Module: {dataset_info['module']}")
        print(f"   Error: {e}")
        print("\n💡 Make sure hydrodataset is properly installed:")
        print("   pip install hydrodataset\n")
        return 1

    # Check if data already exists
    if not force:
        try:
            # Try to read basin IDs to check if data exists
            ds = dataset_class(data_path, download=False)
            basin_ids = ds.read_object_ids()
            if basin_ids is not None and len(basin_ids) > 0:
                print(f"\n✅ Dataset already exists!")
                print(f"   Location: {data_path}")
                print(f"   Basins: {len(basin_ids)}")
                print("\n💡 Use --force to re-download\n")

                # Show sample basin IDs
                if len(basin_ids) > 0:
                    print("📋 Sample basin IDs:")
                    for i, basin_id in enumerate(basin_ids[:5]):
                        print(f"   {i+1}. {basin_id}")
                    if len(basin_ids) > 5:
                        print(f"   ... and {len(basin_ids) - 5} more\n")

                return 0
        except Exception:
            # Data doesn't exist, proceed with download
            pass

    # Download data
    print(f"\n🚀 Starting download: {dataset_info['description']}")
    print(f"   Source: {dataset_info['url']}")
    print(f"   Target: {data_path}")
    print("\n⏳ Downloading via AquaFetch backend...")
    print("   This may take 30 minutes to several hours for first download")
    print("   Data will be cached as .nc files for fast future access")
    print("   Please be patient...\n")

    try:
        # Initialize with download=True triggers automatic download
        ds = dataset_class(data_path, download=True)

        # Verify download
        basin_ids = ds.read_object_ids()

        print(f"\n✅ Download completed successfully!")
        print(f"   Location: {data_path}")
        print(f"   Basins: {len(basin_ids)}")

        # Show sample basin IDs
        print(f"\n📋 Sample basin IDs:")
        for i, basin_id in enumerate(basin_ids[:5]):
            print(f"   {i+1}. {basin_id}")
        if len(basin_ids) > 5:
            print(f"   ... and {len(basin_ids) - 5} more")

        print("\n💡 Next steps:")
        print("   - Use this dataset in your calibration config:")
        print(f"     data:")
        print(f"       dataset: \"{dataset_name}\"")
        print(f"       path: \"{data_path}\"")
        print(f"       basin_ids: [\"{basin_ids[0]}\"]  # Example\n")

        return 0

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Download failed!")
        print(f"   Error: {e}\n")

        # Check for specific error types
        if "503" in error_msg or "Service Unavailable" in error_msg:
            print("🔍 This is a temporary server issue (HTTP 503).")
            print("\n💡 Recommended actions:")
            print("   1. Wait 5-10 minutes and try again")
            print("   2. The data source server may be temporarily down or under maintenance")
            print("   3. Try again during off-peak hours (e.g., late night)")
            print("   4. Check data source status:")
            print(f"      {dataset_info['url']}")
            print("\n⏰ This is NOT a problem with your setup - just retry later!\n")
        elif "404" in error_msg or "Not Found" in error_msg:
            print("🔍 Data file not found (HTTP 404).")
            print("\n💡 Possible causes:")
            print("   1. Dataset URL may have changed")
            print("   2. Data may have been moved or removed")
            print("   3. Check the official dataset page:")
            print(f"      {dataset_info['url']}")
            print("\n📝 Consider reporting this issue on GitHub.\n")
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print("🔍 Connection timeout.")
            print("\n💡 Recommended actions:")
            print("   1. Check your internet connection")
            print("   2. Try using a wired connection instead of WiFi")
            print("   3. Retry with a more stable network")
            print("   4. Consider downloading during off-peak hours\n")
        else:
            # Generic troubleshooting
            print("💡 Troubleshooting:")
            print("   1. Check internet connection")
            print("   2. Ensure sufficient disk space (10-20 GB)")
            print("   3. Try accessing the source URL in browser:")
            print(f"      {dataset_info['url']}")
            print("   4. Check write permissions for:")
            print(f"      {data_path}")
            print("   5. If problem persists, try again in 10-30 minutes\n")

        # Show abbreviated traceback for debugging
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        else:
            print("💡 Use --verbose flag for detailed error trace\n")

        return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download CAMELS datasets using hydrodataset package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # List all available CAMELS datasets
  python download_camels.py --list

  # Download CAMELS-US (uses path from hydro_setting.yml)
  python download_camels.py camels_us

  # Download to specific directory
  python download_camels.py camels_us --data-path D:/data/camels

  # Force re-download even if data exists
  python download_camels.py camels_us --force

  # Download multiple datasets
  python download_camels.py camels_us
  python download_camels.py camels_gb
  python download_camels.py camels_aus

Notes:
------
  - First download takes 30 minutes to several hours
  - CAMELS-US is ~10-20 GB, ensure sufficient disk space
  - Data is cached as .nc files for fast future access
  - Requires stable internet connection
  - Configure hydro_setting.yml for default paths

How it works:
-------------
  This script uses hydrodataset's automatic download via AquaFetch backend.
  Data is fetched from official sources and converted to standardized NetCDF
  format with consistent variable names across all CAMELS datasets.
        """
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        choices=list(CAMELS_DATASETS.keys()),
        help="Name of the CAMELS dataset to download"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available CAMELS datasets"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Path where to download the data (default: from hydro_setting.yml)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data already exists"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        if args.list:
            list_datasets()
            return 0

        if not args.dataset:
            print("\n" + "=" * 80)
            print("CAMELS Dataset Download Tool")
            print("=" * 80)
            print("\n❌ Error: Please specify a dataset name")
            print("\n💡 Quick start:")
            print("   1. List available datasets:")
            print("      python download_camels.py --list")
            print("\n   2. Download a dataset:")
            print("      python download_camels.py camels_us")
            print("\n   3. Download to specific path:")
            print("      python download_camels.py camels_us --data-path D:/data/camels")
            print("\n" + "=" * 80 + "\n")
            return 1

        return download_dataset(
            args.dataset,
            data_path=args.data_path,
            force=args.force
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("💡 You can resume by running the same command again\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If this persists, please report at:")
        print("   https://github.com/OuyangWenyu/HydroDHM/issues\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
