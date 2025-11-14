"""
Author: zhuanglaihong
Date: 2025-10-29
LastEditTime: 2025-10-29
LastEditors: zhuanglaihong
Description: XAJ model calibration script using the latest unified architecture
FilePath: \HydroDHM\hydrodhm\run_xaj\calibrate_xaj_unified.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path
import shutil
import yaml

# Add hydromodel to path
try:
    from hydromodel.trainers.unified_calibrate import calibrate
    from hydromodel.configs.config_manager import (
        setup_configuration_from_args,
        validate_and_show_config,
        save_config_to_file,
        load_simplified_config,
    )
    from hydromodel.models.model_config import MODEL_PARAM_DICT
except ImportError:
    print("Error: hydromodel package not found. Please install it first.")
    print("You can install it with: uv pip install hydromodel")
    sys.exit(1)


def parse_arguments():
    """Parse command-line arguments - simplified version, supports configuration file"""
    parser = argparse.ArgumentParser(
        description="XAJ model calibration script - using unified architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration file format (YAML):
  data:     # data configurations
    dataset: "selfmadehydrodataset"  # dataset type
    path: "C:\\\\Users\\\\wenyu\\\\OneDrive\\\\data\\\\FD_sources"  # data path
    basin_ids: ["changdian_61561"]   # list of basin IDs
    warmup_length: 365               # warm-up period (days)
    train_period: ["2014-10-01", "2018-09-30"]  # training period
    test_period: ["2017-10-01", "2021-09-30"]   # testing period
    output_dir: "results"            # output directory
    experiment_name: "exp_xaj"       # experiment name
    cv_fold: 1                       # cross-validation fold (optional)

  model:    # model configurations
    name: "xaj_mz"                   # model type
    params:                          # model parameters
      source_type: "sources"
      source_book: "HF"
      kernel_size: 15
      time_interval_hours: 24

  training: # training configurations
    algorithm: "SCE_UA"              # algorithm type (SCE_UA/GA/scipy)
    loss: "RMSE"                     # loss function
    SCE_UA:                          # SCE_UA algorithm parameters
      random_seed: 1234
      rep: 100000                    # maximum iterations
      ngs: 100                       # number of complexes
      kstop: 50                      # stopping criterion
      peps: 0.1                      # convergence threshold
      pcento: 0.1                    # convergence percentage
    # GA:                            # GA algorithm parameters (example)
    #   random_seed: 1234
    #   run_counts: 2
    #   pop_num: 50
    #   cross_prob: 0.5
    #   mut_prob: 0.5

  evaluation: # evaluation configurations
    metrics: ["NSE", "KGE", "RMSE"]  # evaluation metrics

Usage examples:
  # Use configuration file (recommended)
  python calibrate_xaj_unified.py --config config.yaml

  # Validate configuration file
  python calibrate_xaj_unified.py --config config.yaml --dry-run

  # Override output directory
  python calibrate_xaj_unified.py --config config.yaml --output-dir new_results
        """,
    )

    # Core arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to simplified configuration file (YAML format)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate configuration, do not perform calibration",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory in configuration",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Override experiment name in configuration",
    )

    parser.add_argument(
        "--save-config",
        action="store_true",
        default=True,
        help="Save configuration file after running (enabled by default)",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    try:
        # Load from configuration file
        if args.config:
            if not os.path.exists(args.config):
                print(f"❌ Configuration file does not exist: {args.config}")
                return 1

            print(f"📄 Loading configuration file: {args.config}")
            config = load_simplified_config(args.config)
        else:
            print(
                "❌ Please provide the configuration file path using the --config argument"
            )
            print("💡 Example: python calibrate_xaj_unified.py --config config.yaml")
            return 1

        if config is None:
            print("❌ Configuration creation failed")
            return 1

        # Apply command-line overrides
        if args.output_dir:
            config["training_cfgs"]["output_dir"] = args.output_dir
            print(f"✓ Output directory overridden to: {args.output_dir}")

        if args.experiment_name:
            config["training_cfgs"]["experiment_name"] = args.experiment_name
            print(f"✓ Experiment name overridden to: {args.experiment_name}")
        # Validate configuration
        print("\n🔍 Validating configuration...")
        if not validate_and_show_config(config, verbose=True):
            print("❌ Configuration validation failed")
            return 1

        if args.dry_run:
            print("\n✅ Configuration validation completed (dry-run mode)")
            return 0

        # Perform calibration
        print("\n🚀 Starting calibration...")
        results = calibrate(config)

        # Save configuration file
        if args.save_config:
            training_cfgs = config.get("training_cfgs", {})
            output_dir = os.path.join(
                training_cfgs.get("output_dir", "results"),
                training_cfgs.get("experiment_name", "experiment"),
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save configuration file
            config_output_path = os.path.join(output_dir, "calibration_config.yaml")

            # Save param_range file (needed for evaluation)
            param_range_file = training_cfgs.get("param_range_file")
            param_range_saved = False

            if param_range_file and os.path.exists(param_range_file):
                # If a parameter file is specified and exists, copy it
                param_range_target = os.path.join(
                    output_dir, os.path.basename(param_range_file)
                )
                shutil.copy(param_range_file, param_range_target)
                # Update the path in the configuration to the filename (relative to the output directory)
                config["training_cfgs"]["param_range_file"] = os.path.basename(
                    param_range_file
                )
                param_range_saved = True
                print(f"💾 Parameter range file saved to: {param_range_target}")
            elif param_range_file is None or not os.path.exists(param_range_file):
                # If not specified or file does not exist, save the default MODEL_PARAM_DICT
                param_range_target = os.path.join(output_dir, "param_range.yaml")
                with open(param_range_target, "w", encoding="utf-8") as f:
                    yaml.dump(
                        MODEL_PARAM_DICT,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                # Update the path in the configuration to the filename (relative to the output directory)
                config["training_cfgs"]["param_range_file"] = "param_range.yaml"
                param_range_saved = True
                print(f"💾 Default parameter range saved to: {param_range_target}")

            save_config_to_file(config, config_output_path)
            print(f"💾 Configuration file saved to: {config_output_path}")

        print("\n✅ XAJ has been calibrated!")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Calibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
