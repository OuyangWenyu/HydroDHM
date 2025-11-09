"""
Author: zhuanglaihong
Date: 2025-10-29
LastEditTime: 2025-10-30
LastEditors: zhuanglaihong
Description: Evaluate calibrated XAJ model using unified configuration format
FilePath: \HydroDHM\hydrodhm\run_xaj\evaluate_xaj_unified.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

try:
    from hydromodel.trainers.unified_evaluate import evaluate
    from hydromodel.configs.config_manager import load_config_from_calibration
except ImportError:
    print("Error: hydromodel package not found or version too old.")
    print("Please install/update it with: uv pip install -U hydromodel")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate calibrated XAJ model using unified configuration format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Evaluate on test period (default)
  python evaluate_xaj_unified.py --exp xaj_SCE_UA

  # Evaluate on train period
  python evaluate_xaj_unified.py --exp xaj_SCE_UA --eval-period train

  # Evaluate on custom period
  python evaluate_xaj_unified.py --exp xaj_SCE_UA \\
      --eval-period custom --custom-period 2020-01-01 2021-12-31

  # Specify custom result directory
  python evaluate_xaj_unified.py --result-dir /path/to/results --exp my_experiment

Notes:
  - Default result directory: results/ (relative to current working directory)
  - Full path will be: results/<experiment_name>/
  - Evaluation outputs saved in: results/<experiment_name>/evaluation_<period>/
  - This script requires the unified calibration_config.yaml format
        """,
    )

    parser.add_argument(
        "--result-dir",
        dest="result_dir",
        help="Root directory of calibration results (default: results)",
        default="results",
        type=str,
    )

    parser.add_argument(
        "--exp",
        dest="exp",
        help="Experiment name (subdirectory in result_dir)",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--eval-period",
        type=str,
        choices=["train", "test", "custom"],
        default="test",
        help="Evaluation period: train (training period), test (testing period), or custom (custom period)",
    )

    parser.add_argument(
        "--custom-period",
        type=str,
        nargs=2,
        help="Custom evaluation period, format: start_date end_date (e.g., 2020-01-01 2021-12-31)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Evaluation results output directory (default: calibration-dir/evaluation_<period>)",
    )

    parser.add_argument(
        "--param-dir",
        type=str,
        help="Parameter files directory (default: use calibration-dir)",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    try:
        # Construct calibration directory path
        calibration_dir = os.path.join(args.result_dir, args.exp)
        calibration_dir = os.path.abspath(calibration_dir)

        # Check if directory exists
        if not os.path.exists(calibration_dir):
            print(f"Error: Calibration directory not found: {calibration_dir}")
            print(f"Please check the experiment name and result directory.")
            return 1

        # Load calibration configuration
        print(f"Loading configuration from: {calibration_dir}")
        config = load_config_from_calibration(calibration_dir)

        # Determine evaluation period
        if args.eval_period == "train":
            eval_period = config["data_cfgs"]["train_period"]
            period_name = "train"
        elif args.eval_period == "test":
            eval_period = config["data_cfgs"]["test_period"]
            period_name = "test"
        elif args.eval_period == "custom":
            if args.custom_period is None:
                print("Error: --custom-period required when --eval-period is 'custom'")
                return 1
            eval_period = list(args.custom_period)
            period_name = f"custom_{args.custom_period[0]}_{args.custom_period[1]}"
        else:
            print(f"Error: Invalid eval-period: {args.eval_period}")
            return 1

        print(f"Evaluating period: {eval_period}")

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(calibration_dir, f"evaluation_{period_name}")

        # Determine parameter directory
        param_dir = args.param_dir if args.param_dir else calibration_dir

        # Create evaluation configuration
        print(f"Results will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        print("\nEvaluation Configuration:")
        print(f"  Calibration directory: {calibration_dir}")
        print(f"  Parameter directory: {param_dir}")
        print(f"  Evaluation period: {eval_period} ({period_name})")
        print(f"  Output directory: {output_dir}")

        # Run evaluation
        print("\nRunning evaluation...")
        results = evaluate(
            config,
            param_dir=param_dir,
            eval_period=eval_period,
            eval_output_dir=output_dir,
        )

        # Save evaluation summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Calibration directory: {calibration_dir}")
        print(f"Evaluation period: {eval_period}")
        print(f"Output directory: {output_dir}")
        print(f"Number of basins: {len(results)}")
        print("\nBasin IDs:")
        for basin_id in results.keys():
            print(f"  - {basin_id}")
        print("=" * 80)

        # Save evaluation info
        eval_info = {
            "calibration_dir": calibration_dir,
            "param_dir": param_dir,
            "eval_period": eval_period,
            "eval_period_type": args.eval_period,
            "output_dir": output_dir,
            "basin_ids": list(results.keys()),
        }

        eval_info_file = os.path.join(output_dir, "evaluation_info.yaml")
        with open(eval_info_file, "w", encoding="utf-8") as f:
            yaml.dump(eval_info, f, allow_unicode=True)

        print(f"\nEvaluation info saved to: {eval_info_file}")
        print("\nEvaluation completed successfully!")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except KeyError as e:
        print(f"Error: Missing configuration key: {e}")
        print("Please check that the calibration configuration is complete.")
        return 1
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
