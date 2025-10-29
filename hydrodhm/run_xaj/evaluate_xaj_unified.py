"""
Author: Wenyu Ouyang
Date: 2025-10-29
LastEditTime: 2025-10-29
LastEditors: Wenyu Ouyang
Description: Evaluate calibrated XAJ model using unified configuration format
FilePath: \HydroDHM\hydrodhm\run_xaj\evaluate_xaj_unified.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from hydromodel.datasets.data_preprocess import cross_val_split_tsdata
    from hydromodel.trainers.evaluate import Evaluator, read_yaml_config
except ImportError:
    print("Error: hydromodel package not found. Please install it first.")
    print("You can install it with: uv pip install hydromodel")
    sys.exit(1)


def load_unified_config(config_path: str) -> dict:
    """Load unified configuration format and convert to evaluation parameters

    Args:
        config_path: Path to calibration_config.yaml

    Returns:
        Dictionary with parameters needed for evaluation
    """
    config = read_yaml_config(config_path)

    # Extract data configurations
    data_cfgs = config.get("data_cfgs", {})
    training_cfgs = config.get("training_cfgs", {})

    # Map data source type names to hydromodel expected format
    # The unified config may use different naming conventions
    data_source_type = data_cfgs.get("data_source_type",
                                     data_cfgs.get("dataset_name", "camels_us"))

    # Normalize data type name for hydromodel compatibility
    data_type_mapping = {
        "camels_us": "camels",
        "camels": "camels",
        "selfmadehydrodataset": "selfmadehydrodataset",
        "owndata": "owndata",
    }

    data_type = data_type_mapping.get(data_source_type.lower(), data_source_type)

    # Map unified config to evaluation parameters
    eval_params = {
        # Basin IDs - handle both single basin and list of basins
        "basin_id": data_cfgs.get("basin_ids", []),

        # Data source information
        "data_type": data_type,
        "data_dir": data_cfgs.get("data_source_path", ""),

        # Time periods
        "calibrate_period": data_cfgs.get("train_period", []),
        "test_period": data_cfgs.get("test_period", []),
        "period": None,  # Will be computed from train and test periods

        # Warmup length
        "warmup": data_cfgs.get("warmup_length", 365),

        # Cross-validation fold (default to 1 if not specified)
        "cv_fold": data_cfgs.get("cv_fold", 1),

        # Experiment name
        "experiment_name": training_cfgs.get("experiment_name", "experiment"),
    }

    # Compute overall period from train and test periods
    if eval_params["calibrate_period"] and eval_params["test_period"]:
        # Get earliest start and latest end
        all_dates = (eval_params["calibrate_period"] +
                    eval_params["test_period"])
        eval_params["period"] = [min(all_dates[::2]), max(all_dates[1::2])]

    # Add valid_period if exists
    if "valid_period" in data_cfgs:
        eval_params["valid_period"] = data_cfgs["valid_period"]

    return eval_params


def evaluate(args):
    """Main evaluation function

    Args:
        args: Command line arguments
    """
    result_dir = args.result_dir
    exp = args.exp
    cali_dir = Path(os.path.join(result_dir, exp))

    # Check if directory exists
    if not cali_dir.exists():
        print(f"Error: Calibration directory not found: {cali_dir}")
        print(f"Please check the experiment name and result directory.")
        sys.exit(1)

    config_path = os.path.join(cali_dir, "calibration_config.yaml")

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        print(f"This script requires the unified configuration format.")
        print(f"For legacy configurations, use evaluate_xaj.py instead.")
        sys.exit(1)

    # Load and convert configuration
    print(f"Loading configuration from: {config_path}")
    eval_params = load_unified_config(config_path)

    # Extract parameters
    kfold = eval_params["cv_fold"]
    basins = eval_params["basin_id"]
    warmup = eval_params["warmup"]
    data_type = eval_params["data_type"]
    data_dir = eval_params["data_dir"]
    train_period = eval_params["calibrate_period"]
    test_period = eval_params["test_period"]
    periods = eval_params["period"]

    print(f"\nEvaluation Configuration:")
    print(f"  Basin(s): {basins}")
    print(f"  Data type: {data_type}")
    print(f"  Data directory: {data_dir}")
    print(f"  Training period: {train_period}")
    print(f"  Test period: {test_period}")
    print(f"  Warmup length: {warmup} days")
    print(f"  CV folds: {kfold}")

    # Prepare data splits
    print(f"\nPreparing data splits...")
    train_and_test_data = cross_val_split_tsdata(
        data_type,
        data_dir,
        kfold,
        train_period,
        test_period,
        periods,
        warmup,
        basins,
    )

    # Evaluate based on number of folds
    if kfold <= 1:
        print(f"\nEvaluating single fold...")
        _evaluate_1fold(train_and_test_data, cali_dir)
    else:
        for fold in range(kfold):
            print(f"\n{'='*60}")
            print(f"Evaluating fold {fold+1}/{kfold}")
            print(f"{'='*60}")
            fold_dir = os.path.join(cali_dir, f"sceua_xaj_cv{fold+1}")

            # Check if fold directory exists
            if not os.path.exists(fold_dir):
                print(f"Warning: Fold directory not found: {fold_dir}")
                print(f"Skipping fold {fold+1}")
                continue

            # Evaluate both train and test period for all basins
            train_data = train_and_test_data[fold][0]
            test_data = train_and_test_data[fold][1]
            _evaluate(cali_dir, fold_dir, train_data, test_data)
            print(f"Finished evaluating fold {fold+1}")

    print(f"\n{'='*60}")
    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {cali_dir}")
    print(f"{'='*60}")


def _evaluate_1fold(train_and_test_data, cali_dir):
    """Evaluate single fold (no cross-validation)

    Args:
        train_and_test_data: Tuple of (train_data, test_data)
        cali_dir: Calibration directory path
    """
    print("Evaluating single fold...")
    train_data = train_and_test_data[0]
    test_data = train_and_test_data[1]
    param_dir = os.path.join(cali_dir, "sceua_xaj")

    # Check if parameter directory exists
    if not os.path.exists(param_dir):
        print(f"Error: Parameter directory not found: {param_dir}")
        print(f"Expected directory structure: {cali_dir}/sceua_xaj/")
        sys.exit(1)

    _evaluate(cali_dir, param_dir, train_data, test_data)
    print("Finished evaluating single fold")


def _evaluate(cali_dir, param_dir, train_data, test_data):
    """Evaluate model on training and test data

    Args:
        cali_dir: Calibration directory path
        param_dir: Parameter directory path
        train_data: Training dataset
        test_data: Test dataset
    """
    # Create evaluation directories
    eval_train_dir = os.path.join(param_dir, "train")
    eval_test_dir = os.path.join(param_dir, "test")

    os.makedirs(eval_train_dir, exist_ok=True)
    os.makedirs(eval_test_dir, exist_ok=True)

    print(f"  Evaluating training period...")
    train_eval = Evaluator(cali_dir, param_dir, eval_train_dir)
    test_eval = Evaluator(cali_dir, param_dir, eval_test_dir)

    # Run predictions
    qsim_train, qobs_train, etsim_train = train_eval.predict(train_data)

    print(f"  Evaluating test period...")
    qsim_test, qobs_test, etsim_test = test_eval.predict(test_data)

    # Save results
    print(f"  Saving results...")
    train_eval.save_results(train_data, qsim_train, qobs_train, etsim_train)
    test_eval.save_results(test_data, qsim_test, qobs_test, etsim_test)

    print(f"  Results saved to:")
    print(f"    Training: {eval_train_dir}")
    print(f"    Test: {eval_test_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate a calibrated XAJ model using unified configuration format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using default paths
  python evaluate_xaj_unified.py --exp expchangdian_61561

  # Specify custom result directory
  python evaluate_xaj_unified.py --result-dir /path/to/results --exp my_experiment

Notes:
  - This script requires the unified calibration_config.yaml format
  - For legacy configurations, use evaluate_xaj.py instead
  - Results will be saved in subdirectories: sceua_xaj/train/ and sceua_xaj/test/
        """
    )

    parser.add_argument(
        "--result-dir",
        dest="result_dir",
        help="Root directory of calibration results (default: ./results)",
        default=os.path.join(os.path.dirname(__file__), "results"),
        type=str,
    )

    parser.add_argument(
        "--exp",
        dest="exp",
        help="Experiment name (subdirectory in result_dir)",
        default="expchangdian_61561",
        type=str,
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        args = parse_arguments()
        evaluate(args)
        return 0
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
