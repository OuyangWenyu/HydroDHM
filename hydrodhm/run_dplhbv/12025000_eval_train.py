"""
This is a simple example of training a standard LSTM model on the CAMELS-US dataset.

To run this script, you need to:
Set up your data path in your `hydro_setting.yml` file.
This file should be in your user directory. If not, create one.
Refer to the __init__.py file in the torchhydro package to ensure `local_data_path` is set correctly.
For example:
   local_data_path:
       root: 'D:/data'
       datasets-origin: 'D:/data'
"""

import os
from hydrodataset.hydro_dataset import StandardVariable
from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

from definitions_private import RESULT_DIR


def main():
    """
    This function defines the parameters for the experiment, updates the default configuration,
    and then runs the training and evaluation pipeline.
    """
    # 1. Define parameters for the experiment
    # You can refer to the `cmd` function in `torchhydro.configs.config` for more details.
    source_path = SETTING["local_data_path"]["datasets-origin"]
    args = cmd(
        # Evaluation mode
        train_mode=False,
        stat_dict_file=os.path.join(
            RESULT_DIR,
            "dplhbv_results",
            "12025000",
            "dapengscaler_stat.json",
        ),
        model_loader={
            "load_way": "pth",
            "pth_path": os.path.join(
                RESULT_DIR,
                "dplhbv_results",
                "12025000",
                "best_model.pth",
            ),
        },
        # Experiment name and output directory
        sub=os.path.join("dplhbv_results", "12025000_eval_train"),
        # Data source configuration
        source_cfgs={"source_name": "camels", "source_path": source_path},
        # Use CPU for this example. To use GPU, set it to [0], [0, 1], etc.
        ctx=[1],
        # Model selection and hyperparameters
        model_name="DplLstmHbv",
        model_hyperparam={
            "n_input_features": 7,
            "n_output_features": 15,
            "n_hidden_states": 128,
            "kernel_size": 30,
            "warmup_length": 365,
            "param_limit_func": "clamp",
            "param_test_way": "final",
        },
        warmup_length=365,
        # Basin IDs for training and evaluation
        gage_id=["12025000"],
        # Training settings
        batch_size=256,
        train_epoch=20,  # Set a small number of epochs for quick testing
        save_epoch=1,
        # Sequence lengths
        hindcast_length=0,
        forecast_length=365,
        # Time settings
        min_time_unit="D",
        min_time_interval="1",
        # Input and output variables
        var_t=[
            "prcp",
            "PET",
            "dayl",
            "srad",
            "tmax",
            "tmin",
            "vp",
        ],
        var_out=["streamflow"],
        var_c=["None"],
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "prcp",
                "PET",
            ],
            "pbm_norm": True,
        },
        # Data components
        dataset="DplDataset",
        # sampler="KuaiSampler",
        scaler="DapengScaler",
        # Date ranges for training, validation, and testing
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["1981-01-01", "2004-12-31"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.005},
        lr_scheduler={"lr_factor": 0.95},
        # Early stopping configuration
        early_stopping=True,
        patience=5,
        # Tensor layout
        which_first_tensor="sequence",
        # metrics
        metrics=["NSE", "KGE", "RMSE", "Corr", "Bias", "FHV", "FLV", "R2"],
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
