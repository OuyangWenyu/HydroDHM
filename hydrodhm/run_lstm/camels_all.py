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

from hydrodataset.camels import Camels

camels = Camels(SETTING["local_data_path"]["datasets-origin"])
gage_id = camels.read_site_info()["gauge_id"].values.tolist()
assert all(x < y for x, y in zip(gage_id, gage_id[1:])), "gage_id should be sorted"

var_c = camels.get_available_static_features()


def main():
    """
    This function defines the parameters for the experiment, updates the default configuration,
    and then runs the training and evaluation pipeline.
    """
    # 1. Define parameters for the experiment
    # You can refer to the `cmd` function in `torchhydro.configs.config` for more details.
    source_path = SETTING["local_data_path"]["datasets-origin"]
    args = cmd(
        # Experiment name and output directory
        sub=os.path.join("lstm_results", "camels_all"),
        # Data source configuration
        source_cfgs={"source_name": "camels", "source_path": source_path},
        # Use CPU for this example. To use GPU, set it to [0], [0, 1], etc.
        ctx=[1],
        # Model selection and hyperparameters
        model_name="SimpleLSTM",
        model_hyperparam={
            "input_size": 30,
            "output_size": 1,
            "hidden_size": 128,
        },
        # Basin IDs for training and evaluation
        gage_id=gage_id,
        # Training settings
        batch_size=256,
        train_epoch=50,  # Set a small number of epochs for quick testing
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
            "dayl",
            "srad",
            "tmax",
            "tmin",
            "vp",
            "PET",
        ],
        var_out=["streamflow"],
        var_c=[
            "area_gages2",
            "p_mean",
            "huc_02",
            "gauge_lat",
            "gauge_lon",
            "elev_mean",
            "slope_mean",
            "geol_1st_class",
            "geol_2nd_class",
            "geol_porostiy",
            "geol_permeability",
            "frac_forest",
            "lai_max",
            "lai_diff",
            "dom_land_cover_frac",
            "dom_land_cover",
            "root_depth_50",
            "root_depth_99",
            "soil_depth_statsgo",
            "soil_porosity",
            "soil_conductivity",
            "max_water_content",
            "pet_mean",
        ],
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "prcp",
                "PET",
            ],
            "pbm_norm": False,
        },
        # Data components
        dataset="StreamflowDataset",
        # sampler="KuaiSampler",
        scaler="StandardScaler",
        # Model loading configuration for evaluation
        model_loader={"load_way": "best"},
        # Date ranges for training, validation, and testing
        train_period=["1980-01-01", "2004-12-31"],
        valid_period=["2005-01-01", "2009-12-31"],
        test_period=["2010-01-01", "2014-12-31"],
        # Loss function and optimizer
        loss_func="RMSESum",
        opt="Adam",
        opt_param={"lr": 0.0005},
        lr_scheduler={"lr_factor": 0.95},
        early_stopping=True,
        patience=5,
        # Tensor layout
        which_first_tensor="sequence",
    )

    # 2. Load default config and update it with your parameters
    config_data = default_config_file()
    update_cfg(config_data, args)

    # 3. Run the training and evaluation pipeline
    train_and_evaluate(config_data)


if __name__ == "__main__":
    main()
