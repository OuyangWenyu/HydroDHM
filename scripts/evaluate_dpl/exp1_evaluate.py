import os
import numpy as np

import torch

from torchhydro import SETTING
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.models.dpl4xaj import DplLstmXaj
from torchhydro.models.kernel_conv import uh_conv, uh_gamma
from torchhydro.models.dpl4xaj_nn4et import DplLstmNnModuleXaj


def dpl_selfmadehydrodataset_args():
    project_name = os.path.join("test_camels", "expdpl61561201")
    train_period = ["2014-10-01", "2018-10-01"]
    valid_period = ["2017-10-01", "2021-10-01"]
    # valid_period = None
    test_period = ["2017-10-01", "2021-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {"time_unit": ["1D"]},
        },
        model_type="MTL",
        ctx=[0],
        model_name="DplLstmXaj",
        # model_name="DplAttrXaj",
        # model_name="DplNnModuleXaj",
        model_hyperparam={
            "n_input_features": 6,
            # "n_input_features": 19,
            "n_output_features": 15,
            "n_hidden_states": 64,
            "kernel_size": 15,
            "warmup_length": 365,
            "param_limit_func": "clamp",
            "param_test_way": "final",
            "source_book": "HF",
            "source_type": "sources",
            # "et_output": 1,
            # "param_var_index": [],
        },
        # loss_func="RMSESum",
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
        dataset="DplDataset",
        scaler="DapengScaler",
        scaler_params={
            "prcp_norm_cols": [
                "streamflow",
            ],
            "gamma_norm_cols": [
                "total_precipitation_hourly",
                "potential_evaporation_hourly",
            ],
            "pbm_norm": True,
        },
        gage_id=[
            # "camels_01013500",
            # "camels_01022500",
            # "camels_01030500",
            # "camels_01031500",
            # "camels_01047000",
            # "camels_01052500",
            # "camels_01054200",
            # "camels_01055000",
            # "camels_01057000",
            # "camels_01170100",
            "changdian_61561"
        ],
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        batch_size=300,
        forecast_history=0,
        forecast_length=365,
        var_t=[
            # although the name is hourly, it might be daily according to your choice
            "total_precipitation_hourly",
            "potential_evaporation_hourly",
            "snow_depth_water_equivalent",
            "snowfall_hourly",
            "dewpoint_temperature_2m",
            "temperature_2m",
        ],
        var_c=[
            # "sgr_dk_sav",
            # "pet_mm_syr",
            # "slp_dg_sav",
            # "for_pc_sse",
            # "pre_mm_syr",
            # "slt_pc_sav",
            # "swc_pc_syr",
            # "soc_th_sav",
            # "cly_pc_sav",
            # "ari_ix_sav",
            # "snd_pc_sav",
            # "ele_mt_sav",
            # "area",
            # "tmp_dc_syr",
            # "crp_pc_sse",
            # "lit_cl_smj",
            # "wet_cl_smj",
            # "snw_pc_syr",
            # "glc_cl_smj",
        ],
        # NOTE: although we set total_evaporation_hourly as output, it is not used in the training process
        var_out=["streamflow", "total_evaporation_hourly"],
        n_output=2,
        # TODO: if chose "mean", metric results' format is different, this should be refactored
        fill_nan=["no", "no"],
        target_as_input=0,
        constant_only=0,
        # train_epoch=100,
        train_epoch=2,
        save_epoch=10,
        model_loader={
            "load_way": "specified",
            # "test_epoch": 100,
            "test_epoch": 2,
        },
        warmup_length=365,
        opt="Adadelta",
        which_first_tensor="sequence",
        train_mode=0,
        weight_path="C:\\Users\\wenyu\\code\\HydroDHM\\results\\test_camels\expdpl61561201\\12_August_202404_08PM_model.pth",
        continue_train=0,
        metrics=["NSE", "RMSE", "Corr", "KGE", "FHV", "FLV"],
    )


if __name__ == "__main__":
    cfg = default_config_file()
    args_ = dpl_selfmadehydrodataset_args()
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print("All processes are finished!")
