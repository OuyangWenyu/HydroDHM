import os
from torchhydro import SETTING
from torchhydro.configs.config import default_config_file, update_cfg, cmd
from torchhydro.trainers.trainer import train_and_evaluate


def dpl_selfmadehydrodataset_args():
    """_summary_

    Returns
    -------
    _type_
        _description_
    """
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
        ctx=[1],
        model_name="DplLstmXaj",
        # model_name="DplAttrXaj",
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
        },
        loss_func="RMSESum",
        dataset="DplDataset",
        scaler="DapengScaler",
        # lr_scheduler = {
        #     "lr": 0.001,
        #     "lr_factor":0.9,
        #     },
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
            # "changdian_61561",
            "songliao_21401550"
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
        var_out=["streamflow", "evap"],
        n_output=2,
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
        lr_scheduler = {
            epoch: 0.5 if 1 <= epoch <= 9 else 
                    0.2 if 10 <= epoch <= 29 else
                    0.1 if 30 <= epoch <= 69 else 
                    0.05 if 70 <= epoch <= 89 else 
                    0.02
            for epoch in range(1, 101)
        },
        which_first_tensor="sequence",
    )


def run_dpl_exp1():
    cfg = default_config_file()
    args_ = dpl_selfmadehydrodataset_args()
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print("All processes are finished!")


run_dpl_exp1()
