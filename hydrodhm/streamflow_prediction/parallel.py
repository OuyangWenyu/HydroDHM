import os
from torchhydro import SETTING
from torchhydro.configs.config import default_config_file, update_cfg, cmd
from torchhydro.trainers.trainer import train_and_evaluate
from concurrent.futures import ProcessPoolExecutor


def dpl_selfmadehydrodataset_args(gage_id):
    project_name = os.path.join("streamflow_prediction_camels20y", gage_id)
    train_period = ["1999-10-01", "2019-10-01"]
    valid_period = ["2018-10-01", "2023-10-01"]
    # valid_period = None
    test_period = ["2018-10-01", "2023-10-01"]
    return cmd(
        sub=project_name,
        source_cfgs={
            "source_name": "selfmadehydrodataset",
            "source_path": SETTING["local_data_path"]["datasets-interim"],
            "other_settings": {"time_unit": ["1D"]},
        },
        model_type="MTL",
        ctx=[2],
        model_name="DplLstmXaj",
        # model_name="DplAttrXaj",
        # model_name="DplNnModuleXaj", # 替换模块
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
            # "et_output": 1, # 添加参数
            # "param_var_index": [], # 添加参数
        },
        # loss_func="RMSESum",
        loss_func="MultiOutLoss", # 替换损失函数
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [2],
            "item_weight": [1, 0],
            "limit_part": [1],
        },  # 添加参数
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
        gage_id=[gage_id],
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
        var_out=["streamflow", "total_evaporation_hourly"], # 添加参数
        n_output=2, # 添加参数
        fill_nan=["no", "no"], # 添加参数
        target_as_input=0,
        constant_only=0,
        # train_epoch=100,
        train_epoch=100,
        save_epoch=1,
        model_loader={
            "load_way": "specified",
            # "test_epoch": 100,
            "test_epoch": 100,
        },
        warmup_length=365,
        opt="Adadelta",
        # opt_param={
        #     "lr":0.1,
        # },# 固定学习率
        # lr_scheduler={
        #     # "lr":0.1,
        #     0: 0.1,
        #     1: 0.1, 
        #     2: 0.05, 
        #     3: 0.02,
        #     4: 0.02,
        # },# 指定每个epoch的学习率
        lr_scheduler = {
            epoch: 0.5 if 1 <= epoch <= 4 else 
                    0.2 if 5 <= epoch <= 19 else
                    0.1 if 20 <= epoch <= 49 else 
                    0.05 if 50 <= epoch <= 79 else 
                    0.02 if 80 <= epoch <= 94 else
                    0.01
            for epoch in range(1, 101)
        },
        # lr_scheduler = {
        #     epoch: 0.5 if 1 <= epoch <= 9 else 
        #             0.2 if 10 <= epoch <= 29 else
        #             0.1 if 30 <= epoch <= 69 else 
        #             0.05 if 70 <= epoch <= 89 else 
        #             0.02
        #     for epoch in range(1, 101)
        # },
        # lr_scheduler = {
        #     epoch: 0.1 if 1 <= epoch <= 9 else 
        #             0.01 if 10 <= epoch <= 49 else 
        #             0.001 if 50 <= epoch <= 79 else 
        #             0.0001
        #     for epoch in range(1, 101)
        # },  # 指定每个阶段epoch的学习率
        which_first_tensor="sequence",
        metrics=["NSE", "RMSE", "Corr", "KGE", "FHV", "FLV"],
    )


def run_dpl_exp(gage_id):
    cfg = default_config_file()
    args_ = dpl_selfmadehydrodataset_args(gage_id)
    update_cfg(cfg, args_)
    train_and_evaluate(cfg)
    print(f"Process for {gage_id} is finished!")


def run_all_gages(gage_ids):
    with ProcessPoolExecutor(max_workers=len(gage_ids)) as executor:
        executor.map(run_dpl_exp, gage_ids)


if __name__ == "__main__":
    gage_ids = [
        # "changdian_61561",
        # "changdian_61700",
        # "changdian_61716",
        # "changdian_62618",
        # "changdian_91000",

        # "changdian_60650",
        # "changdian_60668",
        # "changdian_61239",
        # "changdian_61277",
        # "changdian_62018",
        # "changdian_62315",
        # "changdian_63002",
        # "changdian_63007",
        # "changdian_63458",
        # "changdian_63486",
        # "changdian_63490",
        # "changdian_90813",
        # "changdian_91700",
        # "changdian_92114",
        # "changdian_92116",
        # "changdian_92118",
        # "changdian_92119",
        # "changdian_92146",
        # "changdian_92353",
        # "changdian_92354",
        # "changdian_94470",
        # "changdian_94560",
        # "changdian_94850",
        # "changdian_95350",


        # "camels_01539000",
        # "camels_02231000",
        # "camels_03161000",
        # "camels_03300400",
        # "camels_07261000",

        "camels_11532500",
        "camels_12025000",
        # "camels_12035000",
        "camels_12145500",
        # "camels_14301000",
        "camels_14306500",
        # "camels_14325000",


        # "camels_01440000",
        # "camels_01440400",
        # "camels_01532000",
        # "camels_01552000",
        "camels_02070000",
        # "camels_02137727",
        # "camels_02140991",
        "camels_02177000",
        # "camels_02212600",
        # "camels_02246000",
        # "camels_02427250",
        "camels_03500000",


        "camels_03346000",
        # "camels_05501000",
        # "camels_05514500",
        # "camels_07057500",
        # "camels_07066000",
        # "camels_07145700",
        # "camels_07263295",
        # "camels_07359610",
        

        # "anhui_62909400",
        # "songliao_10911000",
        # "songliao_10912404",
        # "songliao_11002210",
        # "songliao_11400900",
        # "songliao_11606000",
        # "songliao_21110150",
        # "songliao_21110400",
        # "songliao_21113800",
        # "songliao_21200100",
        # "songliao_21300500",
        # "songliao_21401050",
        # "songliao_21401300",
        
        # Add your gage IDs here
        # Add more gage IDs as needed
    ]
    run_all_gages(gage_ids)
