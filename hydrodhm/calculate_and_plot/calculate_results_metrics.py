"""
Author: Wenyu Ouyang
Date: 2024-09-20 09:41:54
LastEditTime: 2024-09-20 20:15:35
LastEditors: Wenyu Ouyang
Description: Calculate the results and metrics of SCEUA and dPL/dPL-NN XAJ models
FilePath: \HydroDHM\hydrodhm\check\calculate_results_metrics.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import (
    RESULT_DIR,
    CAMELS_DPL_PARENT_DIR,
    CHANGDIAN_DPL_PARENT_DIR,
    CAMELS_IDS,
    CHANGDIAN_IDS,
    CHANGDIAN_SCEUA_RESULT_DIRS,
)
from hydrodhm.utils.results_utils import (
    read_dpl_model_metric,
    read_sceua_xaj_et_metric,
    read_sceua_xaj_streamflow_metric,
)


def calculate_sceua_metric(sceua_dir):
    sceua_result_dirs_re = [_dir + "_re" for _dir in CHANGDIAN_SCEUA_RESULT_DIRS]
    for _dir in CHANGDIAN_SCEUA_RESULT_DIRS:
        a_sceua_dir = os.path.join(sceua_dir, _dir)
        inds_df_train, inds_df_valid = read_sceua_xaj_et_metric(
            a_sceua_dir, is_save=True
        )
    for _dir_ in sceua_result_dirs_re:
        a_sceua_dir_ = os.path.join(sceua_dir, _dir_)
        inds_df_train, inds_df_valid = read_sceua_xaj_et_metric(
            a_sceua_dir_, is_save=True
        )


def calculate_sanxia_dpl_metric(sanxia_dpl_parent_dir):
    sanxiabasins_result_dirs = [
        os.path.join(sanxia_dpl_parent_dir, basin_id) for basin_id in CHANGDIAN_IDS
    ]
    for j in range(len(sanxiabasins_result_dirs)):
        inds_df_train_q, inds_df_valid_q, inds_df_train_et, inds_df_valid_et = (
            read_dpl_model_metric(sanxiabasins_result_dirs[j], cfg_runagain=True)
        )


def calculate_camels_dpl_metric(camels_dpl_parent_dir):
    for j in range(len(CAMELS_IDS)):
        camels_dir_ = os.path.join(camels_dpl_parent_dir, CAMELS_IDS[j])
        inds_df_train_q, inds_df_valid_q, inds_df_train_et, inds_df_valid_et = (
            read_dpl_model_metric(camels_dir_, cfg_runagain=True)
        )


sceua_dir1 = os.path.join(RESULT_DIR, "XAJ", "result")
calculate_sceua_metric(sceua_dir1)
for i in range(len(CHANGDIAN_DPL_PARENT_DIR)):
    calculate_sanxia_dpl_metric(CHANGDIAN_DPL_PARENT_DIR[i])
for i in range(len(CAMELS_DPL_PARENT_DIR)):
    calculate_camels_dpl_metric(CAMELS_DPL_PARENT_DIR[i])
