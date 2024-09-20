"""
Author: Wenyu Ouyang
Date: 2024-09-20 09:41:54
LastEditTime: 2024-09-20 16:06:24
LastEditors: Wenyu Ouyang
Description: 
FilePath: \HydroDHM\hydrodhm\check\calculate_results_metrics.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import RESULT_DIR
from hydrodhm.utils.results_utils import (
    read_dpl_model_metric,
    read_sceua_xaj_et_metric,
    read_sceua_xaj_streamflow_metric,
)


def calculate_sceua_metric():
    sceua_dir = os.path.join(RESULT_DIR, "XAJ", "result")
    sceua_result_dirs = [
        "changdian_61561_4_4",
        "changdian_61700_4_4",
        "changdian_61716_4_4",
        "changdian_62618_4_4",
        "changdian_91000_4_4",
    ]
    sceua_result_dirs_re = [_dir + "_re" for _dir in sceua_result_dirs]
    for _dir in sceua_result_dirs:
        a_sceua_dir = os.path.join(sceua_dir, _dir)
        read_sceua_xaj_et_metric(a_sceua_dir)
    for _dir_ in sceua_result_dirs_re:
        a_sceua_dir_ = os.path.join(sceua_dir, _dir_)
        read_sceua_xaj_et_metric(a_sceua_dir_)


def calculate_sanxia_dpl_metric(sanxia_dpl_parent_dir):
    basin_ids = [
        "changdian_61561",
        "changdian_61700",
        "changdian_61716",
        "changdian_62618",
        "changdian_91000",
    ]
    sanxiabasins_result_dirs = [
        os.path.join(sanxia_dpl_parent_dir, basin_id) for basin_id in basin_ids
    ]
    for j in range(len(sanxiabasins_result_dirs)):
        inds_df_train_q, inds_df_valid_q, inds_df_train_et, inds_df_valid_et = (
            read_dpl_model_metric(sanxiabasins_result_dirs[j], cfg_runagain=True)
        )


def calculate_camels_dpl_metric(camels_dpl_parent_dir):
    camels_ids = [
        "camels_02070000",
        "camels_02177000",
        "camels_03346000",
        "camels_03500000",
        "camels_11532500",
        "camels_12025000",
        "camels_12145500",
        "camels_14306500",
    ]
    for j in range(len(camels_ids)):
        camels_dir_ = os.path.join(camels_dpl_parent_dir, camels_ids[j])
        inds_df_train_q, inds_df_valid_q, inds_df_train_et, inds_df_valid_et = (
            read_dpl_model_metric(camels_dir_, cfg_runagain=True)
        )


lrchange3_reverse = os.path.join(
    RESULT_DIR, "dPL", "result", "streamflow_prediction", "lrchange3_reverse"
)
module_reverse = os.path.join(
    RESULT_DIR, "dPL", "result", "streamflow_prediction", "module_reverse"
)
camels_parent_dir = [
    "camels05y",
    "camels05y_module",
    "camels10y",
    "camels10y_module",
    "camels15y",
    "camels15y_module",
    "camels20y",
    "camels20y_module",
]
for i in range(len(camels_parent_dir)):
    calculate_camels_dpl_metric(
        os.path.join(
            RESULT_DIR, "dPL", "result", "data-limited_analysis", camels_parent_dir[i]
        )
    )
