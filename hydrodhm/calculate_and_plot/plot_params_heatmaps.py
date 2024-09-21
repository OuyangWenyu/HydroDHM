"""
Author: Wenyu Ouyang
Date: 2024-09-20 11:40:16
LastEditTime: 2024-09-20 21:09:54
LastEditors: Wenyu Ouyang
Description: Plot the heatmaps of parameters for XAJ models
FilePath: \HydroDHM\hydrodhm\check\plot_params_heatmaps.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path
from torchhydro.configs.model_config import MODEL_PARAM_TEST_WAY


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import CHANGDIAN_ID_NAME_DICT, RESULT_DIR, CHANGDIAN_IDS
from hydrodhm.utils.results_plot_utils import plot_xaj_params_heatmap


def plot_a_changdian_basin_params_heatmap(basin_id):
    changdian_basin_name = CHANGDIAN_ID_NAME_DICT[basin_id]
    changdian_basin_sceua_dir = os.path.join(
        RESULT_DIR, "XAJ", "result", f"{basin_id}_4_4"
    )
    changdian_basin_dpl_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "lrchange3",
        basin_id,
    )
    changdian_basin_dplnn_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "module",
        basin_id,
    )
    plot_xaj_params_heatmap(
        [changdian_basin_sceua_dir, changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$"],
        basin_id + changdian_basin_name,
        param_test_way=[
            None,
            MODEL_PARAM_TEST_WAY["final_period"],
            MODEL_PARAM_TEST_WAY["final_period"],
        ],
    )


def plot_a_changdian_basin_reverse_params_heatmap(basin_id):
    changdian_basin_name = CHANGDIAN_ID_NAME_DICT[basin_id]
    changdian_basin_sceua_dir = os.path.join(
        RESULT_DIR, "XAJ", "result", f"{basin_id}_4_4_re"
    )
    changdian_basin_dpl_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "lrchange3_reverse",
        basin_id,
    )
    changdian_basin_dplnn_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "module_reverse",
        basin_id,
    )
    plot_xaj_params_heatmap(
        [changdian_basin_sceua_dir, changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$"],
        basin_id + changdian_basin_name,
        param_test_way=[
            None,
            MODEL_PARAM_TEST_WAY["final_period"],
            MODEL_PARAM_TEST_WAY["final_period"],
        ],
    )


for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_reverse_params_heatmap(_basin_id)
for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_params_heatmap(_basin_id)
