"""
Author: Wenyu Ouyang
Date: 2024-09-20 11:40:16
LastEditTime: 2024-09-25 11:10:44
LastEditors: Wenyu Ouyang
Description: Plot the heatmaps of parameters for XAJ models
FilePath: \HydroDHM\hydrodhm\calculate_and_plot\plot_params_heatmaps.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path
from torchhydro.configs.model_config import MODEL_PARAM_TEST_WAY


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import (
    CHANGDIAN_ID_NAME_DICT,
    RESULT_DIR,
    CHANGDIAN_IDS,
    SANXIA_DPL_DIR1,
)
from hydrodhm.utils.results_plot_utils import plot_xaj_params_heatmap


def plot_a_changdian_basin_params_heatmap(basin_id, is_reverse=False):
    changdian_basin_name = CHANGDIAN_ID_NAME_DICT[basin_id]
    changdian_basin_sceua_dir = os.path.join(
        RESULT_DIR, "XAJ", "result", f"{basin_id}_4_4"
    )
    changdian_basin_dpl_dir = os.path.join(
        SANXIA_DPL_DIR1,
        "lrchange3_reverse" if is_reverse else "lrchange3",
        basin_id,
    )
    changdian_basin_dplnn_dir = os.path.join(
        SANXIA_DPL_DIR1,
        "module_reverse" if is_reverse else "module",
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
    plot_a_changdian_basin_params_heatmap(_basin_id)
for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_params_heatmap(_basin_id, is_reverse=True)
