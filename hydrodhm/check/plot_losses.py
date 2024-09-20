"""
Author: Wenyu Ouyang
Date: 2024-09-20 21:08:24
LastEditTime: 2024-09-20 21:29:01
LastEditors: Wenyu Ouyang
Description: Plot the time series of losses for a basin
FilePath: \HydroDHM\hydrodhm\check\plot_losses.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))

from definitions import CHANGDIAN_IDS, RESULT_DIR
from hydrodhm.utils.results_plot_utils import plot_losses_ts


def plot_losses_for_changdianbasin(basin_id):
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
    plot_losses_ts(
        [changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        leg_lst=[
            "dPL_train",
            "dPL$_{\mathrm{nn}}$_train",
            "dPL_valid",
            "dPL$_{\mathrm{nn}}$_valid",
        ],
        ylabel=f"Loss_{basin_id}",
    )


def plot_losses_for_reverse_changdianbasin(basin_id):
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
    plot_losses_ts(
        [changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        leg_lst=[
            "dPL_train",
            "dPL$_{\mathrm{nn}}$_train",
            "dPL_valid",
            "dPL$_{\mathrm{nn}}$_valid",
        ],
        ylabel=f"Loss_{basin_id}_reverse",
    )


for _basin_id in CHANGDIAN_IDS:
    plot_losses_for_changdianbasin(_basin_id)
for _basin_id in CHANGDIAN_IDS:
    plot_losses_for_reverse_changdianbasin(_basin_id)
