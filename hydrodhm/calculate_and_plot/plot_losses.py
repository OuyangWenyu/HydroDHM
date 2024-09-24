"""
Author: Wenyu Ouyang
Date: 2024-09-20 21:08:24
LastEditTime: 2024-09-24 20:17:00
LastEditors: Wenyu Ouyang
Description: Plot the time series of losses for a basin
FilePath: \HydroDHM\hydrodhm\calculate_and_plot\plot_losses.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))

from definitions import CHANGDIAN_IDS, RESULT_DIR, SANXIA_DPL_DIR1
from hydrodhm.utils.results_plot_utils import plot_losses_ts


def plot_losses_for_changdianbasin(basin_id, leg_lst=None, is_reverse=False):
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
    plot_losses_ts(
        [changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        leg_lst=leg_lst,
    )


leg_lst = [
    "dPL_calib.",
    "dPL$_{\mathrm{nn}}$_calib.",
    "dPL_val.",
    "dPL$_{\mathrm{nn}}$_val.",
]
for _basin_id in CHANGDIAN_IDS:
    plot_losses_for_changdianbasin(_basin_id, leg_lst=leg_lst)
for _basin_id in CHANGDIAN_IDS:
    plot_losses_for_changdianbasin(_basin_id, is_reverse=True)
