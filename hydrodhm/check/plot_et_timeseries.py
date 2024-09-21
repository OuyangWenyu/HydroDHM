"""
Author: Wenyu Ouyang
Date: 2024-09-20 21:53:39
LastEditTime: 2024-09-20 21:54:49
LastEditors: Wenyu Ouyang
Description: Plot the ET time series
FilePath: \HydroDHM\hydrodhm\check\plot_et_timeseries.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import CHANGDIAN_ID_NAME_DICT, CHANGDIAN_IDS, RESULT_DIR
from hydrodhm.utils.results_plot_utils import plot_xaj_et_time_series


def plot_a_changdian_basin_etts(
    basin_id, reverse=False, legends=["dXAJ", "dXAJ$_{\mathrm{nn}}$", "OBS"]
):
    changdian_basin_name = CHANGDIAN_ID_NAME_DICT[basin_id]
    show_lst = []
    if "eXAJ" in legends:
        changdian_basin_sceua_dir = os.path.join(
            RESULT_DIR,
            "XAJ",
            "result",
            f"{basin_id}_4_4" if not reverse else f"{basin_id}_4_4_re",
        )
        show_lst.append(changdian_basin_sceua_dir)
    changdian_basin_dpl_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "lrchange3" if not reverse else "lrchange3_reverse",
        basin_id,
    )
    show_lst.append(changdian_basin_dpl_dir)
    changdian_basin_dplnn_dir = os.path.join(
        RESULT_DIR,
        "dPL",
        "result",
        "streamflow_prediction",
        "module" if not reverse else "module_reverse",
        basin_id,
    )
    show_lst.append(changdian_basin_dplnn_dir)
    plot_xaj_et_time_series(
        show_lst,
        basin_id,
        changdian_basin_name,
        leg_names=legends,
    )


for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_etts(_basin_id)

for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_etts(_basin_id, reverse=True)
