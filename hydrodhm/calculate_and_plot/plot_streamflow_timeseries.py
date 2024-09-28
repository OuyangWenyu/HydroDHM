"""
Author: Wenyu Ouyang
Date: 2024-09-20 21:35:03
LastEditTime: 2024-09-28 15:58:03
LastEditors: Wenyu Ouyang
Description: Plot the streamflow time series
FilePath: \HydroDHM\hydrodhm\calculate_and_plot\plot_streamflow_timeseries.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import (
    CHANGDIAN_ID_NAME_DICT,
    CHANGDIAN_IDS,
    RESULT_DIR,
    SANXIA_DPL_DIR1,
    SANXIA_SCEUA_DIR1,
)
from hydrodhm.utils.results_plot_utils import plot_xaj_rainfall_runoff


def plot_a_changdian_basin_rainfall_runoff(basin_id, reverse=False, legends=None):
    if legends is None:
        legends = ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$", "OBS"]
    changdian_basin_name = CHANGDIAN_ID_NAME_DICT[basin_id]
    show_lst = []
    if "eXAJ" in legends:
        changdian_basin_sceua_dir = os.path.join(
            SANXIA_SCEUA_DIR1,
            f"{basin_id}_4_4_re" if reverse else f"{basin_id}_4_4",
        )
        show_lst.append(changdian_basin_sceua_dir)
    if "dXAJ" in legends:
        changdian_basin_dpl_dir = os.path.join(
            SANXIA_DPL_DIR1,
            "lrchange3_reverse" if reverse else "lrchange3",
            basin_id,
        )
        show_lst.append(changdian_basin_dpl_dir)
    if "dXAJ$_{\mathrm{nn}}$" in legends:
        changdian_basin_dplnn_dir = os.path.join(
            SANXIA_DPL_DIR1,
            "module_reverse" if reverse else "module",
            basin_id,
        )
        show_lst.append(changdian_basin_dplnn_dir)
    if basin_id == "changdian_91000":
        prcp_interval = 10
    elif basin_id == "changdian_62618":
        prcp_interval = 30
    else:
        prcp_interval = 20
    plot_xaj_rainfall_runoff(
        show_lst,
        basin_id,
        changdian_basin_name,
        leg_names=legends,
        prcp_interval=prcp_interval,
    )


for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_rainfall_runoff(
        _basin_id, reverse=False, legends=["eXAJ", "dXAJ", "OBS"]
    )

for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_rainfall_runoff(
        _basin_id, reverse=True, legends=["eXAJ", "dXAJ", "OBS"]
    )
