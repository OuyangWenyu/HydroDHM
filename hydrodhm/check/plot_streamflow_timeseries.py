import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import CHANGDIAN_ID_NAME_DICT, CHANGDIAN_IDS, RESULT_DIR
from hydrodhm.utils.results_plot_utils import plot_xaj_rainfall_runoff


def plot_a_changdian_basin_rainfall_runoff(basin_id):
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
    plot_xaj_rainfall_runoff(
        [changdian_basin_sceua_dir, changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        basin_id,
        changdian_basin_name,
    )


def plot_a_changdian_basin_reverse_rainfall_runoff(basin_id):
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
    plot_xaj_rainfall_runoff(
        [changdian_basin_sceua_dir, changdian_basin_dpl_dir, changdian_basin_dplnn_dir],
        basin_id,
        changdian_basin_name,
    )


for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_rainfall_runoff(_basin_id)

for _basin_id in CHANGDIAN_IDS:
    plot_a_changdian_basin_reverse_rainfall_runoff(_basin_id)
