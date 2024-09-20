"""
Author: Wenyu Ouyang
Date: 2024-09-20 11:40:16
LastEditTime: 2024-09-20 20:33:41
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
from definitions import RESULT_DIR
from hydrodhm.utils.results_plot_utils import plot_xaj_params_heatmap

changdian_61700_name = "Sanhuangmiao"
changdian67100_sceua_dir = os.path.join(
    RESULT_DIR, "XAJ", "result", "changdian_61700_4_4"
)
changdian67100_dpl_dir = os.path.join(
    RESULT_DIR, "dPL", "result", "streamflow_prediction", "lrchange3", "changdian_61700"
)
changdian67100_dplnn_dir = os.path.join(
    RESULT_DIR, "dPL", "result", "streamflow_prediction", "module", "changdian_61700"
)
plot_xaj_params_heatmap(
    [changdian67100_sceua_dir, changdian67100_dpl_dir, changdian67100_dplnn_dir],
    ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$"],
    "changdian_" + changdian_61700_name,
    param_test_way=[
        None,
        MODEL_PARAM_TEST_WAY["final_period"],
        MODEL_PARAM_TEST_WAY["final_period"],
    ],
)
