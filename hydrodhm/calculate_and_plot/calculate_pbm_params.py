"""
Author: Wenyu Ouyang
Date: 2024-09-20 19:59:39
LastEditTime: 2024-09-24 19:57:47
LastEditors: Wenyu Ouyang
Description: Calculate the parameters of dPL PBM models
FilePath: \HydroDHM\hydrodhm\calculate_and_plot\calculate_pbm_params.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import (
    CHANGDIAN_DPL_PARENT_DIR,
    CAMELS_DPL_PARENT_DIR,
    CHANGDIAN_IDS,
    CAMELS_IDS,
)
from hydrodhm.utils.results_utils import get_pbm_params_from_dpl

for i in range(len(CHANGDIAN_DPL_PARENT_DIR)):
    for j in range(len(CHANGDIAN_IDS)):
        get_pbm_params_from_dpl(
            os.path.join(CHANGDIAN_DPL_PARENT_DIR[i], CHANGDIAN_IDS[j])
        )

# for i in range(len(CAMELS_DPL_PARENT_DIR)):
#     for j in range(len(CAMELS_IDS)):
#         get_pbm_params_from_dpl(os.path.join(CAMELS_DPL_PARENT_DIR[i], CAMELS_IDS[j]))
