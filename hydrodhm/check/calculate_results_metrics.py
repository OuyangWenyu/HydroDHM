"""
Author: Wenyu Ouyang
Date: 2024-09-20 09:41:54
LastEditTime: 2024-09-20 14:18:51
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
    read_sceua_xaj_et_metric,
    read_sceua_xaj_streamflow_metric,
)


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
