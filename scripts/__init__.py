"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:03:43
LastEditTime: 2024-08-14 15:47:43
LastEditors: Wenyu Ouyang
Description: some global variables used in this project
FilePath: \HydroDHM\scripts\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

from torchhydro import SETTING

PROJECT_DIR = os.getcwd()
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
CASE_DIR = os.path.join(RESULT_DIR, "camels")
DATASET_DIR = SETTING["local_data_path"]["basins-interim"]
