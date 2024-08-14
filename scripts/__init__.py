"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:03:43
LastEditTime: 2024-08-14 09:05:25
LastEditors: Wenyu Ouyang
Description: some global variables used in this project
FilePath: \HydroDHM\scripts\__init__.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os

from torchhydro import SETTING

PROJECT_DIR = os.getcwd()
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
DATASET_DIR = SETTING["local_data_path"]["basins-interim"]
