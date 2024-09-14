"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:03:43
LastEditTime: 2024-09-14 16:05:21
LastEditors: Wenyu Ouyang
Description: some global variables used in this project
FilePath: \HydroDHM\definitions.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

# NOTE: create a file in root directory -- definitions_private.py, then write the following code:
# NOTE: set the directories for your own project
# PROJECT_DIR = os.getcwd()
# RESULT_DIR = "C:\\Users\\wenyu\\OneDrive\\Research\\paper5-dplpartofdissertation\\Results"
# DATASET_DIR = SETTING["local_data_path"]["basins-interim"]
import os

from torchhydro import SETTING

try:
    import definitions_private

    PROJECT_DIR = definitions_private.PROJECT_DIR
    RESULT_DIR = definitions_private.RESULT_DIR
    DATASET_DIR = definitions_private.DATASET_DIR
except ImportError:
    PROJECT_DIR = os.getcwd()
    RESULT_DIR = (
        "C:\\Users\\wenyu\\OneDrive\\Research\\paper5-dplpartofdissertation\\Results"
    )
    DATASET_DIR = SETTING["local_data_path"]["basins-interim"]
