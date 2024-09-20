"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:03:43
LastEditTime: 2024-09-20 20:44:21
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
    CAMELS_DPL_PARENT_DIR = definitions_private.CAMELS_DPL_PARENT_DIR
    CHANGDIAN_DPL_PARENT_DIR = definitions_private.CHANGDIAN_DPL_PARENT_DIR
    CAMELS_IDS = definitions_private.CAMELS_IDS
    CHANGDIAN_IDS = definitions_private.CHANGDIAN_IDS
    CHANGDIAN_SCEUA_RESULT_DIRS = definitions_private.CHANGDIAN_SCEUA_RESULT_DIRS
    CHANGDIAN_ID_NAME_DICT = definitions_private.CHANGDIAN_ID_NAME_DICT
except ImportError:
    PROJECT_DIR = os.getcwd()
    RESULT_DIR = (
        "C:\\Users\\wenyu\\OneDrive\\Research\\paper5-dplpartofdissertation\\Results"
    )
    DATASET_DIR = SETTING["local_data_path"]["basins-interim"]

    CAMELS_DPL_PARENT_DIR = [
        os.path.join(RESULT_DIR, "dPL", "result", "data-limited_analysis", tmp_)
        for tmp_ in [
            "camels05y",
            "camels05y_module",
            "camels10y",
            "camels10y_module",
            "camels15y",
            "camels15y_module",
            "camels20y",
            "camels20y_module",
        ]
    ]
    sanxia_dpl_dir1 = os.path.join(RESULT_DIR, "dPL", "result", "streamflow_prediction")
    sanxia_dpl_dir2 = os.path.join(RESULT_DIR, "dPL", "result", "data-limited_analysis")
    CHANGDIAN_DPL_PARENT_DIR = [
        os.path.join(sanxia_dpl_dir1, "lrchange3"),
        os.path.join(sanxia_dpl_dir1, "lrchange3_reverse"),
        os.path.join(sanxia_dpl_dir1, "module"),
        os.path.join(sanxia_dpl_dir1, "module_reverse"),
        os.path.join(sanxia_dpl_dir2, "2to4_1618_1721"),
        os.path.join(sanxia_dpl_dir2, "2to4_1618_1721_module"),
        os.path.join(sanxia_dpl_dir2, "3to4_1417_1721"),
        os.path.join(sanxia_dpl_dir2, "3to4_1417_1721_module"),
        os.path.join(sanxia_dpl_dir2, "3to4_1518_1721"),
        os.path.join(sanxia_dpl_dir2, "3to4_1518_1721_module"),
    ]
    CAMELS_IDS = [
        "camels_02070000",
        "camels_02177000",
        "camels_03346000",
        "camels_03500000",
        "camels_11532500",
        "camels_12025000",
        "camels_12145500",
        "camels_14306500",
    ]
    CHANGDIAN_IDS = [
        "changdian_61561",
        "changdian_61700",
        "changdian_61716",
        "changdian_62618",
        "changdian_91000",
    ]
    CHANGDIAN_SCEUA_RESULT_DIRS = [
        "changdian_61561_4_4",
        "changdian_61700_4_4",
        "changdian_61716_4_4",
        "changdian_62618_4_4",
        "changdian_91000_4_4",
    ]

    CHANGDIAN_ID_NAME_DICT = {
        "changdian_61561": "Duoyingping",
        "changdian_61700": "Sanhuangmiao",
        "changdian_61716": "Dengyingyan",
        "changdian_62618": "Fujiangqiao",
        "changdian_91000": "Ganzi",
    }
