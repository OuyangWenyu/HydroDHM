from pathlib import Path
from hydroutils.hydro_file import unserialize_json
import numpy as np
import pandas as pd
from scripts import CASE_DIR


import os

from torchhydro.configs.model_config import MODEL_PARAM_DICT


MODEL_PARAM_TEST_WAY = {
    # 0. "train_final" -- use the final training period's parameter for each test period
    "final_train_period": "train_final",
    # 1. "final" -- use the final testing period's parameter for each test period
    "final_period": "final",
    # 2. "mean_time" -- Mean values of all training periods' parameters are used
    "mean_all_period": "mean_time",
    # 3. "mean_basin" -- Mean values of all basins' final training periods' parameters is used
    "mean_all_basin": "mean_basin",
    # 4. "var" -- use time series parameters and constant parameters in testing period
    "time_varying": "var",
    "time_scroll": "dynamic",
}


def get_latest_dirs_for_sceua_xaj(exp, comment, cv_fold=2):
    """A same comment means the experiments with same configuration, but different runs.

    We choose the latest experiment.

    Parameters
    ----------
    comment : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    all_dirs = os.listdir(os.path.join(CASE_DIR, exp))
    dirs = []
    for i in range(cv_fold):
        chosen_dirs = [
            dir for dir in all_dirs if dir.endswith(f"fold{str(i)}_" + comment)
        ]
        chosen_dirs.sort()
        fold_i = chosen_dirs[-1]
        dirs.append(fold_i)
    return dirs


def get_pbm_params_from_hydromodelxaj(
    exp, kfold, the_fold, sceua_plan, example, cfg_dir_flow
):
    comment_dir = get_latest_dirs_for_sceua_xaj(
        example + os.sep + exp, sceua_plan, kfold
    )
    normlize_param_file = os.path.join(
        cfg_dir_flow, comment_dir[the_fold], "basins_params.csv"
    )
    parameters = pd.read_csv(normlize_param_file, index_col=0).values
    params_file = os.path.join(
        cfg_dir_flow, comment_dir[the_fold], "basins_renormalization_params.csv"
    )
    params = pd.read_csv(params_file, index_col=0).values
    return parameters, params


def get_latest_pbm_param_file(param_dir):
    """Get the latest parameter file of physics-based models in the current directory.

    Parameters
    ----------
    param_dir : str
        The directory of parameter files.

    Returns
    -------
    str
        The latest parameter file.
    """
    param_file_lst = [
        os.path.join(param_dir, f)
        for f in os.listdir(param_dir)
        if f.startswith("pb_params") and f.endswith(".csv")
    ]
    param_files = [Path(f) for f in param_file_lst]
    param_file_names_lst = [param_file.stem.split("_") for param_file in param_files]
    ctimes = [
        int(param_file_names[param_file_names.index("params") + 1])
        for param_file_names in param_file_names_lst
    ]
    return param_files[ctimes.index(max(ctimes))]


def get_pbm_params_from_dpl(cfg_dir_flow):
    params_file = get_latest_pbm_param_file(cfg_dir_flow)
    parameters = pd.read_csv(params_file).iloc[:, 1:].values
    params = np.zeros(parameters.shape)
    param_range = MODEL_PARAM_DICT["xaj"]["param_range"]
    k_scale = param_range["K"]
    b_scale = param_range["B"]
    im_sacle = param_range["IM"]
    um_scale = param_range["UM"]
    lm_scale = param_range["LM"]
    dm_scale = param_range["DM"]
    c_scale = param_range["C"]
    sm_scale = param_range["SM"]
    ex_scale = param_range["EX"]
    ki_scale = param_range["KI"]
    kg_scale = param_range["KG"]
    a_scale = param_range["A"]
    theta_scale = param_range["THETA"]
    ci_scale = param_range["CI"]
    cg_scale = param_range["CG"]
    params[:, 0] = k_scale[0] + parameters[:, 0] * (k_scale[1] - k_scale[0])
    params[:, 1] = b_scale[0] + parameters[:, 1] * (b_scale[1] - b_scale[0])
    params[:, 2] = im_sacle[0] + parameters[:, 2] * (im_sacle[1] - im_sacle[0])
    params[:, 3] = um_scale[0] + parameters[:, 3] * (um_scale[1] - um_scale[0])
    params[:, 4] = lm_scale[0] + parameters[:, 4] * (lm_scale[1] - lm_scale[0])
    params[:, 5] = dm_scale[0] + parameters[:, 5] * (dm_scale[1] - dm_scale[0])
    params[:, 6] = c_scale[0] + parameters[:, 6] * (c_scale[1] - c_scale[0])
    params[:, 7] = sm_scale[0] + parameters[:, 7] * (sm_scale[1] - sm_scale[0])
    params[:, 8] = ex_scale[0] + parameters[:, 8] * (ex_scale[1] - ex_scale[0])
    ki = ki_scale[0] + parameters[:, 9] * (ki_scale[1] - ki_scale[0])
    kg = kg_scale[0] + parameters[:, 10] * (kg_scale[1] - kg_scale[0])
    # ki+kg should be smaller than 1; if not, we scale them
    params[:, 9] = np.where(ki + kg < 1.0, ki, 1 / (ki + kg) * ki)
    params[:, 10] = np.where(ki + kg < 1.0, kg, 1 / (ki + kg) * kg)
    params[:, 11] = a_scale[0] + parameters[:, 11] * (a_scale[1] - a_scale[0])
    params[:, 12] = theta_scale[0] + parameters[:, 12] * (
        theta_scale[1] - theta_scale[0]
    )
    params[:, 13] = ci_scale[0] + parameters[:, 13] * (ci_scale[1] - ci_scale[0])
    params[:, 14] = cg_scale[0] + parameters[:, 14] * (cg_scale[1] - cg_scale[0])
    params = params.T
    parameters = parameters.T
    return parameters, params


def get_common_path_name(origin_pc_test_path, replace_item):
    if type(origin_pc_test_path) is list:
        return [
            get_common_path_name(a_path, replace_item) for a_path in origin_pc_test_path
        ]
    if origin_pc_test_path.startswith("/"):
        # linux
        origin_pc_test_path_lst = origin_pc_test_path.split("/")
    else:
        # windows
        origin_pc_test_path_lst = origin_pc_test_path.split("\\")
    # NOTE: this is a hard code
    if replace_item == "data_path":
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "data"]
        the_root_dir = definitions.DATASET_DIR
    else:
        pos_lst = [i for i, e in enumerate(origin_pc_test_path_lst) if e == "HydroSPB"]
        the_root_dir = definitions.ROOT_DIR
    if not pos_lst:
        raise ValueError("Can not find the common path name")
    elif len(pos_lst) == 1:
        where_start_same_in_origin_pc = pos_lst[0] + 1
    else:
        for i in pos_lst:
            if os.path.exists(
                os.path.join(
                    the_root_dir, os.sep.join(origin_pc_test_path_lst[i + 1 :])
                )
            ):
                where_start_same_in_origin_pc = i + 1
                break

    return os.sep.join(origin_pc_test_path_lst[where_start_same_in_origin_pc:])


def get_the_new_path_with_diff_part(cfg_json, replace_item):
    the_item = cfg_json["data_params"][replace_item]
    if the_item is None:
        return None
    common_path_name = get_common_path_name(the_item, replace_item)
    dff_path_name = (
        definitions.DATASET_DIR if replace_item == "data_path" else definitions.ROOT_DIR
    )
    if type(common_path_name) is list:
        return [os.path.join(dff_path_name, a_path) for a_path in common_path_name]
    return os.path.join(dff_path_name, common_path_name)


def update_cfg_as_move_to_another_pc(cfg_json):
    """update cfg as move to another pc

    Returns
    -------
    _type_
        _description_
    """
    cfg_json["data_params"]["test_path"] = get_the_new_path_with_diff_part(
        cfg_json, "test_path"
    )
    cfg_json["data_params"]["data_path"] = get_the_new_path_with_diff_part(
        cfg_json, "data_path"
    )
    cfg_json["data_params"]["cache_path"] = get_the_new_path_with_diff_part(
        cfg_json, "cache_path"
    )
    cfg_json["data_params"]["validation_path"] = get_the_new_path_with_diff_part(
        cfg_json, "validation_path"
    )


def get_json_file(cfg_dir):
    json_files_lst = []
    json_files_ctime = []
    for file in os.listdir(cfg_dir):
        if (
            fnmatch.fnmatch(file, "*.json")
            and "_stat" not in file  # statistics json file
            and "_dict" not in file  # data cache json file
        ):
            json_files_lst.append(os.path.join(cfg_dir, file))
            json_files_ctime.append(os.path.getctime(os.path.join(cfg_dir, file)))
    sort_idx = np.argsort(json_files_ctime)
    cfg_file = json_files_lst[sort_idx[-1]]
    cfg_json = unserialize_json(cfg_file)
    if cfg_json["data_params"]["test_path"] != cfg_dir:
        # sometimes we will use files copied from other device, so the dir is not correct for this device
        update_cfg_as_move_to_another_pc(cfg_json=cfg_json)

    return cfg_json
