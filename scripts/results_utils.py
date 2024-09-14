import os
import sys
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hydroutils import hydro_time
from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import unserialize_json
from hydromodel.datasets.data_preprocess import cross_val_split_tsdata
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.model_config import MODEL_PARAM_DICT

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))
from definitions import RESULT_DIR, DATASET_DIR
from scripts.evaluate_xaj import _evaluate_1fold

# ET_MODIS_NAME = "ET_modis16a2006"
ET_MODIS_NAME = "ET_modis16a2gf061"


def read_sceua_xaj_streamflow(result_dir):
    """Read one directory of SCEUA-XAJ results from hydromodel project

    Parameters
    ----------
    result_dir : str
        the directory of SCEUA-XAJ results

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        qsim_train, qsim_test, qobs_train, qobs_test
    """
    train_result_file = os.path.join(
        result_dir, "sceua_xaj", "train", "xaj_mz_evaluation_results.nc"
    )
    test_result_file = os.path.join(
        result_dir,
        "sceua_xaj",
        "test",
        "xaj_mz_evaluation_results.nc",
    )
    data_train = xr.open_dataset(train_result_file)
    data_test = xr.open_dataset(test_result_file)
    qsim_train = data_train["qsim"]
    qsim_test = data_test["qsim"]
    qobs_train = data_train["qobs"]
    qobs_test = data_test["qobs"]
    return (
        qsim_train,
        qsim_test,
        qobs_train,
        qobs_test,
    )


def read_sceua_xaj_streamflow_metric(result_dir):
    """read SCEUA-XAJ metrics from one hydromodel project directory

    Parameters
    ----------
    result_dir : _type_
        the directory of SCEUA-XAJ results
    """
    train_metrics_file = os.path.join(
        result_dir,
        "sceua_xaj",
        "train",
        "basins_metrics.csv",
    )
    test_metrics_file = os.path.join(
        result_dir,
        "sceua_xaj",
        "test",
        "basins_metrics.csv",
    )
    basin_id_train_metric = pd.read_csv(train_metrics_file, index_col=0)
    basin_id_test_metric = pd.read_csv(test_metrics_file, index_col=0)
    print("The metrics of training results of basin " + result_dir + " are:")
    print(basin_id_train_metric)
    print("The metrics of testing results of basin " + result_dir + " are:")
    print(basin_id_test_metric)
    return basin_id_train_metric, basin_id_test_metric


def read_sceua_xaj_et(result_dir, et_type=ET_MODIS_NAME):
    config_yml_file = os.path.join(result_dir, "config.yaml")
    with open(config_yml_file, "r") as file:
        config_data = yaml.safe_load(file)
    basin_ids = config_data["basin_id"]

    train_result_file = os.path.join(
        result_dir,
        "sceua_xaj",
        "train",
        "xaj_mz_evaluation_results.nc",
    )
    test_result_file = os.path.join(
        result_dir,
        "sceua_xaj",
        "test",
        "xaj_mz_evaluation_results.nc",
    )
    et_sim_train_ = xr.open_dataset(train_result_file)
    et_sim_test_ = xr.open_dataset(test_result_file)
    if "etsim" not in et_sim_train_ or "etsim" not in et_sim_test_:
        train_and_test_data = cross_val_split_tsdata(
            config_data["data_type"],
            config_data["data_dir"],
            config_data["cv_fold"],
            config_data["calibrate_period"],
            config_data["test_period"],
            config_data["period"],
            config_data["warmup"],
            config_data["basin_id"],
        )
        _evaluate_1fold(train_and_test_data, result_dir)
    t_range_train = [et_sim_train_.time.values[0], et_sim_train_.time.values[-1]]
    t_range_test = [et_sim_test_.time.values[0], et_sim_test_.time.values[-1]]

    selfmadehydrodataset = SelfMadeHydroDataset(DATASET_DIR, time_unit=["8D"])
    et_obs_train_ = selfmadehydrodataset.read_ts_xrdataset(
        basin_ids, t_range_train, [et_type]
    )
    et_obs_test_ = selfmadehydrodataset.read_ts_xrdataset(
        basin_ids, t_range_test, [et_type]
    )

    et_sim_train = et_sim_train_["etsim"]
    et_sim_test = et_sim_test_["etsim"]
    et_obs_train = et_obs_train_["8D"][et_type]
    et_obs_test = et_obs_test_["8D"][et_type]
    return et_sim_train, et_sim_test, et_obs_train, et_obs_test


def read_sceua_xaj_et_metric(result_dir, et_type=ET_MODIS_NAME):
    (
        pred_train_,
        pred_valid_,
        obs_train_,
        obs_valid_,
    ) = read_sceua_xaj_et(result_dir, et_type)
    inds_df_train = pd.DataFrame(
        stat_error(
            obs_train_.transpose("basin", "time").values,
            pred_train_.transpose("basin", "time").values,
            fill_nan="mean",
        )
    )
    inds_df_valid = pd.DataFrame(
        stat_error(
            obs_valid_.transpose("basin", "time").values,
            pred_valid_.transpose("basin", "time").values,
            fill_nan="mean",
        )
    )
    return inds_df_train, inds_df_valid


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


if __name__ == "__main__":
    # read_sceua_xaj_streamflow(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
    # read_sceua_xaj_streamflow_metric(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
    # read_sceua_xaj_et(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
    read_sceua_xaj_et_metric(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
