import fnmatch
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_file import unserialize_json
from hydromodel.datasets.data_preprocess import cross_val_split_tsdata
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.config import cmd, update_cfg
from torchhydro.configs.model_config import MODEL_PARAM_DICT
from torchhydro.trainers.resulter import Resulter, get_latest_pbm_param_file
from torchhydro.trainers.trainer import train_and_evaluate
from torchhydro.trainers.train_utils import read_pth_from_model_loader

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import RESULT_DIR, DATASET_DIR
from hydrodhm.run_xaj.evaluate_xaj import _evaluate_1fold

ET_MODIS_NAME = "ET_modis16a2006"
# ET_MODIS_NAME = "ET_modis16a2gf061"


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
    config_yml_file = os.path.join(result_dir, "config.yaml")
    with open(config_yml_file, "r") as file:
        config_data = yaml.safe_load(file)
    warmup = config_data["warmup"]
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
    qsim_train = data_train["qsim"].isel(time=slice(warmup, None))
    qsim_test = data_test["qsim"].isel(time=slice(warmup, None))
    qobs_train = data_train["qobs"].isel(time=slice(warmup, None))
    qobs_test = data_test["qobs"].isel(time=slice(warmup, None))
    return [
        qsim_train,
        qsim_test,
        qobs_train,
        qobs_test,
    ]


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


def update_hydromodel_cfgs(result_dir):
    config_yml_file = os.path.join(result_dir, "config.yaml")
    with open(config_yml_file, "r") as file:
        config_data = yaml.safe_load(file)
    config_data["data_dir"] = DATASET_DIR
    config_data["param_range_file"] = os.path.join(result_dir, "param_range.yaml")
    # read denorm_params file
    norm_params, denorm_params = get_pbm_params_from_hydromodelxaj(result_dir)
    if len(denorm_params.columns) > 15:
        kernel_size = int(denorm_params.iloc[:, 15].iloc[0])
        config_data["model"]["kernel_size"] = kernel_size
        denorm_params = denorm_params.iloc[:, :15]
        norm_params = norm_params.iloc[:, :15]
        # save the updated denorm_params file
        denorm_params_file = os.path.join(
            result_dir, "sceua_xaj", "basins_denorm_params.csv"
        )
        denorm_params.to_csv(denorm_params_file)
        # save the updated norm_params file
        norm_params_file = os.path.join(
            result_dir, "sceua_xaj", "basins_norm_params.csv"
        )
        norm_params.to_csv(norm_params_file)
    # read old param_range file
    with open(config_data["param_range_file"], "r") as file:
        param_range = yaml.safe_load(file)
    # delete KERNEL keys in param_range
    if "KERNEL" in param_range["xaj_mz"]["param_name"]:
        param_range["xaj_mz"]["param_name"].remove("KERNEL")
        param_range["xaj_mz"]["param_range"].pop("KERNEL")
    # update the param_range file
    with open(config_data["param_range_file"], "w") as file:
        yaml.dump(param_range, file)
    # save the updated config file
    with open(config_yml_file, "w") as file:
        yaml.dump(config_data, file)
    # read the calibrated parameter file and delete the KERNEL column
    basin_ids = config_data["basin_id"]
    for basin_id in basin_ids:
        param_file = os.path.join(result_dir, "sceua_xaj", f"{basin_id}.csv")
        params = pd.read_csv(param_file)
        if "parKERNEL" in params.columns:
            params = params.drop(columns=["parKERNEL"])
            params.to_csv(param_file, index=False)
    return config_data


def read_sceua_xaj_et(result_dir, et_type=ET_MODIS_NAME):
    config_data = update_hydromodel_cfgs(result_dir)
    basin_ids = config_data["basin_id"]
    warmup = config_data["warmup"]
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
        # close the opened files
        et_sim_train_.close()
        et_sim_test_.close()
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
        et_sim_train_ = xr.open_dataset(train_result_file)
        et_sim_test_ = xr.open_dataset(test_result_file)
    t_range_train = [et_sim_train_.time.values[0], et_sim_train_.time.values[-1]]
    t_range_test = [et_sim_test_.time.values[0], et_sim_test_.time.values[-1]]

    et_obs_train_, et_obs_test_ = _read_et_obs(
        et_type, basin_ids, t_range_train, t_range_test
    )

    et_sim_train = et_sim_train_["etsim"].isel(time=slice(warmup, None))
    et_sim_test = et_sim_test_["etsim"].isel(time=slice(warmup, None))
    desired_train_time = et_sim_train.time.values
    desired_test_time = et_sim_test.time.values
    et_obs_train = et_obs_train_["8D"][et_type].sel(
        time=slice(desired_train_time[0], desired_train_time[-1])
    )
    et_obs_test = et_obs_test_["8D"][et_type].sel(
        time=slice(desired_test_time[0], desired_test_time[-1])
    )
    return [et_sim_train, et_sim_test, et_obs_train, et_obs_test]


def _read_et_obs(et_type, basin_ids, t_range_train, t_range_test):
    selfmadehydrodataset = SelfMadeHydroDataset(DATASET_DIR, time_unit=["8D"])
    et_obs_train_ = selfmadehydrodataset.read_ts_xrdataset(
        basin_ids, t_range_train, [et_type]
    )
    et_obs_test_ = selfmadehydrodataset.read_ts_xrdataset(
        basin_ids, t_range_test, [et_type]
    )
    return et_obs_train_, et_obs_test_


def read_sceua_xaj_et_metric(result_dir, et_type=ET_MODIS_NAME):
    (
        pred_train_,
        pred_valid_,
        obs_train_,
        obs_valid_,
    ) = read_sceua_xaj_et(result_dir, et_type)
    inds_df_train, inds_df_valid = _xrarray_cal_et_metric(
        pred_train_, pred_valid_, obs_train_, obs_valid_
    )
    return inds_df_train, inds_df_valid


def _xrarray_cal_et_metric(pred_train_, pred_valid_, obs_train_, obs_valid_):
    obs_train_aligned, pred_train_aligned = align_time_and_interval(
        obs_train_, pred_train_
    )

    obs_valid_aligned, pred_valid_aligned = align_time_and_interval(
        obs_valid_,
        pred_valid_,
    )

    # Now compute metrics using aligned data
    inds_df_train = pd.DataFrame(
        stat_error(
            obs_train_aligned.transpose("basin", "time").values,
            pred_train_aligned.transpose("basin", "time").values,
            fill_nan="mean",
        )
    )
    inds_df_valid = pd.DataFrame(
        stat_error(
            obs_valid_aligned.transpose("basin", "time").values,
            pred_valid_aligned.transpose("basin", "time").values,
            fill_nan="mean",
        )
    )

    return inds_df_train, inds_df_valid


def align_time_and_interval(obs_, pred_):
    obs_time_min = pd.Timestamp(obs_.time.min().item())
    obs_time_max = pd.Timestamp(obs_.time.max().item())
    pred_time_min = pd.Timestamp(pred_.time.min().item())
    pred_time_max = pd.Timestamp(pred_.time.max().item())

    # Create common time index for training data
    common_time_index = pd.date_range(
        start=max(obs_time_min, pred_time_min),
        end=min(obs_time_max, pred_time_max),
        freq="D",
    )

    # Reindex obs_train_ and pred_train_ to align by time
    obs_aligned = obs_.reindex(time=common_time_index, method=None)
    pred_aligned = pred_.reindex(time=common_time_index, method=None)
    return obs_aligned, pred_aligned


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
    return cfg_json


def update_dl_cfg_paths(cfg_dir_):
    """Update the paths in cfgs when results from one computer are used in another computer

    Parameters
    ----------
    cfg_dir_ : _type_
        _description_
    """
    cfg_ = get_json_file(cfg_dir_)
    cfg_["data_cfgs"]["source_cfgs"]["source_path"] = DATASET_DIR
    cfg_["data_cfgs"]["validation_path"] = cfg_dir_
    cfg_["data_cfgs"]["test_path"] = cfg_dir_
    cfg_["evaluation_cfgs"]["metrics"] = [
        "Bias",
        "RMSE",
        "Corr",
        "NSE",
        "KGE",
        "FLV",
        "FHV",
    ]
    return cfg_


def _cfg4trainperiod(cfg):
    """A new cfg for training period data simulation

    Parameters
    ----------
    project_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    train_period = cfg["data_cfgs"]["t_range_train"]
    valid_period = train_period
    test_period = train_period
    old_model_loader = cfg["evaluation_cfgs"]["model_loader"]
    old_pth_dir = cfg["data_cfgs"]["test_path"]
    weight_path = read_pth_from_model_loader(old_model_loader, old_pth_dir)
    new_args = cmd(
        model_type="MTL",
        ctx=[0],
        # loss_func="RMSESum",
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        # NOTE: although we set total_evaporation_hourly as output, it is not used in the training process
        var_out=["streamflow", "total_evaporation_hourly"],
        n_output=2,
        # TODO: if chose "mean", metric results' format is different, this should be refactored
        fill_nan=["no", "no"],
        train_mode=0,
        weight_path=weight_path,
        model_loader={"load_way": "pth", "pth_path": weight_path},
        continue_train=0,
        metrics=["Bias", "RMSE", "Corr", "NSE", "KGE", "FLV", "FHV"],
    )
    update_cfg(cfg, new_args)
    cfg["data_cfgs"]["validation_path"] = (
        cfg["data_cfgs"]["validation_path"] + "_trainperiod"
    )
    cfg["data_cfgs"]["test_path"] = cfg["data_cfgs"]["test_path"] + "_trainperiod"
    if not os.path.exists(cfg["data_cfgs"]["test_path"]):
        os.makedirs(cfg["data_cfgs"]["test_path"])
    # find the data_dict.json file and copy it to the new directory
    for file in os.listdir(old_pth_dir):
        if fnmatch.fnmatch(file, "*.json") and "_stat" in file:  # statistics json file
            data_dict_file = os.path.join(old_pth_dir, file)
            break
    shutil.copy2(
        data_dict_file,
        os.path.join(cfg["data_cfgs"]["test_path"], data_dict_file.split(os.sep)[-1]),
    )
    return cfg


def cfgrunagain(cfg):
    """A new cfg for test period data simulation

    Parameters
    ----------
    project_name : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    old_model_loader = cfg["evaluation_cfgs"]["model_loader"]
    old_pth_dir = cfg["data_cfgs"]["test_path"]
    weight_path = read_pth_from_model_loader(old_model_loader, old_pth_dir)
    new_args = cmd(
        model_type="MTL",
        ctx=[0],
        # loss_func="RMSESum",
        loss_func="MultiOutLoss",
        loss_param={
            "loss_funcs": "RMSESum",
            "data_gap": [0, 0],
            "device": [0],
            "item_weight": [1, 0],
            "limit_part": [1],
        },
        # NOTE: although we set total_evaporation_hourly as output, it is not used in the training process
        var_out=["streamflow", "total_evaporation_hourly"],
        n_output=2,
        # TODO: if chose "mean", metric results' format is different, this should be refactored
        fill_nan=["no", "no"],
        train_mode=0,
        weight_path=weight_path,
        continue_train=0,
        metrics=["Bias", "RMSE", "Corr", "NSE", "KGE", "FLV", "FHV"],
    )
    update_cfg(cfg, new_args)
    return cfg


def read_dpl_model_q_and_et(cfg_dir_, cfg_dir_train=None, cfg_runagain=False):
    """read dl models simulations

    Parameters
    ----------
    cfg_dir_: str
        the directory of the DL model
    cfg_dir_train: str, optional
        the directory of the DL model for its training period data simulation
        if not provided, the training period data simulation will be the value specified in the cfg_dir_
    cfg_runagain: bool, optional
        whether to run the DL model in cfg_dir_ for cfg_dir_ again, by default False

    Returns
    -------
    tuple[list[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset], list[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]]
        [
            qsim_train,
            qsim_test,
            qobs_train,
            qobs_test,
        ],
        [
            etsim_train,
            etsim_test,
            etobs_train,
            etobs_test,
        ],
    """
    if cfg_dir_train is None:
        cfg_dir_train = cfg_dir_ + "_trainperiod"
        if not os.path.exists(os.path.join(cfg_dir_train, "metric_streamflow.csv")):
            # if the metric file exists, we do not need to run the model again
            # we need to run the trained model to get the training period data simulation
            cfg_train = update_dl_cfg_paths(cfg_dir_)
            cfg_train = _cfg4trainperiod(cfg_train)
            try:
                train_and_evaluate(cfg_train)
            except KeyError:
                cfg_train["training_cfgs"]["train_mode"] = True
                cfg_train["data_cfgs"]["stat_dict_file"] = None
                train_and_evaluate(cfg_train)
            cfg_dir_train = cfg_train["data_cfgs"]["test_path"]
    cfg_train = get_json_file(cfg_dir_train)
    resulter = Resulter(cfg_train)
    pred_train, obs_train = resulter.load_result(convert_flow_unit=True)
    resulter.eval_result(pred_train, obs_train)

    cfg_test = update_dl_cfg_paths(cfg_dir_)
    if cfg_runagain:
        cfg_test = cfgrunagain(cfg_test)
        try:
            train_and_evaluate(cfg_test)
        except KeyError:
            cfg_test["training_cfgs"]["train_mode"] = True
            cfg_test["data_cfgs"]["stat_dict_file"] = None
            train_and_evaluate(cfg_test)
    resulter = Resulter(cfg_test)
    pred_test, obs_test = resulter.load_result(convert_flow_unit=True)
    resulter.eval_result(pred_test, obs_test)

    basin_ids = cfg_test["data_cfgs"]["object_ids"]
    t_range_train = [obs_train.time.values[0], obs_train.time.values[-1]]
    t_range_test = [obs_test.time.values[0], obs_test.time.values[-1]]
    et_obs_train_, et_obs_test_ = _read_et_obs(
        ET_MODIS_NAME, basin_ids, t_range_train, t_range_test
    )
    qsim_train = pred_train["streamflow"]
    qsim_test = pred_test["streamflow"]
    qobs_train = obs_train["streamflow"]
    qobs_test = obs_test["streamflow"]
    etsim_train = pred_train["total_evaporation_hourly"]
    etsim_test = pred_test["total_evaporation_hourly"]
    etobs_train = et_obs_train_["8D"][ET_MODIS_NAME]
    etobs_test = et_obs_test_["8D"][ET_MODIS_NAME]
    return (
        [
            qsim_train,
            qsim_test,
            qobs_train,
            qobs_test,
        ],
        [
            etsim_train,
            etsim_test,
            etobs_train,
            etobs_test,
        ],
    )


def read_dpl_model_metric(cfg_dir_test, cfg_dir_train):
    """read the metrics of DL models for one basin in k-fold cross validation

    Parameters
    ----------
    epoch : int
        the epoch of the DL model
    cv_fold : int, optional
        the number of folds in cross validation, by default 2
    """
    qs, ets = read_dpl_model_q_and_et(cfg_dir_test, cfg_dir_train)
    train_metrics_file = os.path.join(
        cfg_dir_train,
        "metric_streamflow.csv",
    )
    test_metrics_file = os.path.join(
        cfg_dir_test,
        "metric_streamflow.csv",
    )
    train_metric_q = pd.read_csv(train_metrics_file, index_col=0)
    test_metric_q = pd.read_csv(test_metrics_file, index_col=0)
    inds_df_train_et, inds_df_valid_et = _xrarray_cal_et_metric(*ets)
    return [train_metric_q, test_metric_q, inds_df_train_et, inds_df_valid_et]


def get_pbm_params_from_hydromodelxaj(result_dir):
    normlize_param_file = os.path.join(
        result_dir, "sceua_xaj", "basins_norm_params.csv"
    )
    norm_params = pd.read_csv(normlize_param_file, index_col=0)
    params_file = os.path.join(result_dir, "sceua_xaj", "basins_denorm_params.csv")
    denorm_params = pd.read_csv(params_file, index_col=0)
    return norm_params, denorm_params


def get_pbm_params_from_dpl(result_dir):
    params_file = get_latest_pbm_param_file(result_dir)
    if params_file is None:
        _save_pbm_params(result_dir)
        params_file = get_latest_pbm_param_file(result_dir)
    parameters = pd.read_csv(params_file).iloc[:, 1:].values
    params = _denorm_pbm_param(parameters)
    params = params.T
    parameters = parameters.T
    return parameters, params


def _save_pbm_params(result_dir):
    cfg_ = update_dl_cfg_paths(result_dir)
    resulter = Resulter(cfg_)
    resulter.save_intermediate_results(is_pbm_params=True)


def _denorm_pbm_param(norm_params):
    denorm_params = np.zeros(norm_params.shape)
    param_range = MODEL_PARAM_DICT["xaj_mz"]["param_range"]
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
    denorm_params[:, 0] = k_scale[0] + norm_params[:, 0] * (k_scale[1] - k_scale[0])
    denorm_params[:, 1] = b_scale[0] + norm_params[:, 1] * (b_scale[1] - b_scale[0])
    denorm_params[:, 2] = im_sacle[0] + norm_params[:, 2] * (im_sacle[1] - im_sacle[0])
    denorm_params[:, 3] = um_scale[0] + norm_params[:, 3] * (um_scale[1] - um_scale[0])
    denorm_params[:, 4] = lm_scale[0] + norm_params[:, 4] * (lm_scale[1] - lm_scale[0])
    denorm_params[:, 5] = dm_scale[0] + norm_params[:, 5] * (dm_scale[1] - dm_scale[0])
    denorm_params[:, 6] = c_scale[0] + norm_params[:, 6] * (c_scale[1] - c_scale[0])
    denorm_params[:, 7] = sm_scale[0] + norm_params[:, 7] * (sm_scale[1] - sm_scale[0])
    denorm_params[:, 8] = ex_scale[0] + norm_params[:, 8] * (ex_scale[1] - ex_scale[0])
    ki = ki_scale[0] + norm_params[:, 9] * (ki_scale[1] - ki_scale[0])
    kg = kg_scale[0] + norm_params[:, 10] * (kg_scale[1] - kg_scale[0])
    # ki+kg should be smaller than 1; if not, we scale them
    denorm_params[:, 9] = np.where(ki + kg < 1.0, ki, 1 / (ki + kg) * ki)
    denorm_params[:, 10] = np.where(ki + kg < 1.0, kg, 1 / (ki + kg) * kg)
    denorm_params[:, 11] = a_scale[0] + norm_params[:, 11] * (a_scale[1] - a_scale[0])
    denorm_params[:, 12] = theta_scale[0] + norm_params[:, 12] * (
        theta_scale[1] - theta_scale[0]
    )
    denorm_params[:, 13] = ci_scale[0] + norm_params[:, 13] * (
        ci_scale[1] - ci_scale[0]
    )
    denorm_params[:, 14] = cg_scale[0] + norm_params[:, 14] * (
        cg_scale[1] - cg_scale[0]
    )
    return denorm_params


def read_tb_log_loss(result_dir):
    cfg_ = update_dl_cfg_paths(result_dir)
    resulter = Resulter(cfg_)
    df_scalar = resulter.read_tensorboard_log(is_scalar=True)
    df_loss = df_scalar[df_scalar["tag"] == "Loss"]
    df_validloss = df_scalar[df_scalar["tag"] == "ValidLoss"]
    df_validnse = df_scalar[df_scalar["tag"] == "ValidstreamflowNSEmean"]
    return df_loss, df_validloss


if __name__ == "__main__":
    sceua_dir = os.path.join(RESULT_DIR, "XAJ", "changdian_61700_4_4")
    dpl_dir = os.path.join(RESULT_DIR, "dPL", "result", "lrchange3", "changdian_61700")
    dpl_nn_dir = os.path.join(RESULT_DIR, "dPL", "result", "module", "changdian_61700")
    # read_sceua_xaj_streamflow(sceua_dir)
    # read_sceua_xaj_streamflow_metric(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
    # read_sceua_xaj_et(os.path.join(RESULT_DIR, "XAJ", "changdian_61561"))
    # read_sceua_xaj_et_metric(
    #     os.path.join(RESULT_DIR, "XAJ", "result_old", "changdian_61700")
    # )
    # get_pbm_params_from_hydromodelxaj(
    #     os.path.join(RESULT_DIR, "XAJ", "result_old", "changdian_61700")
    # )
    read_dpl_model_q_and_et(dpl_dir)
    read_dpl_model_q_and_et(dpl_nn_dir)
    # read_dpl_model_metric(
    #     os.path.join(
    #         RESULT_DIR,
    #         "dPL",
    #         "streamflow_prediction",
    #         "streamflow_prediction_50epoch",
    #         "changdian_61561",
    #     ),
    #     os.path.join(
    #         RESULT_DIR,
    #         "dPL",
    #         "streamflow_prediction",
    #         "streamflow_prediction_50epoch",
    #         "changdian_61561_trainperiod",
    #     ),
    # )
    get_pbm_params_from_dpl(dpl_dir)
