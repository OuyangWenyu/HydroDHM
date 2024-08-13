"""
Author: Wenyu Ouyang
Date: 2024-08-13 20:14:37
LastEditTime: 2024-08-13 20:35:02
LastEditors: Wenyu Ouyang
Description: Functions for plotting the results of the model
FilePath: \HydroDHM\scripts\results_plot_utils.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import fnmatch
import os
from pathlib import Path

from matplotlib import pyplot as plt
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns

from hydroutils.hydro_plot import (
    plot_boxes_matplotlib,
    plot_ts,
    plot_map_carto,
    plot_rainfall_runoff,
)
from hydroutils.hydro_file import unserialize_json
from hydroutils.hydro_time import t_range_days
from hydrodataset import Camels
from torchhydro import SETTING
from torchhydro.configs.model_config import MODEL_PARAM_DICT

project_dir = os.getcwd()
RESULT_DIR = os.path.join(project_dir, "results")


def plot_stations_in_sanxia(
    data_map,
    pertile_range=[0, 100],
    fig_size=(10, 6),
    cmap_str="jet",
    vmin=None,
    vmax=None,
):
    """Only for the 18 basins in the chosen_stations.shp file

    Parameters
    ----------
    data_map : _type_
        _description_
    pertile_range : list, optional
        _description_, by default [0, 100]
    fig_size : tuple, optional
        _description_, by default (10, 6)
    cmap_str : str, optional
        _description_, by default "jet"
    vmin : _type_, optional
        _description_, by default None
    vmax : _type_, optional
        _description_, by default None
    """
    cc_shpfile_dir = os.path.join(
        SETTING["local_data_path"]["basins-interim"], "shapes"
    )
    sites = gpd.read_file(
        os.path.join(cc_shpfile_dir, "chosen_stations.shp"), encoding="gbk"
    )
    sites_sorted = sites.sort_values(by="BasinID")
    stid = sites_sorted.StationNam.values
    lat = sites_sorted.Latitude.values
    lon = sites_sorted.Longitude.values

    # http://c.biancheng.net/matplotlib/9284.html
    # plt.rcParams["font.sans-serif"] = ["Ubuntu"]  # 设置字体 for Chinese
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": proj})
    # Show a region
    ax.set_extent([102.2, 109.2, 26.5, 32.5])

    ax.add_feature(cartopy.feature.LAND, edgecolor="black")
    ax.add_feature(cartopy.feature.LAKES, edgecolor="black")

    gl = ax.gridlines(draw_labels=True)
    # https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    gl.xlabel_style = {"size": 19, "color": "gray"}
    gl.ylabel_style = {"size": 19, "color": "gray"}
    # Make figure larger
    plt.gcf().set_size_inches(fig_size)

    # Read shape file
    reader = shpreader.Reader(
        os.path.join(cc_shpfile_dir, "river_in_sanxia_chosen.shp")
    )
    geo_records = list(reader.records())
    for geo_record in geo_records:
        geometry = geo_record.geometry
        # refer to: https://scitools.org.uk/cartopy/docs/v0.15/examples/hurricane_katrina.html
        ax.add_geometries(
            [geometry],
            proj,
            facecolor="none",
            edgecolor="blue",
            alpha=0.25,
        )
    if vmin is not None and vmax is not None:
        scat = ax.scatter(
            lon,
            lat,
            s=300,
            c=data_map,
            cmap=cmap_str,
            edgecolor="black",
            alpha=1,
            transform=proj,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        scat = ax.scatter(
            lon,
            lat,
            s=300,
            c=data_map,
            cmap=cmap_str,
            edgecolor="black",
            alpha=1,
            transform=proj,
        )
    cbar = fig.colorbar(scat, orientation="vertical")
    cbar.ax.tick_params(labelsize=25)
    for count, site in enumerate(stid):
        ax.text(
            lon[count],
            lat[count],
            site,
            horizontalalignment="right",
            transform=proj,
            fontsize=28,
        )


def plot_camel_cc_map(exps, inds_df_lst):
    for i in range(len(inds_df_lst)):
        cfg_dir_flow = os.path.join(RESULT_DIR, "camels", exps[i])
        data_map = inds_df_lst[i]["NSE"].values
        # from mplfonts import use_font
        # use_font("Noto Serif CJK SC")
        f = plot_stations_in_sanxia(
            data_map,
            pertile_range=[0, 100],
            fig_size=(16, 10),
            cmap_str="jet",
            vmin=0.0,
            vmax=0.8,
        )
        FIGURE_DPI = 600
        plt.savefig(
            os.path.join(cfg_dir_flow, "camels_cc_map_" + exps[i] + ".png"),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )


def plot_dpl_comp_boxplots(
    exps,
    inds_df_lst,
    leg_names,
    colors,
    fig_size,
    fig_name,
    subplots_adjust_wspace,
    show_inds=["Bias", "RMSE", "Corr", "NSE"],
):
    concat_inds = [
        [df[ind].values if type(df) is pd.DataFrame else df[ind] for df in inds_df_lst]
        for ind in show_inds
    ]
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # https://www.statology.org/change-font-size-matplotlib/
    plt.rc("axes", labelsize=16)
    plt.rc("ytick", labelsize=12)
    FIGURE_DPI = 600
    plot_boxes_matplotlib(
        concat_inds,
        label1=show_inds,
        label2=leg_names,
        colorlst=colors,
        figsize=fig_size,
        subplots_adjust_wspace=subplots_adjust_wspace,
        show_median=False,
        median_line_color="white",
    )
    cfg_dir = os.path.join(RESULT_DIR, "camels", exps[-1])
    plt.savefig(
        os.path.join(cfg_dir, fig_name),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


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
    all_dirs = os.listdir(os.path.join(RESULT_DIR, "camels", exp))
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


def plot_camels_cc_xaj_params_heatmap(
    exps,
    leg_names,
    sites_Chinese,
    kfold=1,
    the_fold=0,
    sceua_plan="HFsourcesrep1000ngs1000",
    example="camels",
    result_dir=None,
    param_test_way=[
        None,
        MODEL_PARAM_TEST_WAY["final_period"],
        MODEL_PARAM_TEST_WAY["final_period"],
    ],
    concat=False,
    concat_leg_names=None,
):
    """Plot CAMELS CC XAJ models' parameters heatmap

    Parameters
    ----------
    exps : _type_
        _description_
    leg_names : _type_
        _description_
    sites_Chinese : _type_
        Chinese names of sites
    kfold : int, optional
        the k of cross fold valid, by default 1
    the_fold : int, optional
        which fold we will show, by default 0
    sceua_plan : str, optional
        _description_, by default "HFsourcesrep1000ngs1000"
    example : str, optional
        _description_, by default "camels"
    result_dir : _type_, optional
        _description_, by default None
    """
    if result_dir is None:
        result_dir = os.path.join(RESULT_DIR, example, exps[-1])
    attrs_type = None
    parameters_concat = []
    for i in range(len(exps)):
        cfg_dir_flow = os.path.join(RESULT_DIR, example, exps[i])
        if leg_names[i] in ["SCE-UA", "SCEUA", "SCE_UA", "新安江模型"]:
            parameters, params = get_pbm_params_from_hydromodelxaj(
                exps[i], kfold, the_fold, sceua_plan, example, cfg_dir_flow
            )
        elif param_test_way[i] == MODEL_PARAM_TEST_WAY["final_period"]:
            parameters, params = get_pbm_params_from_dpl(cfg_dir_flow)
        elif param_test_way[i] == MODEL_PARAM_TEST_WAY["varying_period"]:
            # TODO: add varying period params reading
            parameters, params = get_pbm_params_from_dpl(cfg_dir_flow)

        if attrs_type is None:
            if leg_names[i] in ["SCE-UA", "SCEUA", "SCE_UA", "新安江模型"]:
                cfg_dir_flow = os.path.join(
                    RESULT_DIR,
                    example,
                    exps[i + 1],
                )
            cfg_flow = get_json_file(cfg_dir_flow)
            attrs_type = cfg_flow["data_params"]["constant_cols"]
            sites = cfg_flow["data_params"]["object_ids"]
            camels = Camels(
                os.path.join(
                    SETTING["local_data_path"]["basins-interim"], "camels", "camels_cc"
                ),
                download=False,
                region="CC",
            )
            attrs = camels.read_constant_cols(sites, attrs_type)
            first_params_file = get_latest_pbm_param_file(cfg_dir_flow)
            params_type = pd.read_csv(first_params_file).columns.values[1:]
            params_type = np.array(
                ["$\Theta$" if tmp == "THETA" else tmp for tmp in params_type]
            )
        corrs = np.zeros((params.shape[0], len(attrs_type)))
        for j in range(corrs.shape[0]):
            for k in range(corrs.shape[1]):
                corrs[j][k] = np.corrcoef(params[j, :], attrs[:, k])[0, 1]

        corrs_shown = np.hstack((corrs[:, 0:7], corrs[:, 8:10], corrs[:, 11:]))
        # all low or high prec_timing is same so no corr
        attrs_type_shown = np.hstack(
            (attrs_type[:7], attrs_type[8:10], attrs_type[11:])
        )
        parameters_concat.append(parameters[:15])
        pd.DataFrame(params[:15], columns=sites_Chinese, index=params_type).to_csv(
            os.path.join(
                result_dir,
                "dpl_params_" + exps[i] + "_fold" + str(the_fold) + ".csv",
            )
        )
        plt.figure()
        sns.heatmap(
            pd.DataFrame(parameters[:15], columns=sites_Chinese, index=params_type),
            cmap="RdBu_r",
            fmt=".2g",
            # square=True,
            annot=True,
        )
        plt.savefig(
            os.path.join(
                result_dir,
                "dpl_params_values_" + exps[i] + "_fold" + str(the_fold) + ".png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
        if attrs.shape[-1] > 0:
            plt.figure()
            sns.heatmap(
                pd.DataFrame(corrs_shown, columns=attrs_type_shown, index=params_type),
                cmap="RdBu_r",
                # fig_size=corrs.shape,
                fmt=".2g",
                # square=False,
                annot=True,
            )
            plt.savefig(
                os.path.join(
                    result_dir,
                    "dpl_params_attrs_" + exps[i] + "_fold" + str(the_fold) + ".png",
                ),
                dpi=600,
                bbox_inches="tight",
            )
    if concat:
        if parameters_concat[0].shape[-1] > 1:
            raise ValueError("only support concating for one basin")
        plt.figure()
        sns.heatmap(
            pd.DataFrame(
                np.array(parameters_concat).reshape(len(parameters_concat), -1).T,
                columns=concat_leg_names,
                index=params_type,
            ),
            cmap="RdBu_r",
            fmt=".2g",
            # square=True,
            annot=True,
        )
        plt.savefig(
            os.path.join(
                result_dir,
                f"dpl_params_concat_values_{exps[0]}_fold{str(the_fold)}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )


def plot_dpl_comp_boxplots(
    exps,
    inds_df_lst,
    leg_names,
    colors,
    fig_size,
    fig_name,
    subplots_adjust_wspace,
    show_inds=["Bias", "RMSE", "Corr", "NSE"],
):
    concat_inds = [
        [df[ind].values if type(df) is pd.DataFrame else df[ind] for df in inds_df_lst]
        for ind in show_inds
    ]
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    # https://www.statology.org/change-font-size-matplotlib/
    plt.rc("axes", labelsize=16)
    plt.rc("ytick", labelsize=12)
    FIGURE_DPI = 600
    plot_boxes_matplotlib(
        concat_inds,
        label1=show_inds,
        label2=leg_names,
        colorlst=colors,
        figsize=fig_size,
        subplots_adjust_wspace=subplots_adjust_wspace,
        show_median=False,
        median_line_color="white",
    )
    cfg_dir = os.path.join(RESULT_DIR, "camels", exps[-1])
    plt.savefig(
        os.path.join(cfg_dir, fig_name),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def plot_time_series(
    exps, obss, preds, shown_sites, sites_Chinese, time_range, cases_in_legend
):
    """Plot time series for shown_sites with multiple exps' results

    Parameters
    ----------
    exps : _type_
        _description_
    obss : _type_
        _description_
    shown_sites : _type_
        _description_
    time_range : _type_
        _description_
    """
    # from mplfonts import use_font
    # use_font("Noto Serif CJK SC")
    dates = t_range_days(time_range)
    cfg_dir_flow = os.path.join(RESULT_DIR, "camels", exps[0])
    cfg_flow = get_json_file(cfg_dir_flow)
    test_period = cfg_flow["data_params"]["t_range_test"]
    warmup_length = cfg_flow["data_params"]["warmup_length"]
    sites_all = cfg_flow["data_params"]["object_ids"]
    test_periods = t_range_days(test_period)
    inter_dates, i1, i2 = np.intersect1d(
        test_periods[warmup_length:], dates, return_indices=True
    )
    # preds and obs
    # all obs in obss are same
    obs = obss[0][:, i1]
    cases_len = len(exps) + 1
    for site in shown_sites:
        data_lst = []
        site_idx = sites_all.index(site)
        for pred in preds:
            if pred.shape[1] <= len(i1):
                # for LSTM and SCE-UA we directly provide data in time_range
                data_lst.append(pred[site_idx, :])
            else:
                data_lst.append(pred[site_idx, i1])
        data_lst.append(obs[site_idx, :])
        plot_ts(
            np.tile(dates, (cases_len, 1)).tolist(),
            np.array(data_lst).tolist(),
            leg_lst=cases_in_legend,
            title=site + " " + sites_Chinese[site_idx],
            fig_size=(8, 4),
            xlabel="date",
            ylabel="m$^3$/s",
            linewidth=1,
        )
        plt.savefig(
            os.path.join(cfg_dir_flow, "camels_cc_ts_" + site + ".png"),
            dpi=600,
            bbox_inches="tight",
        )


def plot_computing_time(exps, leg_names):
    """plot computing time for exps

    Parameters
    ----------
    exps : _type_
        _description_
    leg_names : _type_
        legends shown in the plot
    """
    times_lst = []
    losses_lst = []
    runtimes_lst = []
    for i in range(len(exps)):
        if leg_names[i] == "SCE-UA":
            print()
            # SCE-UA's record is different with others
        else:
            cfg_dir_flow = os.path.join(RESULT_DIR, "camels", exps[i])
            cfg_flow = get_json_file(cfg_dir_flow)
            sites_num = len(cfg_flow["data_params"]["object_ids"])
            run_record = cfg_flow["run"]
            times = []
            losses = []
            time_now = 0
            iter_num = int(run_record[i]["iter_num"])
            for j in range(len(run_record)):
                # example for time record: Epoch 1 Loss 2.071 time 64.51
                time_now = time_now + float(run_record[j]["time"].split(" ")[-1])
                times.append(time_now)
                losses.append(float(run_record[j]["train_loss"]))
            epochs = np.arange(1, len(run_record) + 1)
            runtimes = epochs * iter_num * sites_num
        times_lst.append(times)
        losses_lst.append(losses)
        runtimes_lst.append(runtimes)

    plot_ts(
        times_lst,
        losses_lst,
        leg_lst=leg_names,
        fig_size=(8, 4),
        xlabel="time (s)",
        ylabel="RMSE",
        linewidth=1,
    )
    plt.savefig(
        os.path.join(cfg_dir_flow, "camels_cc_computing_time.png"),
        dpi=600,
        bbox_inches="tight",
    )

    plot_ts(
        times_lst,
        runtimes_lst,
        leg_lst=leg_names,
        fig_size=(8, 4),
        xlabel="runs",
        ylabel="RMSE",
        linewidth=1,
    )
    plt.savefig(
        os.path.join(cfg_dir_flow, "camels_cc_run_times.png"),
        dpi=600,
        bbox_inches="tight",
    )


def plot_camels_nse_map(inds_df_lst, exps):
    camels = Camels(os.path.join(definitions.DATASET_DIR, "camels", "camels_us"))
    lat_lon = camels.read_constant_cols(
        camels.camels_sites["gauge_id"].values, ["gauge_lat", "gauge_lon"]
    )
    cfg_dir_flow = os.path.join(RESULT_DIR, "camels", exps[0])
    cfg_flow = get_json_file(cfg_dir_flow)
    sites = cfg_flow["data_params"]["object_ids"]
    all_sites = camels.camels_sites["gauge_id"].values
    sites_chosen, idx1, idx2 = np.intersect1d(all_sites, sites, return_indices=True)
    for i in range(len(exps)):
        plot_map_carto(
            inds_df_lst[i]["NSE"].values,
            lat=lat_lon[idx1, 0],
            lon=lat_lon[idx1, 1],
            # pertile_range=[0, 100],
            value_range=[0, 1],
        )
        FIGURE_DPI = 600
        plt.savefig(
            os.path.join(
                RESULT_DIR,
                "camels",
                exps[i],
                "dpl_camels_nse_map_" + exps[i] + ".png",
            ),
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )


def plot_ts_for_basin_fold(
    leg_lst,
    basin_id,
    fold,
    step_lst,
    value_lst,
    ylabel,
    where_save="transfer_learning",
    sub_dir=os.path.join("results", "tensorboard"),
    batch_size=None,
):
    """Lineplot for loss and metric of DL models for one basin in a fold experiment

    Parameters
    ----------
    leg_lst : list
        a list of legends
    basin_id : _type_
        _description_
    fold : _type_
        _description_
    step_lst : _type_
        _description_
    value_lst : _type_
        _description_
    ylabel : _type_
        _description_
    where_save : str, optional
        _description_, by default "transfer_learning"
    """
    result_dir = os.path.join(
        RESULT_DIR,
        where_save,
        sub_dir,
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plot_ts(
        step_lst,
        value_lst,
        leg_lst=leg_lst,
        fig_size=(6, 4),
        xlabel="代数",
        ylabel=ylabel,
    )
    if batch_size is None:
        plt.savefig(
            os.path.join(
                result_dir,
                f"{basin_id}_fold{fold}_{ylabel}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(
                result_dir,
                f"{basin_id}_fold{fold}_{ylabel}_bsize{batch_size}.png",
            ),
            dpi=600,
            bbox_inches="tight",
        )


def plot_xaj_lstm_tl_dpl_rainfall_runoff(
    time_range, data_ts, leg_names, basin_id, save_file, alpha=0.5, c_lst=None
):
    if c_lst is None:
        c_lst = ["red", "green", "blue", "black"]
    camels_cc = Camels(
        os.path.join(definitions.DATASET_DIR, "camels", "camels_cc"), region="CC"
    )
    cc_shpfile_dir = os.path.join(
        definitions.ROOT_DIR, "hydroSPB", "example", "shpfile"
    )
    sites = gpd.read_file(
        os.path.join(cc_shpfile_dir, "chosen_stations.shp"), encoding="gbk"
    )
    sites_Chinese = sites[sites["BasinID"] == basin_id]
    prcps = camels_cc.read_relevant_cols([basin_id], time_range, [PRCP_ERA5LAND_NAME])
    t_periods = t_range_days(time_range)
    dash_lines = [False] * len(leg_names)
    dash_lines[-1] = True
    plot_rainfall_runoff(
        np.tile(t_periods, (len(leg_names), 1)).tolist(),
        prcps[0, :, 0],
        np.array(data_ts).tolist(),
        leg_lst=leg_names,
        # title=site + " " + sites_Chinese[site_idx],
        title=sites_Chinese["StationNam"].values[0],
        fig_size=(12, 6),
        # xlabel="date",
        # ylabel="streamflow (m$^3$/s)",
        xlabel="日期",
        ylabel="径流（m$^3$/s）",
        linewidth=0.75,
        dash_lines=dash_lines,
        alpha=alpha,
        c_lst=c_lst,
    )
    plt.savefig(
        save_file,
        dpi=600,
        bbox_inches="tight",
    )