import os
import sys
import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.io.shapereader as shpreader

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt

from hydroutils.hydro_time import t_range_days
from hydroutils.hydro_plot import (
    plot_boxes_matplotlib,
    plot_ts,
    plot_map_carto,
    plot_rainfall_runoff,
)
from hydrodatasource.reader.data_source import SelfMadeHydroDataset
from torchhydro.configs.model_config import MODEL_PARAM_TEST_WAY
from torchhydro import SETTING
from torchhydro.trainers.train_utils import (
    get_latest_pbm_param_file,
    read_torchhydro_log_json_file,
)

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import DATASET_DIR, RESULT_DIR
from hydrodhm.utils.results_utils import (
    ET_NAME,
    _save_pbm_params,
    get_pbm_params_from_dpl,
    get_pbm_params_from_hydromodelxaj,
    read_dpl_model_q_and_et,
    read_sceua_xaj_et,
    read_sceua_xaj_streamflow,
    read_tb_log_loss,
    read_dpl_model_metric,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

ID_NAME_DICT = {
    "changdian_61561": "Duoyingping",
    "changdian_61700": "Sanhuangmiao",
    "changdian_61716": "Dengyingyan",
    "changdian_62618": "Fujiangqiao",
    "changdian_91000": "Ganzi",
}


def plot_xaj_params_heatmap(
    result_dirs,
    leg_names,
    fig_name,
    param_test_way=[
        None,
        MODEL_PARAM_TEST_WAY["final_period"],
        MODEL_PARAM_TEST_WAY["final_period"],
    ],
    fig_dir=None,
):
    """Plot CAMELS CC XAJ models' parameters heatmap

    Parameters
    ----------
    exp_dirs : _type_
        the directories of experiments
    leg_names : _type_
        _description_
    fig_name : _type_
        name of the figure
    """
    norm_params_concat = []
    if fig_dir is None:
        fig_dir = result_dirs[0]
    for i in range(len(result_dirs)):
        if leg_names[i] != "eXAJ":
            a_result_dir = result_dirs[i]
            first_params_file = get_latest_pbm_param_file(a_result_dir)
            if first_params_file is None:
                _save_pbm_params(a_result_dir)
                first_params_file = get_latest_pbm_param_file(a_result_dir)
            params_type = pd.read_csv(first_params_file).columns.values[1:]
            params_type = np.array(
                ["$\Theta$" if tmp == "THETA" else tmp for tmp in params_type]
            )
            break
    for i in range(len(result_dirs)):
        a_result_dir = result_dirs[i]
        if leg_names[i] == "eXAJ":
            norm_params_, denorm_params_ = get_pbm_params_from_hydromodelxaj(
                a_result_dir
            )
            norm_params = norm_params_.values.T
            denorm_params = denorm_params_.values.T
        elif param_test_way[i] == MODEL_PARAM_TEST_WAY["final_period"]:
            norm_params, denorm_params = get_pbm_params_from_dpl(a_result_dir)
        norm_params_concat.append(norm_params[:15])
    if norm_params_concat[0].shape[-1] > 1:
        raise ValueError("only support concating for one basin")
    plt.figure()
    sns.heatmap(
        pd.DataFrame(
            np.array(norm_params_concat).reshape(len(norm_params_concat), -1).T,
            columns=leg_names,
            index=params_type,
        ),
        cmap="RdBu_r",
        fmt=".2g",
        # square=True,
        annot=True,
    )
    plt.savefig(
        os.path.join(
            fig_dir,
            f"pbm_params_concat_values_{fig_name}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )


def plot_losses_ts(dl_result_dirs, leg_lst, ylabel="Loss", fig_dir=None):
    if fig_dir is None:
        fig_dir = dl_result_dirs[0]
    step_lst = []
    validloss_lst = []
    loss_lst = []
    for i, a_exp in tqdm(enumerate(dl_result_dirs)):
        df_loss, df_validloss = read_tb_log_loss(a_exp)
        step_lst.append(df_loss["step"].values)
        loss_lst.append(df_loss["value"].values)
        validloss_lst.append(df_validloss["value"].values)
    plot_ts(
        step_lst + step_lst,
        loss_lst + validloss_lst,
        leg_lst=leg_lst,
        fig_size=(6, 4),
        xlabel="Epoch",
        ylabel=ylabel,
    )
    plt.savefig(
        os.path.join(
            fig_dir,
            f"dpl_dplnn_{ylabel}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )


def plot_xaj_rainfall_runoff(
    result_dirs,
    basin_id,
    basin_name,
    alpha=0.5,
    c_lst=None,
    leg_names=["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$", "OBS"],
    fig_dir=None,
):
    if fig_dir is None:
        fig_dir = result_dirs[0]
    if c_lst is None:
        c_lst = ["red", "green", "blue", "black"]
    train_ts = []
    valid_ts = []
    train_periods_wo_warmup = []
    valid_periods_wo_warmup = []
    for j, a_result_dir in enumerate(result_dirs):
        if leg_names[j] == "eXAJ":
            [
                pred_train,
                pred_valid,
                obs_train,
                obs_valid,
            ] = read_sceua_xaj_streamflow(
                a_result_dir,
            )
            train_ts.append(pred_train.values.flatten())
            valid_ts.append(pred_valid.values.flatten())
        else:
            (
                [
                    pred_train,
                    pred_valid,
                    obs_train,
                    obs_valid,
                ],
                [
                    etsim_train,
                    etsim_test,
                    etobs_train,
                    etobs_test,
                ],
            ) = read_dpl_model_q_and_et(
                a_result_dir,
            )
            # TODO: all time-intervals are left-right closed, but we forget to set as this in dpl, so we need to remove the last time point
            train_ts.append(pred_train.values.flatten()[:-1])
            valid_ts.append(pred_valid.values.flatten()[:-1])
    train_ts.append(obs_train.values.flatten()[:-1])
    valid_ts.append(obs_valid.values.flatten()[:-1])
    train_periods_wo_warmup.append(obs_train["time"].values[:-1])
    valid_periods_wo_warmup.append(obs_valid["time"].values[:-1])
    dash_lines = [False] * len(leg_names)
    dash_lines[-1] = True
    selfmadehydrodataset = SelfMadeHydroDataset(DATASET_DIR)
    prcps_train = selfmadehydrodataset.read_ts_xrdataset(
        [basin_id],
        [train_periods_wo_warmup[0][0], train_periods_wo_warmup[0][-1]],
        ["total_precipitation_hourly"],
    )
    prcps_valid = selfmadehydrodataset.read_ts_xrdataset(
        [basin_id],
        [valid_periods_wo_warmup[0][0], valid_periods_wo_warmup[0][-1]],
        ["total_precipitation_hourly"],
    )
    # plot train
    plot_rainfall_runoff(
        train_periods_wo_warmup * len(leg_names),
        prcps_train["1D"]["total_precipitation_hourly"].values.flatten(),
        train_ts,
        leg_lst=leg_names,
        # title=site + " " + sites_Chinese[site_idx],
        title=basin_name,
        fig_size=(12, 6),
        xlabel="date",
        ylabel="streamflow (m$^3$/s)",
        linewidth=0.75,
        dash_lines=dash_lines,
        c_lst=c_lst,
    )
    plt.savefig(
        os.path.join(
            fig_dir, basin_id + "_" + basin_name + "_rainfall_runoff_train.png"
        ),
        dpi=600,
        bbox_inches="tight",
    )
    # plot valid
    plot_rainfall_runoff(
        valid_periods_wo_warmup * len(leg_names),
        prcps_valid["1D"]["total_precipitation_hourly"].values.flatten(),
        valid_ts,
        leg_lst=leg_names,
        # title=site + " " + sites_Chinese[site_idx],
        title=basin_name,
        fig_size=(12, 6),
        xlabel="date",
        ylabel="streamflow (m$^3$/s)",
        linewidth=0.75,
        dash_lines=dash_lines,
        c_lst=c_lst,
    )
    plt.savefig(
        os.path.join(
            fig_dir, basin_id + "_" + basin_name + "_rainfall_runoff_valid.png"
        ),
        dpi=600,
        bbox_inches="tight",
    )


def plot_xaj_et_time_series(
    result_dirs,
    basin_id,
    basin_name,
    leg_names=["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$", "OBS"],
    fig_dir=None,
):
    """Plot ET time series

    Parameters
    ----------
    exps : _type_
        _description_
    basin_id : _type_
        _description_
    basin_name : _type_
        _description_
    """
    if fig_dir is None:
        fig_dir = result_dirs[0]
    train_ets = []
    valid_ets = []
    train_periods_wo_warmup = []
    valid_periods_wo_warmup = []
    print(f"Basin {basin_id}:")
    for j in range(len(result_dirs)):
        if leg_names[j] == "eXAJ":
            (
                pred_et_train,
                pred_et_valid,
                obs_et_train,
                obs_et_valid,
            ) = read_sceua_xaj_et(
                result_dirs[j],
            )
            train_ets.append(pred_et_train.values.flatten())
            valid_ets.append(pred_et_valid.values.flatten())
            train_periods_wo_warmup.append(pred_et_train["time"].values)
            valid_periods_wo_warmup.append(pred_et_valid["time"].values)
        else:
            (
                [
                    pred_train,
                    pred_valid,
                    obs_train,
                    obs_valid,
                ],
                [
                    pred_et_train,
                    pred_et_valid,
                    etobs_train,
                    etobs_test,
                ],
            ) = read_dpl_model_q_and_et(
                result_dirs[j],
            )
            train_ets.append(pred_et_train.values.flatten()[:-1])
            valid_ets.append(pred_et_valid.values.flatten()[:-1])
            train_periods_wo_warmup.append(pred_et_train["time"].values[:-1])
            valid_periods_wo_warmup.append(pred_et_valid["time"].values[:-1])
    train_ets.append(obs_et_train.values.flatten())
    valid_ets.append(obs_et_valid.values.flatten())
    train_periods_wo_warmup.append(obs_et_train["time"].values)
    valid_periods_wo_warmup.append(obs_et_valid["time"].values)
    plot_ts(
        train_periods_wo_warmup,
        train_ets,
        leg_lst=leg_names,
        c_lst=["r", "g", "b", "k"],
        alpha=0.5,
        xlabel="Date",
        ylabel="ET(mm/day)",
        dash_lines=[False, False, False, True],
    )
    plt.savefig(
        os.path.join(
            fig_dir,
            ET_NAME + "_ts_" + basin_id + "_train_period.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plot_ts(
        valid_periods_wo_warmup,
        valid_ets,
        leg_lst=leg_names,
        c_lst=["r", "g", "b", "k"],
        alpha=0.5,
        xlabel="Date",
        ylabel="ET(mm/day)",
        dash_lines=[False, False, False, True],
    )
    plt.savefig(
        os.path.join(
            fig_dir,
            ET_NAME + "_ts_" + basin_id + "_valid_period.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )


def plot_metrics_1model_trained_with_diffperiods(
    all_basin_result_dirs,
    basin_ids,
    show_ind="NSE",
    cases=["3-year-train", "2-year-train", "1-year-train"],
    cfg_runagain=False,
    fig_dir=None,
    train_or_valid="valid",
    var_name="streamflow",
    legend=True,
):
    """_summary_

    Parameters
    ----------
    all_basin_result_dirs : list
        the exps are organized as follows:
            one basin is a list, and has several experiments, from 2-4 to 4-4
    basin_ids : list
        basin ids
    show_ind : str, optional
        the indicator to show, by default "NSE"
    cfg_runagain
        whether to run again, by default False, but for the first time, it should be True
    cases : list, optional
        the cases of train periods, by default ["3-year-train", "2-year-train", "1-year-train"]
    fig_dir : str, optional
        the directory to save the figure, by default None
    train_or_valid : str, optional
        whether to show the train or valid results, by default "valid"
    var_name : str, optional
        the variable name, by default "streamflow"
    """
    if fig_dir is None:
        fig_dir = all_basin_result_dirs[0][0]
    inds_df_dict = {}
    for i in range(len(all_basin_result_dirs)):
        _inds_df_dict = {}
        for j in range(len(all_basin_result_dirs[i])):
            inds_df_train_q, inds_df_valid_q, inds_df_train_et, inds_df_valid_et = (
                read_dpl_model_metric(
                    all_basin_result_dirs[i][j], cfg_runagain=cfg_runagain
                )
            )
            if train_or_valid == "valid" and var_name == "streamflow":
                _inds_df_dict[cases[j]] = inds_df_valid_q
            elif train_or_valid == "valid" and var_name == "et":
                _inds_df_dict[cases[j]] = inds_df_valid_et
            elif train_or_valid == "train" and var_name == "streamflow":
                _inds_df_dict[cases[j]] = inds_df_train_q
            elif train_or_valid == "train" and var_name == "et":
                _inds_df_dict[cases[j]] = inds_df_train_et
            else:
                raise ValueError("train_or_valid or var_name is wrong")
        inds_df_dict[basin_ids[i]] = _inds_df_dict
    df = pd.concat(
        {
            basin: pd.DataFrame(
                {train: stats[show_ind] for train, stats in data.items()}, index=[basin]
            )
            for basin, data in inds_df_dict.items()
        }
    )
    df.index = df.index.droplevel(0)
    # https://www.statology.org/change-font-size-matplotlib/
    plt.rc("axes", titlesize=18)  # Font size for the chart title
    plt.rc("axes", labelsize=16)  # Font size for the axis labels
    plt.rc("xtick", labelsize=16)  # Font size for the x-axis ticks
    plt.rc("ytick", labelsize=16)  # Font size for the y-axis ticks
    plt.rc("legend", fontsize=16)  # Font size for the legend
    plt.rc("legend", title_fontsize=16)  # Font size for the legend title

    plt.figure(figsize=(8, 6))

    # Iterate over each basin_id and plot their lines
    for basin_id in df.index:
        basin_name = ID_NAME_DICT.get(basin_id, basin_id)
        plt.plot(df.columns, df.loc[basin_id], marker="o", label=basin_name)

    if legend:
        plt.legend(loc="best")

    # Set axis labels and title (no need to specify fontsize here)
    plt.xlabel("Data Length")
    plt.ylabel(f"{show_ind}")
    # plt.title("Performance over different train periods")

    plt.grid(True)
    FIGURE_DPI = 600
    plt.savefig(
        os.path.join(fig_dir, "metric_of_1model_trained_with_diffperiods.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def _generate_dpl_result_dirs(
    basin_id, model="dpl", cases=["3-year-train", "2-year-train", "1-year-train"]
):
    """A literal specification of the result directories of dPL

    Parameters
    ----------
    basin_id : _type_
        _description_
    model : str, optional
        _description_, by default "dpl"

    Returns
    -------
    _type_
        _description_
    """
    if model == "dpl":
        post_fix = ""
    elif model == "dpl_nn":
        post_fix = "_module"
    if len(cases) != 3:
        raise ValueError(
            "cases must be a list with 3 elements, we support 2-4 to 4-4 now only"
        )
    return [
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "streamflow_prediction",
            "lrchange3" if model == "dpl" else "module",
            basin_id,
        ),
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "data-limited_analysis",
            "3to4_1518_1721" + post_fix,
            basin_id,
        ),
        os.path.join(
            RESULT_DIR,
            "dPL",
            "result",
            "data-limited_analysis",
            "2to4_1618_1721" + post_fix,
            basin_id,
        ),
    ]


# ----------- The following function is not finished yet ------------


def plot_stations_in_a_boxregion(
    data_map,
    pertile_range=[0, 100],
    fig_size=(10, 6),
    cmap_str="jet",
    vmin=None,
    vmax=None,
    shown_extent=[102.2, 109.2, 26.5, 32.5],
    background_river_shp=os.path.join(
        SETTING["local_data_path"]["basins-interim"], "shapes", "HydroRIVERS_v10.shp"
    ),
):
    """Plot a map of stations in a region within shown_extent

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
    sites = gpd.read_file(os.path.join(cc_shpfile_dir, "basinoutlets.shp"))
    sites_sorted = sites.sort_values(by="BASIN_ID")
    stid = sites_sorted.NAME.values
    lat = sites_sorted.geometry.y.values
    lon = sites_sorted.geometry.x.values

    # http://c.biancheng.net/matplotlib/9284.html
    # plt.rcParams["font.sans-serif"] = ["Ubuntu"]  # 设置字体 for Chinese
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": proj})
    # Show a region
    ax.set_extent(shown_extent)

    ax.add_feature(cartopy.feature.LAND, edgecolor="black")
    ax.add_feature(cartopy.feature.LAKES, edgecolor="black")

    gl = ax.gridlines(draw_labels=True)
    # https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    gl.xlabel_style = {"size": 19, "color": "gray"}
    gl.ylabel_style = {"size": 19, "color": "gray"}
    # Make figure larger
    plt.gcf().set_size_inches(fig_size)

    # Read shape file
    reader = shpreader.Reader(background_river_shp)
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
        cfg_dir_flow = os.path.join(CASE_DIR, exps[i])
        data_map = inds_df_lst[i]["NSE"].values
        # from mplfonts import use_font
        # use_font("Noto Serif CJK SC")
        f = plot_stations_in_a_boxregion(
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
    cfg_dir = os.path.join(CASE_DIR, exps[-1])
    plt.savefig(
        os.path.join(cfg_dir, fig_name),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )


def plot_computing_time(exps, leg_names):
    """plot computing time for exps
    But time is highly impacted by other factors such as the GPU and CPU usage from others in our server computer,
    so we don't use it now.

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
            cfg_flow = read_torchhydro_log_json_file(cfg_dir_flow)
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
    camels = SelfMadeHydroDataset(DATASET_DIR)
    lat_lon = camels.read_constant_cols(
        camels.camels_sites["gauge_id"].values, ["gauge_lat", "gauge_lon"]
    )
    cfg_dir_flow = os.path.join(RESULT_DIR, "camels", exps[0])
    cfg_flow = read_torchhydro_log_json_file(cfg_dir_flow)
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


if __name__ == "__main__":
    sceua_xaj_dir = os.path.join(RESULT_DIR, "XAJ", "changdian_61700_4_4")
    dpl_dir = os.path.join(RESULT_DIR, "dPL", "result", "lrchange3", "changdian_61700")
    dpl_nn_dir = os.path.join(RESULT_DIR, "dPL", "result", "module", "changdian_61700")
    basin_id = "changdian_61700"
    changdian_61700_name = "sanhuangmiao"
    basin_ids = [
        "changdian_61561",
        "changdian_61700",
        "changdian_61716",
        "changdian_62618",
        "changdian_91000",
    ]
    cases = ["3-year", "2-year", "1-year"]
    sanxiabasins_result_dirs = [
        _generate_dpl_result_dirs(basin_id, cases=cases) for basin_id in basin_ids
    ]
    plot_metrics_1model_trained_with_diffperiods(
        sanxiabasins_result_dirs,
        basin_ids,
        # when first time, cfg_runagain=True, then cfg_runagain=False
        cfg_runagain=False,
        cases=cases,
    )
    # plot_xaj_rainfall_runoff(
    #     [sceua_xaj_dir, dpl_dir, dpl_nn_dir], basin_id, changdian_61700_name
    # )
    # plot_xaj_et_time_series(
    #     [sceua_xaj_dir, dpl_dir, dpl_nn_dir], basin_id, changdian_61700_name
    # )
    # plot_losses_ts(
    #     [dpl_dir, dpl_nn_dir],
    #     leg_lst=[
    #         "dPL_train",
    #         "dPL$_{\mathrm{nn}}$_train",
    #         "dPL_valid",
    #         "dPL$_{\mathrm{nn}}$_valid",
    #     ],
    # )
    # plot_xaj_params_heatmap(
    #     [sceua_xaj_dir, dpl_dir, dpl_nn_dir],
    #     ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$"],
    #     basin_id + "_" + changdian_61700_name,
    #     param_test_way=[
    #         None,
    #         MODEL_PARAM_TEST_WAY["final_period"],
    #         MODEL_PARAM_TEST_WAY["final_period"],
    #     ],
    # )
