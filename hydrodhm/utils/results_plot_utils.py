import os
import sys
import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.io.shapereader as shpreader

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

sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import DATASET_DIR, RESULT_DIR
from hydrodhm.utils.results_utils import (
    get_json_file,
    get_latest_pbm_param_file,
    get_pbm_params_from_dpl,
    get_pbm_params_from_hydromodelxaj,
)


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
                raise ValueError(
                    "No parameter file found; Please run get_pbm_params_from_dpl function first"
                )
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
    camels = SelfMadeHydroDataset(
        os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    )
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
    camels_cc = SelfMadeHydroDataset(
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
        xlabel="date",
        ylabel="streamflow (m$^3$/s)",
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


if __name__ == "__main__":
    sceua_xaj_dir = os.path.join(RESULT_DIR, "XAJ", "changdian_61700_4_4")
    dpl_dir = os.path.join(RESULT_DIR, "dPL", "result", "lrchange3", "changdian_61700")
    dpl_nn_dir = os.path.join(RESULT_DIR, "dPL", "result", "module", "changdian_61700")
    changdian_61700_name = "changdian_61700_sanhuangmiao"
    plot_xaj_params_heatmap(
        [sceua_xaj_dir, dpl_dir, dpl_nn_dir],
        ["eXAJ", "dXAJ", "dXAJ$_{\mathrm{nn}}$"],
        changdian_61700_name,
        param_test_way=[
            None,
            MODEL_PARAM_TEST_WAY["final_period"],
            MODEL_PARAM_TEST_WAY["final_period"],
        ],
    )
