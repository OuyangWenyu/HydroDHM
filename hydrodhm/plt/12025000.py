from hydroutils.hydro_plot import plot_rainfall_runoff
from matplotlib import pyplot as plt
import numpy as np
from torchhydro import SETTING
from definitions import RESULT_DIR
from hydrodataset.camels import Camels

import xarray as xr
import os

camels = Camels(SETTING["local_data_path"]["datasets-origin"])
result_dir = RESULT_DIR


def _trans_m3_to_mmday(gage_id, streamflow_m3s):
    area = camels.read_area(gage_id)["area_gages2"].values.item()  # km2
    # Convert m3/s to mm/day
    streamflow_mmday = (streamflow_m3s * 86400) / (area * 1e6) * 1000
    streamflow_mmday.attrs["units"] = "mm/day"
    return streamflow_mmday


def read_tp(gage_id, t_range):
    return camels.read_ts_xrdataset(
        gage_id_lst=[gage_id], t_range=t_range, var_lst=["prcp"]
    )


def read_obs(model, gage_id, t_range):
    if model == "xaj":
        dir = os.path.join(
            result_dir, "xaj_results", gage_id, "xaj_SCE_UA", "evaluation_test"
        )
        streamflow = xr.open_dataset(os.path.join(dir, "xaj_evaluation_results.nc"))[
            "qobs"
        ].sel(time=slice(t_range[0], t_range[1]))
        streamflow = _trans_m3_to_mmday(gage_id, streamflow)
    else:
        dir = os.path.join(result_dir, f"{model}_results", gage_id)
        streamflow = xr.open_dataset(
            os.path.join(dir, "epochbest_model.pthflow_obs.nc")
        )["streamflow"].sel(time=slice(t_range[0], t_range[1]))
    if isinstance(streamflow, xr.DataArray):
        streamflow.name = "streamflow"
        streamflow = streamflow.to_dataset()
    return streamflow


def read_pred(model, gage_id, t_range):
    if model == "xaj":
        dir = os.path.join(
            result_dir, "xaj_results", gage_id, "xaj_SCE_UA", "evaluation_test"
        )
        streamflow = xr.open_dataset(os.path.join(dir, "xaj_evaluation_results.nc"))[
            "qsim"
        ].sel(time=slice(t_range[0], t_range[1]))
        streamflow = _trans_m3_to_mmday(gage_id, streamflow)
    else:
        dir = os.path.join(result_dir, f"{model}_results", gage_id)
        streamflow = xr.open_dataset(
            os.path.join(dir, "epochbest_model.pthflow_pred.nc")
        )["streamflow"].sel(time=slice(t_range[0], t_range[1]))
    if isinstance(streamflow, xr.DataArray):
        streamflow.name = "streamflow"
        streamflow = streamflow.to_dataset()
    return streamflow


if __name__ == "__main__":
    gage_id = "12025000"
    time_period = ["2012-01-01", "2012-03-31"]
    tp = read_tp(gage_id, time_period)
    obs = read_obs("xaj", gage_id, time_period)
    xaj_pred = read_pred("xaj", gage_id, time_period)
    # lstm_pred = read_pred("lstm", gage_id, ("2011-01-01", "2014-12-31"))
    dplxaj_pred = read_pred("dplxaj", gage_id, time_period)
    dplnnxaj_pred = read_pred("dplnnxaj", gage_id, time_period)

    t = tp.time.values

    # get tp
    p = tp["prcp"].values.squeeze()

    # get streamflow
    obs_q = obs["streamflow"].values.squeeze()
    xaj_pred_q = xaj_pred["streamflow"].values.squeeze()
    # lstm_pred_q = lstm_pred["streamflow"].values.squeeze()
    dplxaj_pred_q = dplxaj_pred["streamflow"].values.squeeze()
    dplnnxaj_pred_q = dplnnxaj_pred["streamflow"].values.squeeze()

    # plot
    fig, ax = plot_rainfall_runoff(
        t=t,
        p=p,
        qs=[
            obs_q,
            xaj_pred_q,
            # lstm_pred_q,
            dplxaj_pred_q,
            dplnnxaj_pred_q,
        ],
        fig_size=(12, 6),
        leg_lst=["Observed", "XAJ Model", "DPLXAJ Model", "DPLNNXAJ Model"],
        title=f"Basin {gage_id} - Rainfall-Runoff",
        xlabel="Time",
        ylabel="Streamflow (mm/day)",
        prcp_ylabel="Precipitation (mm/day)",
        linewidth=1.5,
        prcp_interval=20,
    )

    plt.tight_layout()
    plt_result_dir = os.path.join(result_dir, "figures")
    os.makedirs(plt_result_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            plt_result_dir,
            f"rainfall_runoff_{gage_id}_{time_period[0]}_{time_period[1]}.png",
        ),
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
