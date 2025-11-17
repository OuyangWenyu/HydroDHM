from hydroutils.hydro_stat import stat_error
from hydroutils.hydro_plot import plot_rainfall_runoff
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torchhydro import SETTING
from definitions import RESULT_DIR
from hydrodataset.camels import Camels

import xarray as xr
import os
import pickle

camels = Camels(SETTING["local_data_path"]["datasets-origin"])
result_dir = RESULT_DIR


def _trans_m3_to_mmday(gage_id, streamflow_m3s):
    area = camels.read_area(gage_id)["area_gages2"].values.item()  # km2
    # Convert m3/s to mm/day
    streamflow_mmday = (streamflow_m3s * 86400) / (area * 1e6) * 1000
    streamflow_mmday.attrs["units"] = "mm/day"
    return streamflow_mmday


def read_obs_streamflow(model, gage_id, t_range, read_all=False, **kwargs):
    if model == "xaj":
        dir = os.path.join(
            result_dir, "xaj_results", gage_id, "xaj_SCE_UA", "evaluation_test"
        )
        streamflow = xr.open_dataset(os.path.join(dir, "xaj_evaluation_results.nc"))[
            "qobs"
        ].sel(time=slice(t_range[0], t_range[1]))
        streamflow = _trans_m3_to_mmday(gage_id, streamflow)
    elif model == "neuralhydrology":
        if read_all:
            dir = os.path.join(result_dir, "neuralhydrology_results", "camels_all")
        else:
            dir = os.path.join(result_dir, "neuralhydrology_results", gage_id)
        mode = kwargs.get("mode", "test")
        with open(
            os.path.join(dir, f"{mode}_results.p"),
            "rb",
        ) as f:
            data = pickle.load(f)
        streamflow = data[gage_id]["1D"]["xr"]["QObs(mm/d)_obs"].sel(
            date=slice(t_range[0], t_range[1])
        )
        streamflow = streamflow.rename({"date": "time"})
        streamflow = streamflow.squeeze("time_step", drop=True)
        streamflow = streamflow.rename("streamflow")
    else:
        if read_all:
            dir = os.path.join(result_dir, f"{model}_results", "camels_all")
        else:
            dir = os.path.join(result_dir, f"{model}_results", gage_id)
        streamflow = xr.open_dataset(
            os.path.join(dir, "epochbest_model.pthflow_obs.nc")
        )["streamflow"].sel(time=slice(t_range[0], t_range[1]))
    if isinstance(streamflow, xr.DataArray):
        streamflow.name = "streamflow"
        streamflow = streamflow.to_dataset()
    return streamflow


def read_pred_streamflow(model, gage_id, t_range, read_all=False, **kwargs):
    if model == "xaj":
        dir = os.path.join(
            result_dir, "xaj_results", gage_id, "xaj_SCE_UA", "evaluation_test"
        )
        streamflow = xr.open_dataset(os.path.join(dir, "xaj_evaluation_results.nc"))[
            "qsim"
        ].sel(time=slice(t_range[0], t_range[1]))
        streamflow = _trans_m3_to_mmday(gage_id, streamflow)
    elif model == "neuralhydrology":
        if read_all:
            dir = os.path.join(result_dir, "neuralhydrology_results", "camels_all")
        else:
            dir = os.path.join(result_dir, "neuralhydrology_results", gage_id)
        mode = kwargs.get("mode", "test")
        with open(
            os.path.join(dir, f"{mode}_results.p"),
            "rb",
        ) as f:
            data = pickle.load(f)
        streamflow = data[gage_id]["1D"]["xr"]["QObs(mm/d)_sim"].sel(
            date=slice(t_range[0], t_range[1])
        )
        streamflow = streamflow.rename({"date": "time"})
        streamflow = streamflow.squeeze("time_step", drop=True)
        streamflow = streamflow.rename("streamflow")
    else:
        if read_all:
            dir = os.path.join(result_dir, f"{model}_results", "camels_all")
        else:
            dir = os.path.join(result_dir, f"{model}_results", gage_id)
        streamflow = xr.open_dataset(
            os.path.join(dir, "epochbest_model.pthflow_pred.nc")
        )["streamflow"].sel(time=slice(t_range[0], t_range[1]))
    if isinstance(streamflow, xr.DataArray):
        streamflow.name = "streamflow"
        streamflow = streamflow.to_dataset()
    return streamflow


def read_obs_et(gage_id, t_range):
    return camels.read_ts_xrdataset(
        gage_id_lst=[gage_id], t_range=t_range, var_lst=["ET"]
    )


def read_obs_pet(gage_id, t_range):
    return camels.read_ts_xrdataset(
        gage_id_lst=[gage_id], t_range=t_range, var_lst=["PET"]
    )


def read_pred_et(model, gage_id, t_range, **kwargs):
    if model == "xaj":
        # raise NotImplementedError("XAJ model output does not have ET for now.")
        mode = kwargs.get("mode", "test")
        dir = os.path.join(
            result_dir, "xaj_results", gage_id, "xaj_SCE_UA", f"evaluation_{mode}"
        )
        et = xr.open_dataset(os.path.join(dir, "xaj_evaluation_results.nc"))["es"].sel(
            time=slice(t_range[0], t_range[1])
        )
    else:
        dir = os.path.join(result_dir, f"{model}_results", gage_id)
        et = xr.open_dataset(os.path.join(dir, "epochbest_model.pthflow_pred.nc"))[
            "ET"
        ].sel(time=slice(t_range[0], t_range[1]))
    if isinstance(et, xr.DataArray):
        et.name = "ET"
        et = et.to_dataset()
    return et


if __name__ == "__main__":
    # Example usage
    # exp1
    obs = read_obs_et("12025000", ("1982-01-01", "2009-12-31"))
    sim = read_pred_et("xaj", "12025000", ("1982-01-01", "2009-12-31"), mode="train")
    obs = obs["ET"].values.squeeze()
    sim = sim["ET"].values.squeeze()
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
        sim = sim.reshape(1, -1)
    metrics = stat_error(obs, sim)
    print(metrics)

    # exp2
    # # gage_ids = camels.read_site_info()["gauge_id"].values.tolist()
    # gage_id = "12025000"
    # metrics_all = {}
    # # for gage_id in gage_ids:
    # obs = read_obs_streamflow(
    #     "neuralhydrology",
    #     gage_id,
    #     ("1980-01-01", "2004-12-31"),
    #     # read_all=True,
    #     mode="train",
    # )
    # sim = read_pred_streamflow(
    #     "neuralhydrology",
    #     gage_id,
    #     ("1980-01-01", "2004-12-31"),
    #     # read_all=True,
    #     mode="train",
    # )
    # obs = obs["streamflow"].values.squeeze()
    # sim = sim["streamflow"].values.squeeze()
    # if obs.ndim == 1:
    #     obs = obs.reshape(1, -1)
    #     sim = sim.reshape(1, -1)
    # metrics = stat_error(obs, sim)
    # metrics_scalar = {
    #     k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()
    # }
    # metrics_all[gage_id] = metrics
    # metrics_all = pd.DataFrame(metrics_all).T
    # metrics_all.to_csv(
    #     os.path.join(
    #         result_dir, "neuralhydrology_results", gage_id, "train_metrics_all.csv"
    #     )
    # )
