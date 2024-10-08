"""
Author: Wenyu Ouyang
Date: 2024-09-24 18:57:11
LastEditTime: 2024-09-28 16:59:25
LastEditors: Wenyu Ouyang
Description: 
FilePath: \HydroDHM\hydrodhm\calculate_and_plot\calculate_cv_metric.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import sys
from pathlib import Path

import pandas as pd


sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent.parent))
from definitions import (
    CHANGDIAN_DPL_PARENT_DIR,
    CHANGDIAN_IDS,
    CHANGDIAN_SCEUA_RESULT_DIRS,
    SANXIA_SCEUA_DIR1,
)
from hydrodhm.utils.results_utils import (
    read_sceua_xaj_et_metric,
    read_sceua_xaj_streamflow_metric,
)


def calculate_dpl_mean_metrics(
    dir_, dir_reverse, basin_id, metric_names, trainperiod=False
):
    """
    Calculate the coefficient of variation (CV) for the metrics of the streamflow prediction.
    """
    the_dir = f"{basin_id}_trainperiod" if trainperiod else f"{basin_id}"
    basin_dir = os.path.join(dir_, the_dir)
    basin_dir_reverse = os.path.join(dir_reverse, the_dir)
    metric_streamflow_file = os.path.join(basin_dir, "metric_streamflow.csv")
    metric_streamflow_reverse_file = os.path.join(
        basin_dir_reverse, "metric_streamflow.csv"
    )
    metric_streamflow = pd.read_csv(metric_streamflow_file)
    metric_streamflow_reverse = pd.read_csv(metric_streamflow_reverse_file)
    metric_et_file = os.path.join(basin_dir, "metric_total_evaporation_hourly.csv")
    metric_et_reverse_file = os.path.join(
        basin_dir_reverse, "metric_total_evaporation_hourly.csv"
    )
    metric_et = pd.read_csv(metric_et_file)
    metric_et_reverse = pd.read_csv(metric_et_reverse_file)
    metrics_streamflow = {}
    metrics_et = {}
    for metric_name in metric_names:
        metrics_streamflow[metric_name] = (
            (metric_streamflow[metric_name] + metric_streamflow_reverse[metric_name])
            / 2
        ).values[0]
        metrics_et[metric_name] = (
            (metric_et[metric_name] + metric_et_reverse[metric_name]) / 2
        ).values[0]
    return metrics_streamflow, metrics_et


def one_model_all_basins_train_valid_ensemble_mean(model="dpl"):
    """
    Calculate the mean of the metrics of the streamflow prediction for all basins.
    """
    ids_streamflow_metrics = {}
    ids_et_metrics = {}
    ids_streamflow_metrics_trainperiod = {}
    ids_et_metrics_trainperiod = {}
    if model == "dpl":
        dir0 = CHANGDIAN_DPL_PARENT_DIR[0]
        dir1 = CHANGDIAN_DPL_PARENT_DIR[1]
    elif model == "dpl_nn":
        dir0 = CHANGDIAN_DPL_PARENT_DIR[2]
        dir1 = CHANGDIAN_DPL_PARENT_DIR[3]
    elif model == "sceua":
        dir0 = SANXIA_SCEUA_DIR1
    for basin_id in CHANGDIAN_IDS:
        if model == "sceua":
            metrics_streamflow, metrics_et = calculate_sceua_mean_metrics(
                basin_id,
                ["RMSE", "Corr", "NSE", "KGE", "FHV", "FLV"],
            )
            metrics_streamflow_trainperiod, metrics_et_trainperiod = (
                calculate_sceua_mean_metrics(
                    basin_id,
                    ["RMSE", "Corr", "NSE", "KGE", "FHV", "FLV"],
                    trainperiod=True,
                )
            )
        else:
            metrics_streamflow, metrics_et = calculate_dpl_mean_metrics(
                dir0,
                dir1,
                basin_id,
                ["RMSE", "Corr", "NSE", "KGE", "FHV", "FLV"],
            )
            metrics_streamflow_trainperiod, metrics_et_trainperiod = (
                calculate_dpl_mean_metrics(
                    dir0,
                    dir1,
                    basin_id,
                    ["RMSE", "Corr", "NSE", "KGE", "FHV", "FLV"],
                    trainperiod=True,
                )
            )
        ids_streamflow_metrics[basin_id] = metrics_streamflow
        ids_et_metrics[basin_id] = metrics_et
        ids_streamflow_metrics_trainperiod[basin_id] = metrics_streamflow_trainperiod
        ids_et_metrics_trainperiod[basin_id] = metrics_et_trainperiod
    df_streamflow_metrics = pd.DataFrame(ids_streamflow_metrics).T
    df_et_metrics = pd.DataFrame(ids_et_metrics).T
    df_streamflow_metrics_trainperiod = pd.DataFrame(
        ids_streamflow_metrics_trainperiod
    ).T
    df_et_metrics_trainperiod = pd.DataFrame(ids_et_metrics_trainperiod).T
    print("Streamflow metrics:")
    print(df_streamflow_metrics)
    print("ET metrics:")
    print(df_et_metrics)
    print("Streamflow metrics train period:")
    print(df_streamflow_metrics_trainperiod)
    print("ET metrics train period:")
    print(df_et_metrics_trainperiod)


def calculate_sceua_mean_metrics(basin_id, metric_names, trainperiod=False):
    """
    Calculate the mean of the metrics of the streamflow prediction for all basins.
    """
    sceua_dir = SANXIA_SCEUA_DIR1
    ids_streamflow_metrics = {}
    ids_et_metrics = {}
    a_sceua_dir = os.path.join(sceua_dir, basin_id + "_4_4")
    inds_df_train_et, inds_df_valid_et = read_sceua_xaj_et_metric(a_sceua_dir)
    a_sceua_dir_re = os.path.join(sceua_dir, f"{basin_id}_4_4_re")
    inds_df_train_re_et, inds_df_valid_re_et = read_sceua_xaj_et_metric(a_sceua_dir_re)
    inds_df_train_streamflow, inds_df_valid_streamflow = (
        read_sceua_xaj_streamflow_metric(a_sceua_dir)
    )
    inds_df_train_re_streamflow, inds_df_valid_re_streamflow = (
        read_sceua_xaj_streamflow_metric(a_sceua_dir_re)
    )
    for metric_name in metric_names:
        if trainperiod:
            ids_et_metrics[metric_name] = (
                (inds_df_train_et[metric_name] + inds_df_train_re_et[metric_name]) / 2
            ).values[0]
            ids_streamflow_metrics[metric_name] = (
                (
                    inds_df_train_streamflow[metric_name]
                    + inds_df_train_re_streamflow[metric_name]
                )
                / 2
            ).values[0]
        else:
            ids_et_metrics[metric_name] = (
                (inds_df_valid_et[metric_name] + inds_df_valid_re_et[metric_name]) / 2
            ).values[0]
            ids_streamflow_metrics[metric_name] = (
                (
                    inds_df_valid_streamflow[metric_name]
                    + inds_df_valid_re_streamflow[metric_name]
                )
                / 2
            ).values[0]
    return ids_streamflow_metrics, ids_et_metrics


one_model_all_basins_train_valid_ensemble_mean(model="sceua")
# one_model_all_basins_train_valid_ensemble_mean(model="dpl")
# one_model_all_basins_train_valid_ensemble_mean(model="dpl_nn")
