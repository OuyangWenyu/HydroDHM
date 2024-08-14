"""
Author: Wenyu Ouyang
Date: 2024-08-14 09:01:58
LastEditTime: 2024-08-14 09:50:21
LastEditors: Wenyu Ouyang
Description: Check streamflow data by plotting the rainfall-runoff relationship
FilePath: \HydroDHM\scripts\check_streamflow_data.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

from hydroutils.hydro_plot import plot_rainfall_runoff
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

from scripts import DATASET_DIR

datasource = SelfMadeHydroDataset(
    DATASET_DIR,
    download=False,
)
shpfile_dir = os.path.join(DATASET_DIR, "shapes")
basin_id_key = "BASIN_ID"
sites_ids = ["changdian_61561"]
t_range = ["2014-10-01", "2021-09-30"]
streamflows = datasource.read_ts_xrdataset(
    gage_id_lst=sites_ids, t_range=t_range, var_lst=["streamflow"]
)
prcps = datasource.read_ts_xrdataset(sites_ids, t_range, ["total_precipitation_hourly"])
periods = pd.date_range(start=t_range[0], end=t_range[-1], freq="1D")
for i in range(18):
    plot_rainfall_runoff(
        np.tile(periods, (1, 1)).tolist(),
        prcps.values()[i, :, 0],
        [streamflows.values()[i]],
        leg_lst=["OBS"],
        title=sites_ids[i],
        fig_size=(12, 6),
        xlabel="date",
        ylabel="streamflow (mm/d)",
        linewidth=1,
        c_lst="rgb",
        dash_lines=[False, False, True],
    )
