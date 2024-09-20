"""
查看指定流域和时间范围的降雨径流数据
"""

import xarray as xr
import matplotlib.pyplot as plt
import os
import pandas as pd
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

DATASET_DIR = '/ftproot/basins-interim'
datasource = SelfMadeHydroDataset(
    DATASET_DIR,
    download=False,
)

# 指定输出文件夹路径
output_folder = 'results/obs_precipitaton_streamflow/changdian'
# 如果文件夹不存在，创建文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有的 basin 列表
sites_ids = [
        "changdian_61561",
        "changdian_61700",
        "changdian_61716",
        "changdian_62618",
        "changdian_91000",

        "changdian_60650",
        "changdian_60668",
        "changdian_61239",
        "changdian_61277",
        "changdian_62018",
        "changdian_62315",
        "changdian_63002",
        "changdian_63007",
        "changdian_63458",
        "changdian_63486",
        "changdian_63490",
        "changdian_90813",
        "changdian_91700",
        "changdian_92114",
        "changdian_92116",
        "changdian_92118",
        "changdian_92119",
        "changdian_92146",
        "changdian_92353",
        "changdian_92354",
        "changdian_94470",
        "changdian_94560",
        "changdian_94850",
        "changdian_95350",


        # "camels_01539000",
        # "camels_02231000",
        # "camels_03161000",
        # "camels_03300400",
        # "camels_07261000",

        # "camels_11532500",
        # "camels_12025000",
        # "camels_12035000",
        # "camels_12145500",
        # "camels_14301000",
        # "camels_14306500",
        # "camels_14325000",


        # "camels_01440000",
        # "camels_01440400",
        # "camels_01532000",
        # "camels_01552000",
        # "camels_02070000",
        # "camels_02137727",
        # "camels_02140991",
        # "camels_02177000",
        # "camels_02212600",
        # "camels_02246000",
        # "camels_02427250",
        # "camels_03500000",


        # "camels_03346000",
        # "camels_05501000",
        # "camels_05514500",
        # "camels_07057500",
        # "camels_07066000",
        # "camels_07145700",
        # "camels_07263295",
        # "camels_07359610",
        

        # "anhui_62909400",
        # "songliao_10911000",
        # "songliao_10912404",
        # "songliao_11002210",
        # "songliao_11400900",
        # "songliao_11606000",
        # "songliao_21110150",
        # "songliao_21110400",
        # "songliao_21113800",
        # "songliao_21200100",
        # "songliao_21300500",
        # "songliao_21401050",
        # "songliao_21401300",
    ]
t_range = ["2014-10-01", "2021-09-30"]

# 读取所有流域的时间序列数据
dataset = datasource.read_ts_xrdataset(
    gage_id_lst=sites_ids, t_range=t_range, var_lst=["streamflow", "total_precipitation_hourly"]
)['1D']

# 指定完整的时间范围
full_time_index = pd.date_range(start=t_range[0], end=t_range[1], freq='D')

for basin in sites_ids:
    # 选择当前 basin 的数据
    ds_basin = dataset.sel(basin=basin)

    streamflow_data = ds_basin['streamflow'].to_pandas()
    precipitation_data = ds_basin['total_precipitation_hourly'].to_pandas()

    # 重新索引以包含完整的时间范围（保留 NaN 以在图表上显示空白）
    streamflow_data = streamflow_data.reindex(full_time_index)
    precipitation_data = precipitation_data.reindex(full_time_index)

    # 计算缺失率
    total_days = streamflow_data.shape[0]
    missing_days = streamflow_data.isna().sum()
    missing_rate = missing_days / total_days

    # 创建一个图形和两个 y 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))

    # 添加图表标题
    fig.suptitle(f"Observation of Precipitation and Streamflow for {basin}", fontsize=16)

    # 绘制 total_precipitation_hourly 为柱状图，设置 y 轴逆置
    ax1.bar(full_time_index, precipitation_data, color='b', label='Total Precipitation Hourly', width=0.9)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Precipitation (mm/d)', color='b')
    ax1.invert_yaxis()  # y 轴逆置 
    ax1.tick_params(axis='y', labelcolor='b')

    # 直接绘制完整时间轴的 streamflow（NaN 部分自动留空）
    ax2.plot(full_time_index, streamflow_data, 'r-', label='Streamflow')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Streamflow (mm/d)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 在流量图上添加缺失率文本
    ax2.text(0.05, 0.95, f'Missing Rate: {missing_rate:.2%}', transform=ax2.transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # 调整布局
    fig.tight_layout()

    # 保存图像到指定文件夹
    file_path = os.path.join(output_folder, f'{basin}_obs_streamflow_precipitation.png')
    plt.savefig(file_path)

    # 关闭图形以释放内存
    plt.close(fig)

    print(f"Saved figure for basin {basin} to {file_path}")