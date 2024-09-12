import xarray as xr
import matplotlib.pyplot as plt
import os
from hydrodatasource.reader.data_source import SelfMadeHydroDataset

DATASET_DIR = '/ftproot/basins-interim'
datasource = SelfMadeHydroDataset(
    DATASET_DIR,
    download=False,
)

# 指定输出文件夹路径
output_folder = 'results_old/Rainfall_obs/changdian'
# 如果文件夹不存在，创建文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有的 basin 列表
sites_ids = [
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
    # "changdian_61561",
    # "changdian_62618",
    # "changdian_60650",
    # "changdian_61716",
    # "changdian_62018",
    # "changdian_62315",
    # "changdian_91000",
    # "changdian_91700",
    # "changdian_92114",
    # "changdian_92353",
    # "changdian_95350",
    # "anhui_62909400",
    
    "changdian_60668",
    "changdian_61239",
    "changdian_61277",
    "changdian_61700",
    "changdian_63002",
    "changdian_63007",
    "changdian_63458",
    "changdian_63486",
    "changdian_63490",
    "changdian_90813",
    "changdian_92116",
    "changdian_92118",
    "changdian_92119",
    "changdian_92146",
    "changdian_92354",
    "changdian_94470",
    "changdian_94560",
    "changdian_94850",    
    ]
t_range = ["2014-10-01", "2021-10-01"]

dataset = datasource.read_ts_xrdataset(
    gage_id_lst=sites_ids, t_range=t_range, var_lst=["streamflow", "total_precipitation_hourly"]
)['1D']

# 为每个 basin 绘制图像并保存
for basin in sites_ids:
    # 选择当前 basin 的数据
    ds_basin = dataset.sel(basin=basin)

    # 创建一个图形和两个 y 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8)) 

    # 添加图表标题
    fig.suptitle(f"Observation of Precipitation and Streamflow for {basin}", fontsize=16)

    # 绘制 total_precipitation_hourly，设置 y 轴逆置
    ax1.plot(ds_basin['time'], ds_basin['total_precipitation_hourly'], 'b-', label='Total Precipitation Hourly')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Precipitation (mm/d)', color='b')
    ax1.invert_yaxis()  # y 轴逆置
    ax1.tick_params(axis='y', labelcolor='b')

    # 绘制 streamflow
    ax2.plot(ds_basin['time'], ds_basin['streamflow'], 'r-', label='Streamflow')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Streamflow (mm/d)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 调整布局
    fig.tight_layout()

    # 保存图像到指定文件夹
    file_path = os.path.join(output_folder, f'basin_{basin}_streamflow_precipitation.png')
    plt.savefig(file_path)

    # 关闭图形以释放内存
    plt.close(fig)

    print(f"Saved figure for basin {basin} to {file_path}")