<!--
 * @Author: Wenyu Ouyang
 * @Date: 2025-10-29
 * @LastEditTime: 2025-10-29
 * @LastEditors: Wenyu Ouyang
 * @Description: 中文版 README
 * @FilePath: \HydroDHM\README_zh.md
 * Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
-->

# 数据稀缺流域的可微分水文模型

[English](README.md) | 简体中文

本仓库包含论文《A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins》（[Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2024.132471)）的代码。代码基于 [torchhydro](https://github.com/OuyangWenyu/torchhydro) 和 [hydromodel](https://github.com/OuyangWenyu/hydromodel) 包构建。前者是基于 PyTorch 的水文建模框架，后者使用 Numpy 实现了包括新安江模型在内的传统水文模型。

使用这些模型进行的实验可以在 `run_xaj`、`streamflow_prediction` 和 `data-limited_analysis` 目录中找到。`calculate_and_plot` 目录包含用于生成结果和图表的脚本。

该模型在 CAMELS 数据集以及位于三峡库区上游、四川省境内或附近的几个流域上进行了训练和评估。

对于 CAMELS 数据集，我们将在未来提供数据。然而，三峡库区的数据共享可能受到某些政策限制。

## 目录

- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [配置设置](#配置设置)
- [运行 XAJ 率定](#运行-xaj-率定)
- [模型评估](#模型评估)
- [项目结构](#项目结构)

## 环境配置

本项目现在使用 `uv` 进行更快、更可靠的依赖管理，替代了之前的 conda 环境。

### 1. 安装 uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建虚拟环境

导航到项目目录并创建虚拟环境：

```bash
cd HydroDHM
uv venv
```

### 3. 激活虚拟环境

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. 安装依赖

```bash
uv sync
```

这将安装所有必需的依赖项，包括支持 CUDA 11.8 的 PyTorch。

**注意：** 如果需要不同的 CUDA 版本，请在运行 `uv sync` 之前修改 `pyproject.toml` 文件。详见 [UV_SETUP.md](UV_SETUP.md)。

## 数据准备

### 使用 CAMELS 数据集

本项目使用 `hydrodataset` 包来访问水文数据集。该包使用 [AquaFetch](https://github.com/hyex-research/AquaFetch) 后端自动下载和缓存数据。

#### 1. 安装 hydrodataset

该包应该已经作为依赖项的一部分安装。如果没有：

```bash
uv pip install hydrodataset
```

#### 2. 配置数据路径

确保您的 `hydro_setting.yml`（在主目录中）已配置数据路径：

```yaml
local_data_path:
  datasets-origin: 'D:\\data'  # 更新为您的路径
  cache: 'D:\\data\\.cache'
```

#### 3. 下载 CAMELS 数据

**方式 A: 使用下载脚本（推荐新手使用）**

我们在 `hydrodhm/data_tools/` 中提供了一个便捷的下载脚本，方便数据集管理：

```bash
cd hydrodhm/data_tools

# 列出所有可用的 CAMELS 数据集
python download_camels.py --list

# 下载 CAMELS-US（使用 hydro_setting.yml 中的路径）
python download_camels.py camels_us

# 或指定自定义路径
python download_camels.py camels_us --data-path D:/data/camels

# 下载其他数据集
python download_camels.py camels_gb
python download_camels.py camels_aus
```

脚本功能：
- 通过 hydrodataset 的 AquaFetch 后端自动下载数据
- 转换为标准化的 NetCDF 格式
- 通过显示流域数量验证下载
- 提供下一步操作的有用指导

详细说明请参见 `hydrodhm/data_tools/DOWNLOAD_GUIDE.md`。

**方式 B: 直接使用 Python**

`hydrodataset` 包会自动处理数据下载。只需初始化数据集类，数据将在首次访问时自动下载和缓存：

```python
from hydrodataset.camels_us import CamelsUs
from hydrodataset import SETTING

# 从 hydro_setting.yml 获取数据路径
data_path = SETTING["local_data_path"]["datasets-origin"]

# 初始化数据集 - 如果不存在会自动下载
# 这将通过 AquaFetch 获取原始数据并缓存为 .nc 文件
ds = CamelsUs(data_path)

# 访问流域 ID
basin_ids = ds.read_object_ids()
print(f"找到 {len(basin_ids)} 个流域")

# 读取时间序列数据（流量、降水等）
ts_data = ds.read_ts_xrdataset(
    gage_id_lst=basin_ids[:2],
    t_range=["1990-01-01", "1995-12-31"],
    var_lst=["streamflow", "precipitation"]
)

# 读取静态属性
attr_data = ds.read_attr_xrdataset(
    gage_id_lst=basin_ids[:2],
    var_lst=["area", "p_mean"]
)
```

**工作原理：**
1. **首次访问**：首次初始化数据集类时，`hydrodataset` 使用 AquaFetch 下载原始数据
2. **自动缓存**：数据被处理为标准化的 NetCDF (`.nc`) 格式并缓存在您配置的数据目录中
3. **快速后续访问**：所有未来的数据请求直接从快速的 `.nc` 缓存文件读取

**重要提示：**
- 首次下载可能需要 30 分钟到几个小时，取决于数据集大小
- CAMELS-US 数据集约 10-20 GB
- 请确保有足够的磁盘空间
- 建议初次下载时使用稳定的网络连接
- 缓存的 `.nc` 文件存储在 `{data_path}/{dataset_name}/` 目录中

#### 4. 可用的 CAMELS 数据集

`hydrodataset` 包支持多个 CAMELS 数据集：

- `camels_us` - 美国 (671 个流域)
- `camels_aus` - 澳大利亚 (222 个流域)
- `camels_gb` - 英国 (671 个流域)
- `camels_br` - 巴西 (897 个流域)
- `camels_ch` - 瑞士 (331 个流域)
- `camels_cl` - 智利 (516 个流域)
- `camels_de` - 德国 (1555 个流域)
- `camels_dk` - 丹麦 (304 个流域)
- `camels_fr` - 法国 (662 个流域)
- `camels_nz` - 新西兰 (343 个流域)
- `camels_se` - 瑞典 (54 个流域)

要使用不同的数据集，只需导入并初始化相应的类：

```python
from hydrodataset.camels_gb import CamelsGb
from hydrodataset.camels_aus import CamelsAus

# 使用 CAMELS-GB
gb_ds = CamelsGb(data_path)

# 使用 CAMELS-AUS
aus_ds = CamelsAus(data_path)
```

**标准化变量名：**

所有 CAMELS 数据集都使用标准化的变量名以保持一致性：
- `streamflow` - 观测流量
- `precipitation` - 降水数据
- `temperature_max` / `temperature_min` - 温度数据
- `area` - 流域面积
- `p_mean` - 年平均降水量
- 还有更多...

这使您可以在不同的 CAMELS 数据集中使用相同的代码，无需修改。

有关更多详细信息，请参阅 [hydrodataset 文档](https://OuyangWenyu.github.io/hydrodataset)。

### 使用自己的数据

如果您有自己的水文数据，请按照 `selfmadehydrodataset` 格式组织。所需的数据结构请参见 [hydromodel 文档](https://github.com/OuyangWenyu/hydromodel)。

## 配置设置

### 设置 hydro_setting.yml

`hydro_setting.yml` 文件配置 hydromodel 包的数据路径和其他全局设置。

#### 1. 位置

在您的**主目录**（与用户文件夹同一级别）创建此文件：

**Windows:**
```
C:\Users\YourUsername\hydro_setting.yml
```

**macOS/Linux:**
```
~/hydro_setting.yml
```

或者，您可以将其放在项目目录中，并设置 `HYDRO_SETTING_FILE` 环境变量指向它。

#### 2. 创建配置文件

**Windows:**
```powershell
# 将示例文件复制到主目录
cp hydro_setting.example.yml $env:USERPROFILE\hydro_setting.yml

# 编辑文件
notepad $env:USERPROFILE\hydro_setting.yml
```

**macOS/Linux:**
```bash
# 将示例文件复制到主目录
cp hydro_setting.example.yml ~/hydro_setting.yml

# 编辑文件
nano ~/hydro_setting.yml
```

#### 3. 配置内容

使用您的数据路径编辑 `hydro_setting.yml` 文件。文件结构如下：

```yaml
# 本地数据路径（必需）
local_data_path:
  # 所有数据的根目录
  root: 'D:\\data'

  # 原始数据集目录
  datasets-origin: 'D:\\data'

  # 中间/处理后的数据集目录
  datasets-interim: 'D:\\data'

  # 原始流域数据目录
  basins-origin: 'D:\\data'

  # 中间/处理后的流域数据目录
  basins-interim: 'D:\\data'

  # 缓存目录
  cache: 'C:\\Users\YourUsername\\.cache'

# MinIO 配置（可选 - 如果不使用请留空）
minio:
  server_url: ''
  client_endpoint: ''
  access_key: ''
  secret: ''

# PostgreSQL 配置（可选 - 如果不使用请留空）
postgres:
  server_url: ''
  port: 5432
  username: ''
  password: ''
  database: ''

# 特定数据集路径（可选 - 可以覆盖 local_data_path 设置）
# camels_us: 'D:\project\data\CAMELS_US'
# selfmadehydrodataset: 'D:\project\data\FD_sources'
```

**需要配置的重要路径：**
- `local_data_path.root`: 所有水文数据的主目录
- `local_data_path.datasets-origin`: 存储下载的原始数据集的位置
- `local_data_path.cache`: 临时文件的缓存目录
- 您也可以直接指定数据集路径，如 `camels_us` 或 `selfmadehydrodataset` 来覆盖默认结构

**路径格式注意事项：**
- Windows: 使用单引号和双反斜杠：`'D:\\path\\to\\data'` 或正斜杠：`'D:/path/to/data'`
- Linux/macOS: 使用单引号：`'/home/user/data'`

#### 4. 验证配置

测试配置是否正确加载：

```python
from hydromodel import SETTING

print("配置加载成功！")
print("根数据路径:", SETTING.get("local_data_path", {}).get("root", "未配置"))
print("缓存目录:", SETTING.get("local_data_path", {}).get("cache", "未配置"))

# 如果您指定了特定数据集路径：
# print("CAMELS-US 路径:", SETTING.get("camels_us", "未配置"))
```

## 运行 XAJ 率定

项目包含用于 XAJ 模型的传统和现代率定脚本。

### 使用新的统一率定脚本（推荐）

#### 1. 准备配置文件

复制并修改示例配置：

```bash
cd hydrodhm/run_xaj
cp config_example.yaml my_calibration_config.yaml
```

编辑 `my_calibration_config.yaml` 设置您的参数：

```yaml
data:
  dataset: "camels_us"  # 或 "selfmadehydrodataset"
  path: "D:/path/to/data"     # 或完整的数据路径
  basin_ids: ["01013500"]  # 要率定的流域 ID
  train_period: ["1990-10-01", "2000-09-30"]
  test_period: ["2000-10-01", "2010-09-30"]
  warmup_length: 365
  output_dir: "D:/results/hydro/XAJ"
  experiment_name: "camels_xaj_test"

model:
  name: "xaj_mz"
  params:
    source_type: "sources"
    source_book: "HF"
    kernel_size: 15
    time_interval_hours: 24

training:
  algorithm: "SCE_UA"
  loss: "RMSE"
  SCE_UA:
    random_seed: 1234
    rep: 1000      
    ngs: 1000
    kstop: 50
    peps: 0.1
    pcento: 0.1

evaluation:
  metrics: ["NSE", "KGE", "RMSE"]
```

#### 2. 验证配置（可选但推荐）

在运行完整率定之前，验证您的配置：

```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml --dry-run
```

这将检查：
- 数据路径是否有效
- 流域 ID 是否存在
- 时间段格式是否正确
- 是否存在所有必需参数

#### 3. 运行率定

```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml
```

**快速测试**（较少迭代）：
```bash
# 临时覆盖参数进行快速测试
python calibrate_xaj_unified.py --config my_calibration_config.yaml --experiment-name quick_test
```

#### 4. 监控进度

率定过程将显示：
- 当前迭代次数
- 找到的最佳目标函数值
- 参数值

进度也会保存到输出目录。

#### 5. 查看结果

结果保存在 `output_dir/experiment_name` 指定的目录中：

```
D:/results/hydro/XAJ/camels_xaj_test/
├── calibration_config.yaml      # 使用的配置（如果使用 --save-config）
├── param_range.yaml             # 参数范围（如果使用 --save-config）
└── <basin_id>_sceua.csv        # 每个流域的 SCE-UA 优化历史
```

**注意：** 使用 `--save-config` 标志保存配置文件：
```bash
python calibrate_xaj_unified.py --config my_calibration_config.yaml --save-config
```

**关键输出文件：**
- `<basin_id>_sceua.csv`: 包含所有迭代和参数值的完整优化历史
  - `like1` 值最小的行包含最佳参数
- `calibration_config.yaml`: 用于重现性和评估的完整配置
- `param_range.yaml`: 参数范围（评估时需要用于反归一化参数）

### 使用传统脚本

为了与旧工作流程兼容，您仍然可以使用传统脚本：

```bash
python calibrate_xaj.py \
    --data_type camels \
    --data_dir camels_us \
    --basin_id 01013500 \
    --calibrate_period 1990-10-01 2000-09-30 \
    --test_period 2000-10-01 2010-09-30 \
    --exp test_calibration
```

所有可用的命令行参数请参见 `calibrate_xaj.py`。

### 多流域率定

要率定多个流域，在配置中列出它们：

```yaml
data:
  basin_ids:
    - "01013500"
    - "01022500"
    - "01030500"
```

脚本将独立率定每个流域。

### 交叉验证

在配置中启用 k 折交叉验证：

```yaml
data:
  cv_fold: 5  # 5 折交叉验证
```

## 模型评估

率定完成后，您可以使用新的统一评估接口在不同时间段评估模型性能。

### 使用统一评估脚本（推荐）

新的 `evaluate_xaj_unified.py` 脚本提供了简化而强大的评估接口：

#### 1. 基础评估（测试期）

在测试期进行评估（默认）：

```bash
cd hydrodhm/run_xaj
python evaluate_xaj_unified.py --exp expchangdian_61561
```

**参数：**
- `--exp`: 实验名称（包含率定结果的子目录）
- `--result-dir`: 存储结果的根目录（默认：`./results`）
- `--eval-period`: 要评估的时期（`train`、`test` 或 `custom`）

#### 2. 评估不同时期

**评估训练期：**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 --eval-period train
```

**评估测试期（默认）：**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 --eval-period test
```

**评估自定义时期：**
```bash
python evaluate_xaj_unified.py --exp expchangdian_61561 \
    --eval-period custom --custom-period 2020-01-01 2021-12-31
```

#### 3. 脚本功能

评估脚本将：
- 从率定结果中加载 `calibration_config.yaml`
- 自动查找并加载率定参数
- 使用 hydromodel 的统一 `evaluate()` API
- 计算全面的性能指标
- 以多种格式保存结果

#### 4. 查看评估结果

结果保存在实验文件夹内的子目录中：

```
results/expchangdian_61561/
├── calibration_config.yaml           # 原始率定配置
├── param_range.yaml                  # 参数范围（如果已保存）
├── <basin_id>_sceua.csv             # SCE-UA 率定结果
├── evaluation_test/                  # 测试期评估结果
│   ├── basins_metrics.csv            # 性能指标（NSE、KGE、RMSE 等）
│   ├── basins_norm_params.csv        # 归一化参数 [0,1]
│   ├── basins_denorm_params.csv      # 物理参数值
│   ├── xaj_mz_evaluation_results.nc  # 模拟结果（NetCDF）
├── evaluation_train/                 # 训练期评估结果
│   └── ...
└── evaluation_custom_2020-01-01_2021-12-31/  # 自定义时期结果
    └── ...
```

**关键输出文件：**
- `basins_metrics.csv`: 每个流域的性能指标（NSE、KGE、RMSE、PBIAS 等）
- `basins_denorm_params.csv`: 物理单位的率定参数
- `xaj_mz_evaluation_results.nc`: 包含时间序列数据的完整模拟结果

#### 5. 理解指标

评估会自动计算多个指标：

| 指标 | 描述 | 范围 | 最佳值 |
|------|------|------|--------|
| NSE | 纳什效率系数 | (-∞, 1] | 1.0 |
| KGE | Kling-Gupta 效率系数 | (-∞, 1] | 1.0 |
| RMSE | 均方根误差 | [0, ∞) | 0.0 |
| PBIAS | 百分比偏差 | (-∞, ∞) | 0.0 |

所有指标都会打印到控制台并保存到 CSV 文件以便分析。

### 使用传统评估脚本

为了与不使用统一配置格式的旧率定结果兼容：

```bash
python evaluate_xaj.py --exp expchangdian_61561 --result-dir results
```

**注意：** 传统脚本需要旧的配置格式，与统一脚本相比功能有限。

## 可视化

评估完成后，您可以使用统一可视化脚本生成发表级质量的图表。

### 使用统一可视化脚本

`visualize_unified.py` 脚本从评估结果创建全面的可视化：

#### 1. 基本用法（生成所有图表）

```bash
cd hydrodhm/run_xaj
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test
```

这将在 `figures/` 子目录中生成所有可用的图表类型。

#### 2. 生成特定图表类型

**仅时间序列图：**
```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --plot-types timeseries
```

**多种特定类型：**
```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --plot-types timeseries scatter fdc
```

**可用的图表类型：**
- `timeseries` - 带降水的时间序列图（双轴图）
- `scatter` - 观测值 vs 模拟值散点图，带密度着色
- `fdc` - 流量历时曲线（对数刻度）
- `monthly` - 月均值对比，带误差棒
- `metrics` - 多流域指标对比（仅用于多流域）
- `all` - 生成所有图表类型（默认）

#### 3. 可视化特定流域

对于多流域率定，选择特定流域绘图：

```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --basins 01013500 01022500
```

#### 4. 自定义输出目录

将图表保存到不同位置：

```bash
python visualize_unified.py --eval-dir results/your_exp_name/evaluation_test \
    --output-dir D:/my_figures
```

#### 5. 生成的图表

脚本生成适合发表的高质量图表（300 DPI）：

```
results/your_exp_name/evaluation_test/figures/
├── timeseries_01013500.png      # 带降水的时间序列图
├── scatter_01013500.png          # 带1:1线的散点图
├── fdc_01013500.png             # 流量历时曲线
├── monthly_01013500.png         # 月均值图
└── metrics_comparison.png       # 多流域对比（如适用）
```

**生成图表的特点：**
- **300 DPI 分辨率** - 可直接用于发表
- **自动计算指标** - 每张图上显示 NSE、KGE、RMSE、PBIAS
- **专业格式** - 一致的配色方案和字体
- **周期检测** - 自动标注"训练期"或"测试期"

#### 6. 完整工作流程示例

从率定到可视化的完整流程：

```bash
# 1. 率定模型
python calibrate_xaj_unified.py --config my_config.yaml --save-config

# 2. 在测试期评估
python evaluate_xaj_unified.py --exp my_experiment --eval-period test

# 3. 生成所有可视化
python visualize_unified.py --eval-dir results/my_experiment/evaluation_test

# 4. 查看图表
ls results/my_experiment/evaluation_test/figures/
```

#### 7. 解读结果

**时间序列图：**
- 上图：降水柱状图（倒置坐标轴）
- 下图：观测值（蓝色实线） vs 模拟值（红色虚线）径流
- 指标框：快速性能评估

**散点图：**
- 热图显示点密度（对数刻度）
- 黑色虚线：1:1 完美拟合
- 点在线上方：模型高估
- 点在线下方：模型低估

**流量历时曲线：**
- Y 轴对数刻度
- 显示整个时期的流量分布
- 良好匹配表示正确的流态表现

**月均值对比：**
- 柱状图带误差棒（标准差）
- 识别季节性偏差
- 对水资源规划有用

### 可视化技巧

1. **用于论文/演示文稿**：使用默认设置（300 DPI，所有图表类型）
2. **用于快速检查**：仅生成 `timeseries` 和 `scatter` 图
3. **用于多流域研究**：包含 `metrics` 对比图
4. **保存配置文件**：率定时始终使用 `--save-config` 以确保可重复性


## 项目结构

```
HydroDHM/
├── hydrodhm/
│   ├── run_xaj/                  # XAJ 模型率定
│   │   ├── calibrate_xaj.py              # 传统率定脚本
│   │   ├── calibrate_xaj_unified.py      # 新统一率定脚本
│   │   ├── config_example.yaml           # 示例配置
│   │   ├── evaluate_xaj.py               # 传统评估脚本
│   │   ├── evaluate_xaj_unified.py       # 新统一评估脚本
│   │   ├── visualize.py                  # 传统可视化工具
│   │   └── visualize_unified.py          # 新统一可视化工具
│   ├── streamflow_prediction/    # 神经网络模型
│   ├── data-limited_analysis/    # 数据有限流域实验
│   └── calculate_and_plot/       # 结果分析和可视化
├── pyproject.toml                # 项目依赖 (uv)
├── .python-version               # Python 版本 (3.11)
├── UV_SETUP.md                   # UV 环境设置指南
├── README.md                     # 英文版本
└── README_zh.md                  # 本文件（中文版本）
```

## 引用

如果您在研究中使用此代码，请引用：

```bibtex
@article{ouyang2024differentiable,
  title={A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins},
  author={Ouyang, Wenyu and others},
  journal={Journal of Hydrology},
  year={2024},
  doi={10.1016/j.jhydrol.2024.132471}
}
```

## 许可证

Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.

## 常见问题

### 常见问题

**1. "hydromodel not found" 错误：**
```bash
uv pip install hydromodel hydrodataset
```

**2. PyTorch CUDA 不可用：**
- 检查是否安装了 NVIDIA 驱动程序
- 验证 CUDA 版本匹配（默认：11.8）
- 参见 `pyproject.toml` 更改 CUDA 版本

**3. 数据下载失败：**
- 检查互联网连接
- 确保有足够的磁盘空间
- 尝试从 [CAMELS 网站](https://ral.ucar.edu/solutions/products/camels) 手动下载

**4. "Basin ID not found" 错误：**
- 验证流域 ID 存在于数据集中
- 检查 `hydro_setting.yml` 中的数据路径
- 确保数据已正确下载

**5. 评估脚本报错 - 数据类型或变量名冲突：**
- 确保您的配置文件格式正确
- 如果使用旧格式的率定结果，请使用传统评估脚本 `evaluate_xaj.py`
- 检查 `hydro_setting.yml` 中的数据集路径配置
- 更新 hydromodel 和 hydrodataset 包到最新版本

如需更多帮助，请在 GitHub 仓库上提出 issue。
