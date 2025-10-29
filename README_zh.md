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

### 下载 CAMELS 数据集

本项目使用 `hydrodataset` 包来下载和管理水文数据集。

#### 1. 安装 hydrodataset

该包应该已经作为依赖项的一部分安装。如果没有：

```bash
uv pip install hydrodataset
```

#### 2. 下载 CAMELS-US 数据集

使用 Python 下载数据集：

```python
from hydrodataset import Camels

# 初始化 CAMELS 数据集下载器
camels = Camels(data_path="path/to/your/data/directory")

# 下载数据集（这可能需要一段时间）
camels.download_data()
```

或使用命令行接口：

```bash
python -c "from hydrodataset import Camels; Camels(data_path='D:/data/camels_us').download_data()"
```

**推荐的数据目录结构：**
```
D:/data/
└── CAMELS_US/
    ├── basin_timeseries_v1p2_metForcing_obsFlow/
    ├── camels_attributes_v2.0/
    └── ...
```

#### 3. 验证下载

检查数据是否正确下载：

```python
from hydrodataset import Camels

camels = Camels(data_path="D:/data/CAMELS_US")
basin_ids = camels.read_object_ids()  # 获取流域 ID 列表
print(f"找到 {len(basin_ids)} 个流域")
```

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
  root: 'D:\data'

  # 原始数据集目录
  datasets-origin: 'D:\data'

  # 中间/处理后的数据集目录
  datasets-interim: 'D:\data'

  # 原始流域数据目录
  basins-origin: 'D:\data'

  # 中间/处理后的流域数据目录
  basins-interim: 'D:\data'

  # 缓存目录
  cache: 'C:\Users\YourUsername\.cache'

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
- Windows: 使用单引号和单反斜杠：`'D:\path\to\data'` 或正斜杠：`'D:/path/to/data'`
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
    rep: 1000      # 生产环境运行时增加
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
├── calibration_config.yaml      # 使用的配置
├── calibrated_params.json       # 找到的最佳参数
├── calibration_history.csv      # 优化历史
└── plots/                       # 可视化（如果启用）
```

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

率定完成后，您可以评估模型在训练期和测试期的性能。

### 使用统一评估脚本

新的 `evaluate_xaj_unified.py` 脚本适用于统一配置格式：

#### 1. 运行评估

```bash
cd hydrodhm/run_xaj
python evaluate_xaj_unified.py --exp expchangdian_61561 --result-dir results
```

**参数：**
- `--exp`: 实验名称（包含率定结果的子目录）
- `--result-dir`: 存储结果的根目录（默认：`./results`）

#### 2. 脚本功能

评估脚本将：
- 从率定结果中加载 `calibration_config.yaml`
- 使用率定的参数运行预测
- 评估训练期和测试期的性能
- 保存结果，包括：
  - 模拟流量时间序列
  - 观测值与模拟值的比较
  - 性能指标（NSE、KGE、RMSE 等）

#### 3. 查看评估结果

结果保存在实验文件夹内的子目录中：

```
results/expchangdian_61561/
├── sceua_xaj/                    # 单折（无交叉验证）
│   ├── train/                    # 训练期评估
│   │   ├── flow_pred.csv         # 预测流量
│   │   ├── flow_obs.csv          # 观测流量
│   │   └── metrics.json          # 性能指标
│   └── test/                     # 测试期评估
│       ├── flow_pred.csv
│       ├── flow_obs.csv
│       └── metrics.json
```

或对于交叉验证：

```
results/expchangdian_61561/
├── sceua_xaj_cv1/                # 折 1
│   ├── train/
│   └── test/
├── sceua_xaj_cv2/                # 折 2
│   ├── train/
│   └── test/
└── ...
```

### 使用传统评估脚本

为了与旧率定结果兼容：

```bash
python evaluate_xaj.py --exp expchangdian_61561 --result-dir results
```

**注意：** 传统脚本需要旧的配置格式。

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
│   │   └── visualize.py                  # 可视化工具
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
