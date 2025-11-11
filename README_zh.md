<!--
 * @Author: Wenyu Ouyang
 * @Date: 2025-10-29
 * @LastEditTime: 2025-11-09
 * @LastEditors: Wenyu Ouyang
 * @Description: 中文版 README
 * @FilePath: \HydroDHM\README_zh.md
 * Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
-->

# HydroDHM - 可微分水文模型

基于PyTorch的可微分水文建模框架，专为数据稀缺流域设计，提供新安江(XAJ)模型的自动率定和评估工具。

本仓库包含论文《A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins》（[Journal of Hydrology](https://doi.org/10.1016/j.jhydrol.2024.132471)）的代码。

[English](README.md) | 简体中文

## 功能特性

**物理模型：**
- XAJ模型的命令行率定、评估和可视化工具
- 多种优化算法（SCE-UA、GA、scipy）

**深度学习模型：**
- LSTM神经网络径流预测
- DPL-XAJ：物理约束混合深度学习模型
- 支持CAMELS数据集的训练脚本

**通用功能：**
- 支持CAMELS数据集和自定义水文数据
- 全面的评估指标和发表级质量的可视化图表
- 基于[hydromodel](https://github.com/OuyangWenyu/hydromodel)和[torchhydro](https://github.com/OuyangWenyu/torchhydro)构建

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/OuyangWenyu/HydroDHM.git
cd HydroDHM

# 安装uv（如果尚未安装）
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建并激活虚拟环境
uv venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 安装依赖
uv sync --all-extras
```

### 2. 配置

在用户主目录创建`hydro_setting.yml`：

```yaml
local_data_path:
  datasets-origin: 'D:\data'  # 修改为你的数据路径
  cache: 'D:\data\.cache'
```

### 3. 下载 CAMELS 数据集

**方式A：自动下载（推荐）**

使用我们的下载工具自动获取 CAMELS-US 数据：

```bash
# 列出所有可用的 CAMELS 数据集
python hydrodhm/data_tools/download_camels.py --list

# 下载 CAMELS-US（使用 hydro_setting.yml 中的路径）
python hydrodhm/data_tools/download_camels.py camels_us

# 或指定下载路径
python hydrodhm/data_tools/download_camels.py camels_us --data-path D:/data/
```

**注意：** 首次下载需要 1-3 小时（约 15GB）。数据会缓存为 NetCDF 文件，后续访问即时加载。

**方式B：首次运行时自动下载**

跳过此步骤，数据将在首次运行模型时自动下载。但这可能会中断您的工作流程。

**方式C：手动下载**

可能由于网络原因无法自动下载，如果可以您需要手动下载：
1. 访问 [CAMELS-US on Zenodo](https://zenodo.org/records/15529996)
2. 下载并解压到您的 `datasets-origin` 目录
3. 确保目录结构为：`datasets-origin/CAMELS_US/...` (这里的camels_us数据集路径名要大写)

### 4. 准备配置文件

根据你的数据类型选择配置模板：

**使用CAMELS数据集：**
```bash
cd hydrodhm/run_xaj
cp config_camels.yaml my_config.yaml
```

**使用自定义数据集：**
```bash
cd hydrodhm/run_xaj
cp config_custom.yaml my_config.yaml
```

详细示例见`config_camels.yaml`和`config_custom.yaml`。

编辑`my_config.yaml`设置您的流域ID、时间段和参数。

### 5. 运行XAJ模型工作流

**步骤1：率定模型**
```bash
python calibrate_xaj_unified.py --config my_config.yaml
```

**步骤2：在测试期评估**
```bash
python evaluate_xaj_unified.py --exp your_experiment_name --eval-period test
```

**步骤3：生成可视化图表**
```bash
python visualize_unified.py --eval-dir results/your_experiment_name/evaluation_test
```



## 深度学习模型

HydroDHM也提供基于[torchhydro](https://github.com/OuyangWenyu/torchhydro)的深度学习模型用于径流预测。

### LSTM模型

训练标准LSTM神经网络：

```bash
cd hydrodhm/run_lstm

# 训练LSTM模型
python lstm_camels_example.py
```

### DPL-XAJ模型

训练混合物理-机器学习模型（LSTM + XAJ）：

```bash
cd hydrodhm/run_lstm

# 训练DPL-XAJ模型
python dpl_xaj_example.py
```

### 模型对比

| 模型 | 类型 | 优势 | 适用场景 |
|------|------|------|----------|
| **XAJ** | 物理模型 | 可解释性强，参数少 | 物理理解、传统率定 |
| **LSTM** | 纯深度学习 | 训练快，数据驱动 | 大数据集，纯预测 |
| **DPL-XAJ** | 混合物理-ML | 结合物理与数据 | 兼顾两者优势 |

### 调试和参数调优

关于深度学习模型配置和调试的详细指导，请参阅：

📖 **[LSTM配置和调试](docs/usage.md#lstm-configuration-details-and-debugging)** | **[DPL-XAJ配置和调试](docs/usage.md#dpl-xaj-configuration-details-and-debugging)**

本指南涵盖：
- 带注释的完整配置示例
- 常见错误及逐步解决方案
- 参数调优策略
- 维度不匹配调试
- 内存优化技巧
- 配置检查清单

## 文档

详细信息请参考：

- **物理模型（XAJ）**：
  - [hydromodel包](https://github.com/OuyangWenyu/hydromodel) - 模型实现和API
  - 自定义数据集的数据格式要求
  - 参数范围和物理含义

- **深度学习模型（LSTM、DPL-XAJ）**：
  - [torchhydro包](https://github.com/OuyangWenyu/torchhydro) - 深度学习框架
  - [torchhydro文档](https://OuyangWenyu.github.io/torchhydro) - 完整API参考
  - 模型架构和训练策略

- **数据集**：
  - [hydrodataset](https://github.com/OuyangWenyu/hydrodataset) - CAMELS等数据集（自动下载）



## 许可证

Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
