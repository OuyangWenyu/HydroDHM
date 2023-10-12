<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-12 15:44:30
 * @LastEditTime: 2023-10-12 16:22:11
 * @LastEditors: Wenyu Ouyang
 * @Description: README FOR HYDROTL
 * @FilePath: \HydroTL\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# A transfer-learning model for daily streamflow prediction in data-scarce basins

## Introduction

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems. Transfer learning is also used in more traditional machine learning, although it is less prevalent.

For streamflow prediction, transfer learning is a promising approach to improve the performance of deep learning models in data-scarce basins. In this study, we propose a transfer-learning model for daily streamflow prediction in data-scarce basins. The proposed model is based on the Long Short-Term Memory (LSTM) network and the transfer-learning technique. The LSTM network is used to capture the temporal dependencies of streamflow time series. The transfer-learning technique is used to transfer the knowledge learned from the source basin to the target basin. The proposed model is evaluated in the some basins in China: Duoyingping, Fujiangqiao etc.

## Environment

Use conda to create a new environment and install the required packages.

```bash
# you can use mamba to install env more quickly
conda install mamba -c conda-forge
# windows
mamba env create -f env-windows.yml
# linux
mamba env create -f env-linux.yml
```

To run notebooks in this repository, you need to activate the environment, install and choose the jupyter kernel.

```bash
# check if you have installed jupyterlab
jupyter lab --version
conda activate TL # activate environment
# install ipykernel and choose kernel
mamba install ipykernel
python -m ipykernel install --user --name TL --display-name "TL" # install kernel
jupyter kernelspec list # check kernel
# you can see tl
```

## Data

Download the data from ...

Source domain data:

- CAMELS

target domain data:

- Duoyingping
- Fujiangqiao
