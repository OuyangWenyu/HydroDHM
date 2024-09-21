<!--
 * @Author: Wenyu Ouyang
 * @Date: 2023-10-12 15:44:30
 * @LastEditTime: 2024-09-21 14:39:10
 * @LastEditors: Wenyu Ouyang
 * @Description: README FOR HYDROTL
 * @FilePath: \HydroDHM\README.md
 * Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
-->
# A Differentiable Hydrological Model in Data-Scarce Basins

This repository contains the code for the paper "A Differentiable, Physics-Based Hydrological Model and Its Evaluation for Data-Limited Basins" (link to be added in the future). The code builds upon the [torchhydro](https://github.com/OuyangWenyu/torchhydro) and the [hydromodel](https://github.com/OuyangWenyu/hydromodel) packages. The former is a PyTorch-based hydrological modeling framework, while the latter provides implementations of traditional hydrological models, including the Xin'anjiang model, using Numpy.

The experiments conducted with these models can be found in the `run_xaj`, `streamflow_prediction` and `data-limited_analysis` directories. The `calculate_and_plot` directory contains the scripts used to generate the results and figures.

The model is trained and evaluated on the CAMELS dataset as well as on several basins located upstream of the Three Gorges Reservoir Area, in or near Sichuan Province, China.

For the CAMELS dataset, we will make the data available in the future. However, data sharing for the Three Gorges Reservoir Area may be subject to certain policy restrictions.
