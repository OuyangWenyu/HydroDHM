from hydroutils.hydro_plot import plot_boxes_matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from torchhydro import SETTING
from definitions import RESULT_DIR

if __name__ == "__main__":
    """
    This script reads NSE values from two CSV files and generates a boxplot comparing them.
    """

    # 读取两个 CSV 文件
    df1 = pd.read_csv(
        os.path.join(RESULT_DIR, "neuralhydrology", "camels_all", "test_metrics.csv")
    )  # 包含 basin 列

    df2 = pd.read_csv(
        os.path.join(RESULT_DIR, "lstm_results", "camels_all", "metric_streamflow.csv")
    )  # 包含 basin_id 列

    # 提取 NSE 数据
    nse1 = df1["NSE"].values
    nse2 = df2["NSE"].values

    # 准备数据列表
    data = [[nse1, nse2]]  # 双层列表：外层是子图，内层是每个子图中的多个箱子

    # 绘制箱型图
    fig = plot_boxes_matplotlib(
        data=data,
        label1=["NSE"],  # 子图标签
        label2=["NeuralHydrology", "Torchhydro"],  # 图例标签
        colorlst="ry",  # 第一个箱子红色，第二个蓝色
        title="NSE Comparison",
        figsize=(4, 8),
        ylabel=["NSE"],
        show_median=True,
    )

    plt.savefig(
        os.path.join(RESULT_DIR, "figures", "nse_boxplot.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
