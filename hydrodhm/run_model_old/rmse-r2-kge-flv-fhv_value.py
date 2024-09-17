import os
import json
import glob
import matplotlib.pyplot as plt
import re
import csv

# 定义Streamflow_Prediction目录的路径
root_dir = './results/streamflow_prediction_100epoch'
figure_dir = './results/evaluation_indices/100epoch'
csv_file = './results/evaluation_indices/100epoch/rmse_r2_kge_flv_fhv_summary.csv'

# 确保保存图表的目录存在
os.makedirs(figure_dir, exist_ok=True)

# 初始化CSV文件，写入表头，新增5个指标列：rmse, r2, kge, flv, fhv
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['basin_id', 'dpl_rmse', 'dpl_r2', 'dpl_kge', 'dpl_fhv', 'dpl_flv'])

# 遍历Streamflow_Prediction目录中的所有文件夹
for location in os.listdir(root_dir):
    location_dir = os.path.join(root_dir, location)
    
    # 使用glob查找匹配的json文件
    json_files = glob.glob(os.path.join(location_dir, '06_September_2024*.json'))
    
    # 如果找到了匹配的json文件
    if json_files:
        json_file = json_files[0]  # 假设只有一个匹配文件
        
        # 读取JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取epoch, rmse, r2, kge, flv 和 fhv值
        epochs = []
        rmse_values = []
        r2_values = []
        kge_values = []
        flv_values = []
        fhv_values = []
        
        for run in data['run']:
            epochs.append(run['epoch'])
            
            # 提取RMSE值
            rmse_values.append(run['validation_metric']['RMSE of streamflow'][0])
            
            # 提取R²值
            r2_values.append(run['validation_metric']['R2 of streamflow'][0])
            
            # 提取KGE值
            kge_values.append(run['validation_metric']['KGE of streamflow'][0])

            # 提取FHV值
            fhv_values.append(run['validation_metric']['FHV of streamflow'][0])     

            # 提取FLV值
            flv_values.append(run['validation_metric']['FLV of streamflow'][0])
        
        # 创建图像和五个子图，分别绘制rmse, r2, kge, flv, fhv
        fig, axs = plt.subplots(5, 1, figsize=(10, 20))
        
        # 绘制RMSE图表
        axs[0].plot(epochs, rmse_values, marker='o', label='RMSE')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('RMSE')
        axs[0].set_title(f'RMSE over Epochs for {location}')
        axs[0].legend()
        axs[0].grid(True)
        
        # 绘制R²图表
        axs[1].plot(epochs, r2_values, marker='o', label='R²')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('R²')
        axs[1].set_title(f'R² over Epochs for {location}')
        axs[1].legend()
        axs[1].grid(True)
        
        # 绘制KGE图表
        axs[2].plot(epochs, kge_values, marker='o', label='KGE')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('KGE')
        axs[2].set_title(f'KGE over Epochs for {location}')
        axs[2].legend()
        axs[2].grid(True)
        
        # 绘制FHV图表
        axs[4].plot(epochs, fhv_values, marker='o', label='FHV')
        axs[4].set_xlabel('Epoch')
        axs[4].set_ylabel('FHV')
        axs[4].set_title(f'FHV over Epochs for {location}')
        axs[4].legend()
        axs[4].grid(True)

        # 绘制FLV图表
        axs[3].plot(epochs, flv_values, marker='o', label='FLV')
        axs[3].set_xlabel('Epoch')
        axs[3].set_ylabel('FLV')
        axs[3].set_title(f'FLV over Epochs for {location}')
        axs[3].legend()
        axs[3].grid(True)
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表到指定目录
        output_filename = os.path.join(figure_dir, f'rmse_r2_kge_fhv_flv_{location}.png')
        plt.savefig(output_filename, dpi=300)
        
        # 显示图表（可选）
        plt.show()

        # 将最终的rmse, r2, kge, fhv, flv写入CSV，保留3位小数
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([location, 
                             f'{rmse_values[-1]:.3f}', 
                             f'{r2_values[-1]:.3f}', 
                             f'{kge_values[-1]:.3f}',
                             f'{fhv_values[-1]:.3f}', 
                             f'{flv_values[-1]:.3f}'])  # 保留3位小数写入每个流域的最终值
