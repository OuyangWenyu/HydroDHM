import os
import json
import glob
import matplotlib.pyplot as plt
import re
import csv

# 定义Streamflow_Prediction目录的路径
root_dir = './results/streamflow_prediction_50epoch'
figure_dir = './results/evaluation_indices/50epoch'
csv_file = './results/evaluation_indices/50epoch/loss_nse_summary.csv'

# 确保保存图表的目录存在
os.makedirs(figure_dir, exist_ok=True)

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['basin_id', 'train_loss', 'validation_loss', 'dpl_nse', 'dpl_nse_max', 'epoch_of_max_nse'])

# 遍历Streamflow_Prediction目录中的所有文件夹
for location in os.listdir(root_dir):
    location_dir = os.path.join(root_dir, location)
    
    # 使用glob查找匹配的json文件
    json_files = glob.glob(os.path.join(location_dir, '05_September_2024*.json'))
    
    # 如果找到了匹配的json文件
    if json_files:
        json_file = json_files[0]  # 假设只有一个匹配文件
        
        # 读取JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取epoch, train_loss, validation_loss 和 NSE值
        epochs = []
        train_losses = []
        validation_losses = []
        nse_values = []
        
        for run in data['run']:
            epochs.append(run['epoch'])
            train_losses.append(float(run['train_loss']))
            
            # 使用正则表达式从字符串中提取数值部分
            validation_loss_str = run['validation_loss']
            match = re.search(r'tensor\(([\d\.]+)', validation_loss_str)
            if match:
                validation_losses.append(float(match.group(1)))
            else:
                validation_losses.append(float('nan'))  # 如果没有匹配到，则加入NaN
            
            # 提取NSE值
            nse_values.append(run['validation_metric']['NSE of streamflow'][0])
        
        # 找到NSE最大值及其对应的epoch
        max_nse = max(nse_values)
        epoch_of_max_nse = epochs[nse_values.index(max_nse)]

        # 创建图像和两个子图
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制Loss图表
        axs[0].plot(epochs, train_losses, marker='o', label='Train Loss')
        axs[0].plot(epochs, validation_losses, marker='o', label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Train and Validation Loss over Epochs for {location}')
        axs[0].legend()
        axs[0].grid(True)
        
        # 在最后一个epoch的损失值附近标注数值
        final_epoch = epochs[-1]
        final_train_loss = train_losses[-1]
        final_validation_loss = validation_losses[-1]
        axs[0].text(final_epoch, final_train_loss + 0.02, f'{final_train_loss:.2f}', fontsize=12, color='blue', ha='center')
        axs[0].text(final_epoch, final_validation_loss + 0.02, f'{final_validation_loss:.2f}', fontsize=12, color='red', ha='center')
        
        # 绘制NSE图表
        axs[1].plot(epochs, nse_values, marker='o')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('NSE')
        axs[1].set_title(f'NSE over Epochs for {location}')
        axs[1].grid(True)
        
        # 在NSE最大值处标注
        axs[1].text(epoch_of_max_nse, max_nse - 0.02, f'Max NSE: {max_nse:.2f}', fontsize=12, color='red', ha='center', va='top', fontweight='bold')
        axs[1].plot(epoch_of_max_nse, max_nse, marker='x', color='red', markersize=10)

        # 在最后一个epoch的NSE值附近标注数值
        final_nse = nse_values[-1]
        axs[1].text(final_epoch, final_nse + 0.02, f'Final NSE: {final_nse:.2f}', fontsize=12, color='green', ha='center', fontweight='bold')
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表到指定目录
        output_filename = os.path.join(figure_dir, f'loss_nse_{location}.png')
        plt.savefig(output_filename, dpi=300)
        
        # 显示图表（可选）
        plt.show()

        # 将最终的train_loss, validation_loss 和 nse写入CSV，保留3位小数
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([location, 
                             f'{final_train_loss:.3f}', 
                             f'{final_validation_loss:.3f}', 
                             f'{final_nse:.3f}',
                             f'{max_nse:.3f}',
                             epoch_of_max_nse])  # 保留3位小数写入每个流域的最终值