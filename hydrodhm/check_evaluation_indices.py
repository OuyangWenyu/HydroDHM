import os
import json
import glob
import matplotlib.pyplot as plt
import re
import csv

# 定义Streamflow_Prediction目录的路径
root_dir = './results/streamflow_prediction_camels'
figure_dir = './results/evaluation_indices/camels'
csv_file = './results/evaluation_indices/camels/evaluation_indices_summary.csv'

# 确保保存图表的目录存在
os.makedirs(figure_dir, exist_ok=True)

# 初始化CSV文件，写入表头，新增指标列，包括与 epoch_of_max_nse 对应的值
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['basin_id', 'train_loss_final', 'validation_loss_final', 
                     'dpl_nse_final', 'dpl_rmse_final', 'dpl_r2_final', 'dpl_kge_final', 'dpl_fhv_final', 'dpl_flv_final', 
                     'epoch_of_max_nse', 'dpl_nse_max', 
                     'dpl_rmse_relational', 'dpl_r2_relational', 'dpl_kge_relational', 'dpl_fhv_relational', 'dpl_flv_relational'])

# 遍历Streamflow_Prediction目录中的所有文件夹
for location in os.listdir(root_dir):
    location_dir = os.path.join(root_dir, location)
    
    # 使用glob查找匹配的json文件
    json_files = glob.glob(os.path.join(location_dir, '11_September_2024*.json'))
    
    # 如果找到了匹配的json文件
    if json_files:
        json_file = json_files[0]  # 假设只有一个匹配文件
        
        # 读取JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取epoch, train_loss, validation_loss, NSE, rmse, r2, kge, flv 和 fhv值
        epochs = []
        train_losses = []
        validation_losses = []
        nse_values = []
        rmse_values = []
        r2_values = []
        kge_values = []
        fhv_values = []
        flv_values = []
        
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
        
        # 找到NSE最大值及其对应的epoch
        max_nse = max(nse_values)
        epoch_of_max_nse = epochs[nse_values.index(max_nse)]
        
        # 提取与 epoch_of_max_nse 对应的其他指标值
        rmse_at_max_nse = rmse_values[epochs.index(epoch_of_max_nse)]
        r2_at_max_nse = r2_values[epochs.index(epoch_of_max_nse)]
        kge_at_max_nse = kge_values[epochs.index(epoch_of_max_nse)]
        fhv_at_max_nse = fhv_values[epochs.index(epoch_of_max_nse)]
        flv_at_max_nse = flv_values[epochs.index(epoch_of_max_nse)]
        
        # 创建单列的图像，共7个子图
        fig, axs = plt.subplots(7, 1, figsize=(10, 35))  # 将子图设为7行1列
        
        # 第一张图：绘制Loss图表
        axs[0].plot(epochs, train_losses, marker='o', markersize=4, label='Train Loss')
        axs[0].plot(epochs, validation_losses, marker='o', markersize=4, label='Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f'Train and Validation Loss over Epochs for {location}')
        axs[0].legend()
        axs[0].grid(True)
        
        # 在最后一个epoch的损失值附近标注数值
        final_epoch = epochs[-1]
        final_train_loss = train_losses[-1]
        final_validation_loss = validation_losses[-1]
        axs[0].text(final_epoch, final_train_loss + 0.02, f'{final_train_loss:.2f}', fontsize=15, color='blue', ha='center', fontweight='bold')
        axs[0].text(final_epoch, final_validation_loss + 0.02, f'{final_validation_loss:.2f}', fontsize=15, color='red', ha='center', fontweight='bold')
        
        # 第二张图：绘制NSE图表
        axs[1].plot(epochs, nse_values, marker='o', markersize=4)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('NSE')
        axs[1].set_title(f'NSE over Epochs for {location}')
        axs[1].grid(True)
        
        # 在NSE最大值处标注
        axs[1].text(epoch_of_max_nse, max_nse - 0.02, f'Max NSE: {max_nse:.2f}', fontsize=15, color='red', ha='center', va='top', fontweight='bold')
        axs[1].plot(epoch_of_max_nse, max_nse, marker='x', color='red', markersize=10)

        # 在最后一个epoch的NSE值附近标注数值
        final_nse = nse_values[-1]
        axs[1].text(final_epoch, final_nse + 0.02, f'Final NSE: {final_nse:.2f}', fontsize=15, color='green', ha='center', fontweight='bold')

        # 下面5个子图：绘制RMSE, R², KGE, FLV, FHV图表
        metrics = [
            ('RMSE', rmse_values, 'RMSE'),
            ('R²', r2_values, 'R²'),
            ('KGE', kge_values, 'KGE'),
            ('FHV', fhv_values, 'FHV'),
            ('FLV', flv_values, 'FLV')
        ]

        for i, (metric_name, metric_values, label) in enumerate(metrics):
            axs[i+2].plot(epochs, metric_values, marker='o', markersize=4, label=metric_name)
            axs[i+2].set_xlabel('Epoch')
            axs[i+2].set_ylabel(metric_name)
            axs[i+2].set_title(f'{metric_name} over Epochs for {location}')
            axs[i+2].legend()
            axs[i+2].grid(True)

            # 标出 epoch_of_max_nse 对应的值，并加粗
            value_at_max_nse = metric_values[epochs.index(epoch_of_max_nse)]
            axs[i+2].plot(epoch_of_max_nse, value_at_max_nse, marker='x', color='red', markersize=10)
            axs[i+2].text(epoch_of_max_nse, value_at_max_nse + 0.02, f'{value_at_max_nse:.2f}', fontsize=15, color='red', ha='center', fontweight='bold')

        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表到指定目录
        output_filename = os.path.join(figure_dir, f'evaluation_indices_of_{location}.png')
        plt.savefig(output_filename, dpi=600)
        
        # 将最终的train_loss, validation_loss, nse, rmse, r2, kge, fhv, flv写入CSV，保留3位小数，并增加最大epoch对应的值
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([location, 
                             f'{final_train_loss:.3f}', 
                             f'{final_validation_loss:.3f}', 
                             f'{final_nse:.3f}',
                             f'{rmse_values[-1]:.3f}', 
                             f'{r2_values[-1]:.3f}', 
                             f'{kge_values[-1]:.3f}',
                             f'{fhv_values[-1]:.3f}', 
                             f'{flv_values[-1]:.3f}',                         
                             epoch_of_max_nse,
                             f'{max_nse:.3f}',
                             f'{rmse_at_max_nse:.3f}', 
                             f'{r2_at_max_nse:.3f}', 
                             f'{kge_at_max_nse:.3f}',
                             f'{fhv_at_max_nse:.3f}', 
                             f'{flv_at_max_nse:.3f}'])
