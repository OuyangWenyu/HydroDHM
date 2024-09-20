"""
查看不同径流数据缺失率情况下模型训练效果，输出结果为验证集上的指标变化图表，包括Loss, NSE, RMSE, Corr, KGE, FHV, FLV
"""


import os
import json
import glob
import matplotlib.pyplot as plt
import re
import csv

# 定义Streamflow_Prediction目录的路径
root_dir = './results/data-limited_analysis/camels10y_module'
figure_dir = './results/evaluation_indices/data-limited_analysis/camels10y_module'
csv_file = './results/evaluation_indices/data-limited_analysis/camels10y_module/evaluation_indices_summary.csv'

# 确保保存图表的目录存在
os.makedirs(figure_dir, exist_ok=True)

# 初始化CSV文件，写入表头
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['basin_id', 'train_loss_final', 'validation_loss_final', 
                     'dpl_nse_final', 'dpl_rmse_final', 'dpl_corr_final', 'dpl_kge_final', 'dpl_fhv_final', 'dpl_flv_final', 
                     'epoch_of_min_vali_loss', 'dpl_nse_relational', 
                     'dpl_rmse_relational', 'dpl_corr_relational', 'dpl_kge_relational', 'dpl_fhv_relational', 'dpl_flv_relational'])

# 遍历Streamflow_Prediction目录中的所有文件夹
for location in os.listdir(root_dir):
    location_dir = os.path.join(root_dir, location)
    
    # 使用glob查找匹配的json文件
    json_files = glob.glob(os.path.join(location_dir, '*_September_2024*.json'))
    
    # 如果找到了匹配的json文件
    if json_files:
        json_file = json_files[0]  # 假设只有一个匹配文件
        
        # 读取JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 提取epoch, train_loss, validation_loss, NSE, RMSE, Corr, KGE, FLV 和 FHV值
        epochs = []
        train_losses = []
        validation_losses = []
        nse_values = []
        rmse_values = []
        corr_values = []
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
            
            # 提取Corr值
            corr_values.append(run['validation_metric']['Corr of streamflow'][0])
            
            # 提取KGE值
            kge_values.append(run['validation_metric']['KGE of streamflow'][0])

            # 提取FHV值
            fhv_values.append(run['validation_metric']['FHV of streamflow'][0])     

            # 提取FLV值
            flv_values.append(run['validation_metric']['FLV of streamflow'][0])
    

        # 找到validation_loss最小值及其对应的epoch
        min_validation_loss = min(validation_losses)
        epoch_of_min_validation_loss = epochs[validation_losses.index(min_validation_loss)]
        
        # 提取与 epoch_of_min_validation_loss 对应的指标值
        nse_at_min_loss = nse_values[epochs.index(epoch_of_min_validation_loss)]
        rmse_at_min_loss = rmse_values[epochs.index(epoch_of_min_validation_loss)]
        corr_at_min_loss = corr_values[epochs.index(epoch_of_min_validation_loss)]
        kge_at_min_loss = kge_values[epochs.index(epoch_of_min_validation_loss)]
        fhv_at_min_loss = fhv_values[epochs.index(epoch_of_min_validation_loss)]
        flv_at_min_loss = flv_values[epochs.index(epoch_of_min_validation_loss)]
        
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

        # 标注 validation_loss 最小值处
        axs[0].plot(epoch_of_min_validation_loss, min_validation_loss, marker='x', color='red', markersize=10)
        axs[0].text(epoch_of_min_validation_loss, min_validation_loss + 0.02, 
                    f'Min Vali. Loss: {min_validation_loss:.2f}', fontsize=15, color='red', ha='center', fontweight='bold')
        
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
        
        # 在validation_loss最小处标注
        axs[1].text(epoch_of_min_validation_loss, nse_at_min_loss - 0.02, f'NSE at Min-loss: {nse_at_min_loss:.2f}', fontsize=15, color='red', ha='center', va='top', fontweight='bold')
        axs[1].plot(epoch_of_min_validation_loss, nse_at_min_loss, marker='x', color='red', markersize=10)

        # 在最后一个epoch的NSE值附近标注数值
        final_nse = nse_values[-1]
        axs[1].text(final_epoch, final_nse + 0.02, f'Final NSE: {final_nse:.2f}', fontsize=15, color='green', ha='center', fontweight='bold')

        # 下面5个子图：绘制RMSE, Corr, KGE, FLV, FHV图表
        metrics = [
            ('RMSE', rmse_values, 'RMSE'),
            ('Corr', corr_values, 'Corr'),
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

            # 标出epoch_of_min_validation_loss对应的值，并加粗
            value_at_min_validation_loss = metric_values[epochs.index(epoch_of_min_validation_loss)]
            axs[i+2].plot(epoch_of_min_validation_loss, value_at_min_validation_loss, marker='x', color='red', markersize=10)
            axs[i+2].text(epoch_of_min_validation_loss, value_at_min_validation_loss + 0.02, f'{value_at_min_validation_loss:.2f}', fontsize=15, color='red', ha='center', fontweight='bold')

        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表到指定目录
        output_filename = os.path.join(figure_dir, f'evaluation_indices_of_{location}.png')
        plt.savefig(output_filename, dpi=600)
        
        # 将最终的train_loss, validation_loss, nse, rmse, corr, kge, fhv, flv写入CSV，保留3位小数，并增加最大epoch对应的值
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([location, 
                             f'{final_train_loss:.3f}', 
                             f'{final_validation_loss:.3f}', 
                             f'{final_nse:.3f}',
                             f'{rmse_values[-1]:.3f}', 
                             f'{corr_values[-1]:.3f}', 
                             f'{kge_values[-1]:.3f}',
                             f'{fhv_values[-1]:.3f}', 
                             f'{flv_values[-1]:.3f}',                         
                             epoch_of_min_validation_loss,
                             f'{nse_at_min_loss:.3f}',
                             f'{rmse_at_min_loss:.3f}', 
                             f'{corr_at_min_loss:.3f}', 
                             f'{kge_at_min_loss:.3f}',
                             f'{fhv_at_min_loss:.3f}', 
                             f'{flv_at_min_loss:.3f}'])

