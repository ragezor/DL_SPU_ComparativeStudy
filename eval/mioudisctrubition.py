import os
import re

import matplotlib.pyplot as plt
import numpy as np

# 指定文件夹路径
folder_path = r'G:\metric_map\real\SSIM'
data='blender.txt'

# 获取文件夹下所有以'zer.txt'结尾的文件
file_list = [file for file in os.listdir(folder_path) if file.endswith(data)]

# 定义颜色列表，用于每个散点图的染色
color_list = ['red', 'blue', 'green', 'orange', 'purple']

# 自定义'myloss'散点图的颜色
myloss_color = 'green'

# 创建画布和子图
fig, ax = plt.subplots()

# 存储'myloss'文件的数据
myloss_data = []

# 遍历文件列表，绘制除'myloss'外的散点图
for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    #     data = [float(re.sub(r'%', '', line.strip())) / 100 for line in lines]

    # data = [value for value in data if value <= 1]

    # 读取文件数据
    data = np.loadtxt(file_path)
    # print(len(data))
    data = data[data <= 1]


    # 提取网络名和数据集名
    network_name = file_name.replace('output', '').replace('blender.txt', '')
    dataset_name = 'blender'

    # 判断是否为'myloss'文件
    color = color_list[i % len(color_list)]  # 使用color_list中的颜色循环使用
    if network_name == 'erfnet':
        network_name = 'ERFNet'
        color = 'blue'
    elif network_name == 'tiramisu':
        network_name = 'PhaseNet2.0'
        color = 'orange'
    elif network_name == 'REDN':
        color = 'red'
    elif network_name == "VuRNet":
        color = 'purple'
    elif network_name == "Zmyloss":
        myloss_data=data
        network_name = 'Our method'
        color = myloss_color
        print(1)
        # 选择颜色





        # 绘制散点图
    ax.scatter(range(len(data)), data, label=f'{network_name}', color=color, marker='o', s=5)

# 绘制'myloss'的散点图
# myloss_data = np.array([float(value.strip('%')) for value in myloss_data])
# #
# # # 将'myloss'数据转换为小数
# myloss_data = myloss_data / 100
# ax.scatter(range(len(myloss_data)), myloss_data, label='Our method', color=myloss_color, marker='o', s=5)

# 添加图例
ax.legend(loc='lower right')

# 设置横轴和纵轴标签
ax.set_xlabel('sample number')
ax.set_ylabel('SSIM')

# 显示图形
plt.show()
