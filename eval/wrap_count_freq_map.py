import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.ticker as ticker  # 导入ticker模块

# 设置字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 创建数据集
data = {
    'dataset#1': [2.198, 2.953, 3.861, 4.994, 6.247, 7.568, 9.318, 13.690, 13.838, 9.287, 7.408, 5.981, 4.704, 3.534, 2.558, 1.860, 0.002],
    'dataset#2': [5.111, 4.423, 5.032, 5.751, 6.459, 7.094, 7.575, 8.104, 8.149, 7.619, 7.257, 6.607, 5.920, 5.172, 4.483, 5.168, 0.076]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置柱状图的样式和设置
plt.figure(figsize=(10, 6))  # 设置图形大小
ax = df.plot(kind='bar', fontsize=22)  # 绘制柱状图并设置字体大小为16号
plt.xlabel('Wrap-count', fontsize=22)  # 设置X轴标签和字体大小
plt.ylabel('Proportion (%)', fontsize=22)  # 设置Y轴标签和字体大小
plt.ylim(0, 16)  # 设置Y轴坐标范围

# 在每个柱状图上显示对应的数字（横着显示）
for i in ax.patches:
    x = i.get_x() + i.get_width() / 2
    y = i.get_height()
    y += 0.3  # 调整文字到柱状图的距离
    ax.text(x, y, f"{i.get_height():.2f}", ha='center', va='bottom', rotation='vertical', fontsize=16)

# 设置坐标轴字体为竖直显示
plt.xticks(rotation='horizontal', fontsize=22)

# 设置纵坐标刻度为每5个数有一个尺度
ax.yaxis.set_major_locator(ticker.MultipleLocator(base=5))

plt.yticks(fontsize=22)

# 显示图例并设置字体大小，并固定在右上角，稍微向左下方偏移
plt.legend(fontsize=22, loc='upper right', bbox_to_anchor=(0.8, 0.7), bbox_transform=ax.transAxes)

plt.show()
