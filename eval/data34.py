
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import matplotlib.ticker as ticker  # 导入ticker模块

# 设置字体为Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 创建数据集
data = {
    'dataset#3': [1.130, 0.960, 1.000, 1.420, 2.080, 2.610, 3.550, 3.330, 3.430, 3.400, 3.150, 3.710, 3.800, 3.800, 3.730, 3.600, 2.290, 53.000],
    'dataset#4': [1.960, 2.510, 2.600, 2.770, 3.000, 3.140, 3.340, 3.580, 3.730, 3.660, 3.680, 3.640, 3.570, 3.440, 3.300, 3.060, 0.950, 48.080]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置柱状图的样式和设置
plt.figure(figsize=(10, 6))  # 设置图形大小
ax = df.plot(kind='bar', fontsize=22)  # 绘制柱状图并设置字体大小为16号
plt.xlabel('Wrap-count', fontsize=22)  # 设置X轴标签和字体大小
plt.ylabel('Proportion (%)', fontsize=22)  # 设置Y轴标签和字体大小
plt.ylim(0, 60)  # 设置Y轴坐标范围

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
