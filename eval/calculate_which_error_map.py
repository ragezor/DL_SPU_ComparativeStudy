import numpy as np

# 自定义变量 mean_metric
mean_metric = 7.67


# 文件路径
file_path = r'G:\metric_map\RMSE\outputVuRNetblender.txt'

# 读取文件数据
with open(file_path, 'r') as file:
    lines = file.readlines()

# 将数据转换为浮点数列表
data = [float(line.strip()) for line in lines]

# 计算与 mean_metric 的差值，并取绝对值
diff = np.abs(np.array(data) - mean_metric)

# 获取与 mean_metric 最接近的三个值的索引
indices = np.argsort(diff)[:3]

# 获取对应的行数（从1开始）
line_numbers = [index + 1 for index in indices]

# 输出结果
for i in range(3):
    closest_value = data[indices[i]]
    print(f"最接近的值: {closest_value}, 行数: {line_numbers[i]}")
