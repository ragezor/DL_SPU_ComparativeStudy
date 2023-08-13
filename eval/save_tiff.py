import os
import matplotlib.pyplot as plt
import numpy as np

root_folder = r"G:\metric_map"
output_folder = r"G:\zhuzhuangtu"

network_names_simu = ["erfnet", "myloss", "REDN", "tiramisu", "VuRNet"]
network_names_real = ["erfnet", "REDN", "tiramisu", "VuRNet"]
datasets_real = ["blender"]
datasets_simu = ["blender", "gaus", "zer"]

indicators = {
    "mIoU": [("real", "mIoU"), ("simu", "mIoU")],
    "SSIM": [("real", "SSIM"), ("simu", "SSIM")],
    "RMSE": [("real", "RMSE"), ("simu", "RMSE")]
}

# 创建保存图像的文件夹
os.makedirs(output_folder, exist_ok=True)

for indicator, paths in indicators.items():
    for path_type, indicator_path in paths:
        if path_type == "real":
            network_names = network_names_real
            datasets = datasets_real
        elif path_type == "simu":
            network_names = network_names_simu
            datasets = datasets_simu

        for network_name in network_names:
            print(network_name)
            # 跳过指定条件下的文件夹
            if indicator == "mIoU" and network_name == "VuRNet":
                continue


            for dataset in datasets:
                if path_type == "simu" and dataset == "blender" and network_name == "myloss":
                    continue
                file_path = os.path.join(root_folder, path_type, indicator_path, f"output{network_name}{dataset}.txt")
                print(file_path)

                # 读取文件并提取指标值
                values = []
                with open(file_path, "r") as file:
                    for line in file:
                        value = line.strip()
                        if indicator == "mIoU" or indicator == "SSIM":
                            value = float(value.strip('%'))
                        else:
                            value = float(value)
                        values.append(value)

                # 根据指标范围设置横坐标刻度
                if indicator == "mIoU":
                    x_ticks = np.linspace(0, 100, 10)
                    xlabel = "mIoU (%)"
                elif indicator == "SSIM":
                    x_ticks = np.linspace(0, 1, 10)
                    xlabel = "SSIM"
                elif indicator == "RMSE":
                    x_ticks = np.linspace(0, 20, 100)
                    xlabel = "RMSE"

                # 统计指标在每个范围内的数量
                hist, bins = np.histogram(values, bins=x_ticks)

                # 计算每个范围内指标的比例
                ratio = hist / len(values) * 100

                # 绘制柱状图
                plt.bar(x_ticks[:-1], ratio, width=(x_ticks[1] - x_ticks[0]))
                plt.xlabel(xlabel, fontsize=16)  # 设置横坐标字体大小
                plt.ylabel('Proportion (%)', fontsize=16)  # 设置纵坐标字体大小
                plt.xticks(fontsize=12)  # 设置刻度标签字体大小
                plt.yticks(fontsize=12)  # 设置刻度标签字体大小
                plt.ylim(0, 100)  # 设置纵坐标范围为0到100

                # 设置边框粗细
                ax = plt.gca()
                # ax.spines['top'].set_linewidth(2)  # 上边框粗细
                # ax.spines['right'].set_linewidth(2)  # 右边框粗细
                # ax.spines['bottom'].set_linewidth(2)  # 下边框粗细
                # ax.spines['left'].set_linewidth(2)  # 左边框粗细

                # 提取文件路径和文件名，用于保存图像
                file_dir, file_name = os.path.split(file_path)
                file_dir = file_dir.split(os.sep)[-3:]
                file_dir = "_".join(file_dir)
                file_name = file_name.split(".")[0]
                save_path = os.path.join(output_folder, f"{file_dir}_{file_name}.svg")

                # 保存图像为.svg格式
                plt.savefig(save_path, format='svg')

                # 清除图形以便下次绘制
                plt.clf()

               


                print(f"Saved: {save_path}")
