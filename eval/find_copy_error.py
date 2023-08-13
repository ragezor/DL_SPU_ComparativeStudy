import os
import shutil

# 文件夹根目录
root_folder = r'G:\error map\simu'

# 图片根目录
image_root_folder = r'D:\all_error_pic'

# 遍历根目录下的子文件夹
for sub_folder in os.listdir(root_folder):
    sub_folder_path = os.path.join(root_folder, sub_folder)
    if not os.path.isdir(sub_folder_path):
        continue

    # 遍历每个子文件夹下的txt文件
    for txt_file in os.listdir(sub_folder_path):
        if not txt_file.endswith('.txt'):
            continue

        txt_file_path = os.path.join(sub_folder_path, txt_file)
        network_name = txt_file.replace('.txt', '')

        # 获取txt文件中的行数信息
        line_numbers = []
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 2 and '行数' in parts[1]:
                    line_number = int(parts[1].strip().split(':')[1].strip())
                    line_numbers.append(line_number)

        # 构造对应的图片目录路径
        image_folder = os.path.join(image_root_folder, f'{network_name}._blender_test')

        # 创建目标目录
        target_folder = os.path.join(sub_folder_path, network_name)
        os.makedirs(target_folder, exist_ok=True)

        # 复制图片
        for line_number in line_numbers:
            image_path = os.path.join(image_folder, f'{line_number}.png')
            print(image_path)
            if os.path.exists(image_path):
                target_image_path = os.path.join(target_folder, f'{line_number}.png')
                shutil.copy(image_path, target_image_path)

