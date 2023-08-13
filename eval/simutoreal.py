import os
import cv2

# 指定输入和输出文件夹路径
input_folder = r'H:\2022_1_5_work\work\new_blender\input\test'
output_folder = r'H:\2022_1_5_work\work\simuforreal\input\test'

# 获取输入文件夹下所有图片文件
image_files = [file for file in os.listdir(input_folder) if file.endswith('.png')]  # 根据实际情况修改文件扩展名

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 循环处理每张图片
for image_file in image_files:
    # 读取图片
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)

    # 将像素值为0的点改为255
    img[img == 0] = 255

    # 构建输出文件路径
    filename, ext = os.path.splitext(image_file)
    output_path = os.path.join(output_folder, filename + ext)

    # 保存新图片
    cv2.imwrite(output_path, img)
