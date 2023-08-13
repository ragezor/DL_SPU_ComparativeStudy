from PIL import Image
import os

def process_images(source_dir, target_dir):
    # 获取source_dir下所有的png文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                if "label" in root:  # 只处理label文件夹下的PNG文件
                    process_label_image(file_path, target_dir)
                else:  # 复制input文件夹下的PNG文件到目标文件夹
                    process_input_image(file_path, target_dir)

def process_input_image(file_path, target_dir):
    # 直接复制input文件夹下的PNG文件到目标文件夹
    save_path = os.path.join(target_dir, "input", os.path.relpath(file_path, start=source_dir))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(file_path, "rb") as f_src, open(save_path, "wb") as f_dst:
        f_dst.write(f_src.read())

def process_label_image(file_path, target_dir):
    # 读取图像
    image = Image.open(file_path)
    # 获取图像数据
    data = image.load()
    # 将17像素点转换为255
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if data[x, y] == 17:
                data[x, y] = 255
    # 获取保存路径
    save_path = os.path.join(target_dir, "label", os.path.relpath(file_path, start=source_dir))
    # 确保保存路径所在的文件夹存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存图像
    image.save(save_path)



if __name__ == "__main__":
    source_dir = "H:/2022_1_5_work/work/new_blender/"
    target_dir = "G:/Dataset/#3"
    process_images(source_dir, target_dir)
