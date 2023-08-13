import os
from svgutils.compose import SVG
from svgutils.transform import fromstring

folder_path = r"G:\zhuzhuangtu"
output_path = r"G:\zhuzhuangtu\combined_plot.svg"

# 获取文件夹中的所有文件
files = [file for file in os.listdir(folder_path) if file.endswith(".svg")]

# 创建一个新的SVG对象
combined_svg = SVG()

# 遍历文件并将它们添加到组合的SVG对象中
for i, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r') as f:
        svg_data = f.read()
    svg = fromstring(svg_data)
    combined_svg.append(svg)

# 保存合并后的SVG图像
combined_svg.save(output_path)
