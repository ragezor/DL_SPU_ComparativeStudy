import math
import os
import  numpy as np
import  cv2
     #矩阵转图片使用
import matplotlib.image
from PIL import Image
g = os.walk("D:/2022_1_5_work/work/blender_data/input/train")
# os.makedirs(r'D:/2022_1_5_work/work/blender_data/mask/train')
# os.makedirs(r'D:/2022_1_5_work/work/blender_data/wary/train')
e=os.walk("D:/2022_1_5_work/work/blender_data/input/eval")
# os.makedirs(r'D:/2022_1_5_work/work/blender_data/mask/eval')
# os.makedirs(r'D:/2022_1_5_work/work/blender_data/wary/eval')
# for path, d, filelist in g:
#     for filename in filelist:
#         if filename.endswith('png'):
#             filead=os.path.join(path, filename)
#             image=Image.open(filead).convert('L')
#             image_np=np.array(image)
#
#             # new_im = Image.fromarray(w_x_new)
#             # new_im.show()
#
#             size_x=image_np.shape[0]
#             size_y=image_np.shape[1]
#             mask = np.ones((size_x, size_y))*255
#             for i in range(size_x):
#                 for j in range(size_y):
#                     ele=image_np[i][j]
#                     if ele==0:
#                         mask[i][j]=0
#
#
#             w_x_name=filename.replace('warpped','mask')
#
#             cv2.imwrite("D:/2022_1_5_work/work/blender_data/mask/train/{}".format(w_x_name), mask)
#

            #cv2.imwrite('warx01.png', w_x_new)  #使用cv2实现图片与numpy数组的相互转化
for path, d, filelist in e:
    for filename in filelist:
        if filename.endswith('png'):
            filead = os.path.join(path, filename)
            image = Image.open(filead).convert('L')
            image_np = np.array(image)

            # new_im = Image.fromarray(w_x_new)
            # new_im.show()

            size_x = image_np.shape[0]
            size_y = image_np.shape[1]
            mask = np.ones((size_x, size_y))*255
            for i in range(size_x):
                for j in range(size_y):
                    ele = image_np[i][j]
                    if ele == 0:
                        mask[i][j] = 0

            w_x_name = filename.replace('warpped', 'mask')

            cv2.imwrite("D:/2022_1_5_work/work/blender_data/mask/eval/{}".format(w_x_name), mask)



