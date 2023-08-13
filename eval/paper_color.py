import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import numpy as np
from boto import sns

from  transform import  colormap_cityscapes

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    # def rgb_to_hex(r, g, b):
    #     return ('#' + '{:02X}' * 3).format(r, g, b)
    #
    #
    # # fig, ax = plt.subplots(3,3,figsize=(10, 10))
    # # fig, axes = plt.subplots(3, 3)
    # # fig.subplots_adjust(bottom=0.5)
    # colors = [(139, 119, 101), [244, 35, 232], [0, 139, 139], [125, 38, 205], [190, 153, 153], [153, 153, 153]
    #     , [250, 170, 30], [220, 220, 0], [255, 250, 205], [152, 251, 152], [70, 130, 180], [0, 255, 255], [255, 0, 0],
    #           [205, 129, 98], [0, 191, 255], [255, 62, 150]]
    # for i in range(len(colors)):
    #     colors[i] = rgb_to_hex(colors[i][0], colors[i][1], colors[i][2])
    #
    # cmap = mpl.colors.ListedColormap(colors)
    # norm = mpl.colors.Normalize(vmin=1, vmax=16)
    # # # fig, ax = plt.subplots(figsize=(3,4))
    # # # fig.subplots_adjust(bottom=0.5)
    # # plt.figure(figsize=(6, 6))
    #
    # # # plt.subplot(121)
    # fig, ax = plt.subplots(figsize=(6, 1))
    # fig.subplots_adjust(bottom=0.5)
    # fig2, axes2 = plt.subplots(2, 2)
    # # cmap = mpl.cm.cool
    # norm = mpl.colors.Normalize(vmin=1, vmax=16)
    #
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='horizontal')
    # cb1.set_label('Fringe Orders')
    # fig, ax = plt.subplots(figsize=(1, 1))

    plt.subplot(2,2,1)
    plt.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000588_000019_leftImg8bit.png'))
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000766_000019_leftImg8bit.png'))
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_001081_000019_leftImg8bit.png'))

    plt.axis('off')
    # plt.subplot(122)
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='horizontal')

   # im1 = ax[0][0].imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    #im2 = ax[0][1].imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    #
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='horizontal')
    # cb1.set_label('Some Units')
    # fig, ax = plt.subplots(figsize=(1, 1))
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm)

    plt.show()