#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author's_name_is_NIKOLA_SS
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import matplotlib.image as mpimg
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False #防止乱码
if __name__ == '__main__':
    # def f(t):
    #     return np.exp(t) * np.cos(2 * np.pi * t)
    #
    #
    # t1 = np.arange(0.0, 5.0, 0.02)  # 初始化数据
    #
    # ax1 = plt.subplot(3,3,4)
    #
    # ax1.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    # # ax1.set_title('输出图')
    # plt.axis('off')
    # ax11 = plt.subplot(2, 2, 2)
    # ax11.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    # plt.axis('off')
    #
    # ax12 = plt.subplot(2, 2, 3)
    # ax12.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    # plt.axis('off')
    #
    # ax13 = plt.subplot(2, 2, 4)
    # ax13.imshow(mpimg.imread('H:/real_pic/erfnet_expri_color/gt/test/C0001/C0001_000421_000019_leftImg8bit.png'))
    # plt.axis('off')

    # ax2 = plt.subplot(13,8,1)
    #
    # ax2.plot(t1, f(t1), 'r')
    # # ax2.set_title('输入图1')
    def rgb_to_hex(r, g, b):
        return ('#' + '{:02X}' * 3).format(r, g, b)


    # fig, ax = plt.subplots(3,3,figsize=(10, 10))
    # fig, axes = plt.subplots(3, 3)
    # fig.subplots_adjust(bottom=0.5)
    colors = [(139, 119, 101), [244, 35, 232], [0, 139, 139], [125, 38, 205], [190, 153, 153], [153, 153, 153]
        , [250, 170, 30], [220, 220, 0], [255, 250, 205], [152, 251, 152], [70, 130, 180], [0, 255, 255], [255, 0, 0],
              [205, 129, 98], [0, 191, 255], [255, 62, 150],[139, 0, 0],[255, 255, 255]]
    for i in range(len(colors)):
        colors[i] = rgb_to_hex(colors[i][0], colors[i][1], colors[i][2])

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    fig2, axes2 = plt.subplots(2, 2)
    # cmap = mpl.cm.cool
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=17)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Fringe Orders')
    # fig, ax = plt.subplots(figsize=(1, 1))
    # ax3 = plt.subplot(442)
    # cmap = mpl.colors.ListedColormap(colors)
    # norm = mpl.colors.Normalize(vmin=1, vmax=16)
    #
    # cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='horizontal')
    #
    # ax3.plot(t1, f(t1), 'b')
    # # ax3.set_title('输入图2')

    plt.show()


