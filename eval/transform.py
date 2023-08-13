# Code with transformations for Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch

from PIL import Image

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([139, 119, 101])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([0, 139, 139])
    cmap[3, :] = np.array([125, 38, 205])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])
    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([255, 250, 205])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])
    cmap[11, :] = np.array([0, 255, 255])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([205, 129, 98])
    cmap[14, :] = np.array([0, 191, 255])
    cmap[15, :] = np.array([255, 62, 150])
    cmap[16, :] = np.array([139, 0, 0])
    cmap[17, :] = np.array([255, 255, 255])
    # cmap[0, :] = np.array([255, 255, 255])
    # cmap[1, :] = np.array([139, 119, 101])
    # cmap[2, :] = np.array([244, 35, 232])
    # cmap[3 :] = np.array([0, 139, 139])
    # cmap[4, :] = np.array([125, 38, 205])
    # cmap[5, :] = np.array([190, 153, 153])
    # cmap[6, :] = np.array([153, 153, 153])
    # cmap[7, :] = np.array([250, 170, 30])
    # cmap[9, :] = np.array([220, 220, 0])
    # cmap[9, :] = np.array([255, 250, 205])
    # cmap[10, :] = np.array([152, 251, 152])
    # cmap[11, :] = np.array([70, 130, 180])
    # cmap[12, :] = np.array([0, 255, 255])
    # cmap[13, :] = np.array([255, 0, 0])
    # cmap[14, :] = np.array([205, 129, 98])
    # cmap[15, :] = np.array([0, 191, 255])
    # cmap[16, :] = np.array([255, 62, 150])
    # cmap[16, :] = np.array([139, 0, 0])
    # cmap[17, :] = np.array([106, 168, 79])
    # cmap[18, :] = np.array([166, 77, 121])
    # cmap[19, :] = np.array([127, 95, 0])
    # cmap[20, :] = np.array([61, 134, 198])
    # cmap[21, :] = np.array([180, 167, 214])
    # cmap[22, :] = np.array([76, 17, 48])
    # cmap[23, :] = np.array([97, 220, 0])
    # cmap[24, :] = np.array([39, 0, 184])
    # cmap[25, :] = np.array([49, 80,200])
    # cmap[26, :] = np.array([175, 70, 100])
    # cmap[27, :] = np.array([149, 80, 0])
    # cmap[28, :] = np.array([234, 0, 85])
    # cmap[29, :] = np.array([176, 84, 24])
    # cmap[30, :] = np.array([75, 240, 80])
    # cmap[31, :] = np.array([166, 0, 60])
    # cmap[32, :] = np.array([188, 0, 88])


    # cmap[17, :] = np.array([255, 255, 255])



    return cmap

def colormap_error(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([255, 255, 255])
    cmap[1, :] = np.array([0, 0, 0])
    cmap[2, :] = np.array([0, 0, 0])
    cmap[3, :] = np.array([0, 0, 0])
    cmap[4, :] = np.array([0, 0, 0])
    cmap[5, :] = np.array([0, 0, 0])
    cmap[6, :] = np.array([0, 0, 0])
    cmap[7, :] = np.array([0, 0, 0])
    cmap[8, :] = np.array([0, 0, 0])
    cmap[9, :] = np.array([0, 0, 0])
    cmap[10, :] = np.array([0, 0, 0])
    cmap[11, :] = np.array([0, 0, 0])
    cmap[12, :] = np.array([0, 0, 0])
    cmap[13, :] = np.array([0, 0, 0])
    cmap[14, :] = np.array([0, 0, 0])
    cmap[15, :] = np.array([0, 0, 0])
    cmap[16, :] = np.array([0, 0, 0])
    cmap[17, :] = np.array([255, 255, 255])
    # cmap[0, :] = np.array([255, 255, 255])
    # cmap[1, :] = np.array([139, 119, 101])
    # cmap[2, :] = np.array([244, 35, 232])
    # cmap[3 :] = np.array([0, 139, 139])
    # cmap[4, :] = np.array([125, 38, 205])
    # cmap[5, :] = np.array([190, 153, 153])
    # cmap[6, :] = np.array([153, 153, 153])
    # cmap[7, :] = np.array([250, 170, 30])
    # cmap[9, :] = np.array([220, 220, 0])
    # cmap[9, :] = np.array([255, 250, 205])
    # cmap[10, :] = np.array([152, 251, 152])
    # cmap[11, :] = np.array([70, 130, 180])
    # cmap[12, :] = np.array([0, 255, 255])
    # cmap[13, :] = np.array([255, 0, 0])
    # cmap[14, :] = np.array([205, 129, 98])
    # cmap[15, :] = np.array([0, 191, 255])
    # cmap[16, :] = np.array([255, 62, 150])
    # cmap[16, :] = np.array([139, 0, 0])
    # cmap[17, :] = np.array([106, 168, 79])
    # cmap[18, :] = np.array([166, 77, 121])
    # cmap[19, :] = np.array([127, 95, 0])
    # cmap[20, :] = np.array([61, 134, 198])
    # cmap[21, :] = np.array([180, 167, 214])
    # cmap[22, :] = np.array([76, 17, 48])
    # cmap[23, :] = np.array([97, 220, 0])
    # cmap[24, :] = np.array([39, 0, 184])
    # cmap[25, :] = np.array([49, 80,200])
    # cmap[26, :] = np.array([175, 70, 100])
    # cmap[27, :] = np.array([149, 80, 0])
    # cmap[28, :] = np.array([234, 0, 85])
    # cmap[29, :] = np.array([176, 84, 24])
    # cmap[30, :] = np.array([75, 240, 80])
    # cmap[31, :] = np.array([166, 0, 60])
    # cmap[32, :] = np.array([188, 0, 88])


    # cmap[17, :] = np.array([255, 255, 255])



    return cmap

def colormap_blender(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([139, 119, 101])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([0, 139, 139])
    cmap[3, :] = np.array([125, 38, 205])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])
    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([255, 250, 205])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])
    cmap[11, :] = np.array([0, 255, 255])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([205, 129, 98])
    cmap[14, :] = np.array([0, 191, 255])
    cmap[15, :] = np.array([255, 62, 150])
    cmap[16, :] = np.array([139, 0, 0])
    cmap[17, :] = np.array([255, 255, 255])
    return  cmap
def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=33):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

class Colorizeerror:

    def __init__(self, n=33):
        #self.cmap = colormap(256)
        self.cmap = colormap_error(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
class Colorizeblender:

    def __init__(self, n=33):
        #self.cmap = colormap(256)
        self.cmap = colormap_blender(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image