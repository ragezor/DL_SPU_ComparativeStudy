import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
    return filename.startswith("order")
def is_warx(filename):
    return filename.startswith("warx")
def is_wary(filename):
    return filename.startswith("wary")


def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')


def image_path_city(root, name):
    return os.path.join(root, f'{name}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'input')
        self.labels_root = os.path.join(root, 'label')
        # self.warx_root=os.path.join(root,'warx')
        # self.wary_root=os.path.join(root,'wary')

        self.filenames = [image_basename(f)
                          for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform
        # self.warx_transform=input_transform
        # self.wary_transform=input_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('L')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('L')
        # with open(image_path(self.warx_root_root, filename, '.png'), 'rb') as f:
        #     warx= load_image(f).convert('L')
        # with open(image_path(self.wary_root_root, filename, '.png'), 'rb') as f:
        #     wary = load_image(f).convert('L')

        if self.input_transform is not None:
            image = self.input_transform(image)
            # warx=self.input_transform(warx)
            # wary = self.input_transform(wary)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # return image, label,warx,wary

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'input/')
        self.labels_root = os.path.join(root, 'label/')
        # self.warx_root=os.path.join(root,'warx/')
        # self.wary_root=os.path.join(root,'wary/')

        self.images_root += subset
        self.labels_root += subset
        # self.wary_root+=subset
        # self.warx_root+=subset

        print(self.images_root)
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in
                          fn if is_image(f)]
        self.filenames.sort()

        # [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        # self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in
                            fn if is_label(f)]
        self.filenamesGt.sort()

        # self.filenameswx = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.warx_root)) for f in
        #                     fn if is_image(f)]
        # self.filenameswx.sort()
        #
        # self.filenameswy = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.wary_root)) for f in
        #                     fn if is_image(f)]
        # self.filenameswy.sort()

        self.co_transform = co_transform  # ADDED THIS

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        # filenamewx=self.filenameswx[index]
        # filenamewy=self.filenameswy[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('L')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('L')
        # with open(image_path_city(self.warx_root, filenamewx), 'rb') as f:
        #     warx = load_image(f).convert('L')
        # with open(image_path_city(self.wary_root, filenamewy), 'rb') as f:
        #     wary = load_image(f).convert('L')

        if self.co_transform is not None:
            # image, label,warx,wary = self.co_transform(image, label,warx,wary)
            image, label = self.co_transform(image, label)
        # print(1)
        return image, label

        # return image, label, warx, wary

    def __len__(self):
        return len(self.filenames)

