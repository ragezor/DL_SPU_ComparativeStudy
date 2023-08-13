import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

import sys
sys.path.append("..")

from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

NUM_CLASSES = 17

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    ToTensor(),
])
target_transform_cityscapes = Compose([
    ToLabel(),
    Relabel(255, 16),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    model.eval()
    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    i =0
    time_train = []

    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(loader):

            torch.cuda.synchronize()
            start_time = time.time()
            inputs = Variable(images.cuda())

            outputs = model(inputs)[0].max(0)[1].byte()
            torch.cuda.synchronize()
            label = outputs.cpu().data
            fwt = time.time() - start_time
            if i != 0:  # first run always takes some time for setup
                time_train.append(fwt)
                print("Forward time per img (b=%d): %.6f (Mean: %.6f)" % (
                    args.batch_size, fwt / args.batch_size, sum(time_train) / len(time_train) / args.batch_size))

            i = i + 1
        print("fps(nobatch)"+str(1/(sum(time_train)/1000)))
        print(str(sum(time_train)))
if __name__ == '__main__':
    parser = ArgumentParser()



    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="D:/2022_1_5_work/work/blender/100blender/result/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="test")  # can be val, test, train, demoSequence
    parser.add_argument('--datadir', default="D:/2022_1_5_work/work/blender_data/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true',default=False)

    main(parser.parse_args())
