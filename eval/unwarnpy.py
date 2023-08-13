import math

import numpy as np
import torch
import os
# import importlib
# import imageio
# from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from  DLPU import  DLPU
from  PhUn import  PhUn
from VuRNet import  Net
import sys
sys.path.append("../..")

from transform import Relabel, ToLabel, Colorize
from  dataset import  cityscapes
from  HiPhase import  HiPhaseNet
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

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model =Net()

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    loader = DataLoader(
        cityscapes (args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        targets=Variable(labels)
        with torch.no_grad():
            # outputs = model(inputs)
            #outputs = F.interpolate(outputs, size=1024, mode='bilinear', align_corners=False)
            test_preds = model(inputs).detach()
            inputs = inputs * torch.tensor(2 * math.pi) - torch.tensor(math.pi)
            targets = targets[0][0]
            inputs = inputs[0][0]
            # mask1=targets!=17


            #test_preds = F.interpolate(test_preds, size=256, mode='bilinear', align_corners=False)
            # test_preds1 = test_preds[0].max(0)[1].byte().data
            # test_preds2 = inputs + test_preds1 * 2 * math.pi
            targets1 = inputs + targets * 2 * math.pi
            # targets1=targets1*mask1
            # test_preds2=test_preds2*mask1
            # mask2=targets1>0
            # mask3=test_preds2>0
            # mask_final=mask3*mask2
            targets1=targets1
            test_preds2=test_preds
            order = torch.round(abs(test_preds2 - inputs)/(math.pi*2))
            test_preds2 = order * torch.tensor(2 * math.pi) + inputs
            targets1 = targets1.cpu()
            test_preds2 = test_preds2.cpu()
            test_preds2=test_preds2[0][0]
            # print(test_preds2.size())
            sub=targets1-test_preds2

            # np.save('gtun.npy', targets1)
            # np.save('preun_erfnet.npy', test_preds2)
        # label = outputs[0].max(0)[1].byte().cpu().data

        # label=label.numpy()
        # np.save("label.npy",label)
        filenameSave = "E:/simu_un/vur_expri_gray/gaus/pre/" + filename[0].split("input/")[1]
        filenameSave=filenameSave[:-4]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        # label = test_preds2.cpu().numpy()
        # label_save = Image.fromarray(np.uint8(np.round(label)))
        # label_save = Image.fromarray(label)
        # label_save=label_save.convert('RGB')
        np.save(filenameSave+'.npy', test_preds2)

        # label_save.save(filenameSave)


        filenameSave = "E:/simu_un/vur_expri_gray/gaus/gt/" + filename[0].split("input/")[1]
        filenameSave = filenameSave[:-4]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        gt = targets1.cpu().numpy()
        # # label_save = Image.fromarray(np.uint8(np.round(label)))
        np.save(filenameSave+'.npy', gt)
        # gt_save.save(filenameSave)



        print(step, filenameSave)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadDir', default="H:/2022_1_5_work/new_vurnet/gaus/100gaus/result/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="VuRNet.py")
    parser.add_argument('--subset', default="test")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="D:/2022_1_5_work/work/new_gaus/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true',default=False)

    main(parser.parse_args())

