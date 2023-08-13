# # Code to produce colored segmentation output in Pytorch for all cityscapes subsets
# # Sept 2017
# # Eduardo Romera
# #######################
# import time
#
# import numpy as np
# import torch
# import os
# import importlib
#
# from PIL import Image
# from argparse import ArgumentParser
#
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
# from torchvision.transforms import ToTensor, ToPILImage
#
# from dataset import cityscapes
# from erfnet import ERFNet
# from transform import Relabel, ToLabel, Colorize
#
# import visdom
#
#
# NUM_CHANNELS = 1
# NUM_CLASSES = 17
#
#
# image_transform = ToPILImage()
# input_transform_cityscapes = Compose([
#     Resize((256,256),Image.BILINEAR),
#     ToTensor(),
#     #Normalize([.485, .456, .406], [.229, .224, .225]),
# ])
# target_transform_cityscapes = Compose([
#     Resize((256,256),Image.NEAREST),
#     ToLabel(),
#     Relabel(255, 16),   #ignore label to 16
# ])
#
# # cityscapes_trainIds2labelIds = Compose([
# #     # Relabel(19, 255),
# #     # Relabel(18, 33),
# #     # Relabel(17, 32),
# #     Relabel(16, 31),
# #     Relabel(15, 28),
# #     Relabel(14, 27),
# #     Relabel(13, 26),
# #     Relabel(12, 25),
# #     Relabel(11, 24),
# #     Relabel(10, 23),
# #     Relabel(9, 22),
# #     Relabel(8, 21),
# #     Relabel(7, 20),
# #     Relabel(6, 19),
# #     Relabel(5, 17),
# #     Relabel(4, 13),
# #     Relabel(3, 12),
# #     Relabel(2, 11),
# #     Relabel(1, 8),
# #     Relabel(0, 7),
# #     Relabel(255, 0),
# #     ToPILImage(),
# # ])
#
# def main(args):
#
#
#     modelpath = args.loadDir + args.loadModel
#     weightspath = args.loadDir + args.loadWeights
#
#     print ("Loading model: " + modelpath)
#     print ("Loading weights: " + weightspath)
#
#     #Import ERFNet model from the folder
#     #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
#     model = ERFNet(NUM_CLASSES)
#
#     model = torch.nn.DataParallel(model)
#     if (not args.cpu):
#         model = model.cuda()
#
#     #model.load_state_dict(torch.load(args.state))
#     #model.load_state_dict(torch.load(weightspath)) #not working if missing key
#
#     def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
#         own_state = model.state_dict()
#         for name, param in state_dict.items():
#             if name not in own_state:
#                  continue
#             own_state[name].copy_(param)
#         return model
#
#     model = load_my_state_dict(model, torch.load(weightspath))
#     print ("Model and weights LOADED successfully")
#
#     model.eval()
#
#     if(not os.path.exists(args.datadir)):
#         print ("Error: datadir could not be loaded")
#
#
#     loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
#         num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
#     # loader_label=DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
#     #     num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
#     print(loader)
#
#
#     # For visualizer:
#     # must launch in other window "python3.6 -m visdom.server -port 8097"
#     # and access localhost:8097 to see it
#
#
#
#     if (args.visualize):
#         vis = visdom.Visdom()
#     time_res=0
#     for step, (images, labels, filename, filenameGt) in enumerate(loader):
#
#         print(1)
#
#
#
#
#         images = images.cuda()
#
#         labels = labels.cuda()
#         time_start = time.time();
#         inputs = Variable(images)
#         # targets = Variable(labels).float()
#
#         with torch.no_grad():
#
#             outputs = model(inputs)
#             time_end = time.time();
#             time_res = time_res + time_end - time_start
#             # outputs2=model(targets.float())
#
#
#         label = outputs[0].max(0)[1].byte().cpu().data
#         labels = labels.squeeze(0).squeeze(0).byte().cpu().data
#
#
#         label2=labels-label
#         labels_num=label2.numpy()
#         tem=labels_num*labels_num
#
#         # sub2=sum(sum(abs(label2)).int()).int();
#         # sub=sub+sub2
#
#
#         # label2 = targets[0].max(0)[1].byte().cpu().data
#         # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
#         label_color = Colorize()(label.unsqueeze(0))
#         label_color2 = Colorize()(labels.unsqueeze(0))
#         label_color3 = Colorize()(label2.unsqueeze(0))
#
#
#
#         # print("filename start")
#         filenameSave = "D:/2022_1_5_work/work/metric/gaus/cre/estimate/" + "cre_estimate"+filename[0].split("input")[1]
#         filenameGtSave = "D:/2022_1_5_work/work/metric/gaus/cre/True/" +"cre_true"+ filenameGt[0].split("label")[1]
#         filenameSubGtSave = "D:/2022_1_5_work/work/metric/gaus/cre/sub/" + "cre_sub"+filenameGt[0].split("label")[1]
#
#         #filenameSubGrayGtSave = "D:/2022_1_5_work/work/result//zer/10zer/save_result/sub/gray/" + filenameGt[0].split("label")[1]
#         # print("filename done")
#         os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
#         #image_transform(label.byte()).save(filenameSave)
#         label_save = ToPILImage()(label_color)
#         label_save.save(filenameSave)
#         os.makedirs(os.path.dirname(filenameGtSave), exist_ok=True)
#         #image_transform(label.byte()).save(filenameSave)
#         label_save2 = ToPILImage()(label_color2)
#         label_save2.save(filenameGtSave)
#         os.makedirs(os.path.dirname(filenameSubGtSave), exist_ok=True)
#         label_save3 = ToPILImage()(label_color3)
#         label_save3.save(filenameSubGtSave)
#         # os.makedirs(os.path.dirname(filenameSubGrayGtSave), exist_ok=True)
#         # label_save3 = ToPILImage()(label2)
#         # label_save3.save(filenameSubGrayGtSave)
#
#
#         if (args.visualize):
#             vis.image(label_color.numpy())
#         print (step, filenameSave)
#
#
#
#
#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#
#     parser.add_argument('--state')
#
#     parser.add_argument('--loadDir',default="D:/2022_1_5_work/erfnet_metric/gaus/100gaus/")
#     parser.add_argument('--loadWeights', default="modelcre_best.pth")
#     parser.add_argument('--loadModel', default="erfnet.py")
#     parser.add_argument('--subset',default="test")  #can be val, test, train, demoSequence
#
#     parser.add_argument('--datadir',type=str, default= "D:/2022_1_5_work/work/new_gaus/")
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--batch-size', type=int, default=100)
#     parser.add_argument('--cpu', action='store_true',default=False)
#
#     parser.add_argument('--visualize', action='store_true',default=False)
#     main(parser.parse_args())

import math

import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from transform import Relabel, ToLabel, Colorize, Colorizeerror
from  REDN import  frrn
from  EESANet import  EESANet
from  erfnet import  ERFNet
from  dataset import  cityscapes
from erfnet_1cha import  ERFNet1cha
from  HiPhase import HiPhaseNet
NUM_CLASSES = 17
from  tiramisu import  PhaseNet2

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    ToTensor(),
])
target_transform_cityscapes = Compose([
    ToLabel(),
    Relabel(255, 16),
])

def main(args):
    min_val = math.inf
    max_val = 0
    max_name = ''
    min_name = ''
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    # model = ERFNet(NUM_CLASSES)
    # model=ERFNet1cha()
    model=EESANet(1,48,NUM_CLASSES)
    # model=PhaseNet2(NUM_CLASSES)

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

    # loader = DataLoader(
    #     dentalphase(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
    #     num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
    num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)
            outputs=torch.round(outputs)
            #metric_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
           # mse_loss = metric_loss(outputs.max(1)[1].unsqueeze(1).data, torch.tensor(labels, dtype=torch.float32))
            # if mse_loss < min_val:
            #     min_val = mse_loss
            #     min_name = filename[0]
            # if mse_loss > max_val:
            #     max_val = mse_loss
            #     max_name = filename[0]

        # label = outputs[0][0].byte().cpu().data
        label = outputs[0].max(0)[1].byte().cpu().data
        # label = outputs[0].max(0)[1].byte().cpu().data
        labels = labels.squeeze(0).squeeze(0).byte().cpu().data
        cha=labels.sub(label)
        # cha=cha*3
        # new_label=labels-cha
        sub=abs(labels.sub(label))
        # abscha=abs(cha)

        # label_color = Colorize()(label.unsqueeze(0))
        # gt_color=Colorize()(labels.unsqueeze(0))
        sub_color=Colorizeerror()(sub.unsqueeze(0))
        # cha_color=Colorize()(abscha.unsqueeze(0))
        # newlabel_color=Colorize()(new_label.unsqueeze(0))
        metric="cre"
        # filenameSave = "D:/simu_data/phasenet2/blender/"+metric+"/pre/" +filename[0].split("input/")[1][:-4]+"pre_"+metric+".png"
        # gtSave = "D:/simu_data/phasenet2/blender/"+metric+"/gt/" +filename[0].split("input/")[1][:-4]+"gt_"+metric+".png"
        subSave = "D:/simu_data/error/ernet_noise/gaus/"+metric+"/sub/" +filename[0].split("input/")[1][:-4]+ "sub_"+metric+".png"
        #chaSave="D:/simu_data/phasenet2/blender/"+metric+"/cha/" +filename[0].split("input/")[1][:-4]+ "sub_"+metric+".png"
        #newPreSave="D:/simu_data/phasenet2/blender/"+metric+"/newpre/" +filename[0].split("input/")[1][:-4]+ "sub_"+metric+".png"
        # os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        # label_save = ToPILImage()(label_color)
        # label_save.save(filenameSave)
        #
        # os.makedirs(os.path.dirname(gtSave), exist_ok=True)
        # gt_save = ToPILImage()(gt_color)
        # gt_save.save(gtSave)

        os.makedirs(os.path.dirname(subSave), exist_ok=True)
        sub_save = ToPILImage()(sub_color)
        sub_save.save(subSave)

        # os.makedirs(os.path.dirname(chaSave), exist_ok=True)
        # sub_save = ToPILImage()(cha_color)
        # sub_save.save(chaSave)
        #
        # os.makedirs(os.path.dirname(newPreSave), exist_ok=True)
        # sub_save = ToPILImage()(newlabel_color)
        # sub_save.save(newPreSave)
        print(step, subSave)
    # file = open("D:/simu_data/metric/erfnet/BestAndWorst.txt", 'w')
    #
    # file.write("worst:"+max_name+",        best:"+min_name)  # msg也就是下面的Hello world!
    #
    # file.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="H:/2022_1_5_work/work/myloss/16_noise/gaus/100gaus/result/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="test")  # can be val, test, train, demoSequence
    parser.add_argument('--datadir', default="E:/16gaus_noise/")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=False)
    main(parser.parse_args())

