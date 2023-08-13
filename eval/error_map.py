# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################
import math

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time
from  ssim import  ssim
from PIL import Image
from argparse import ArgumentParser
import  HiPhase
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from tiramisu import  PhaseNet2
from dataset import cityscapes
from  deeplabv3p import  DeepLabv3_plus
from SegNet import  SegNet
from erfnet import ERFNet
from  HiPhase import  HiPhaseNet
# from  segformer import  Segformer
from  REDN import  frrn
from  DLPU import DLPU
from  VUR_net import  VURnet
from  PhUn  import  PhUn
# from  transphase import  TransPhase
from  VuRNet import  Net as VuRNet
from  EESANet import  EESANet
from transform import Relabel, ToLabel, Colorize
from  segformer import  Segformer
from iouEval import iouEval, getColorEntry
import  xlrd
# import xlwt
import  os
# import  re
from xlutils.copy import  copy
from  erfnet_1cha import ERFNet1cha

xlsfilename='C:/Users/Administrator/Desktop/paper_final.xls'
NUM_CHANNELS = 1
NUM_CLASSES = 17
rb=xlrd.open_workbook_xls(xlsfilename)
wb=copy(rb)
sheet_order=wb.get_sheet(1)
# sheet_miou=wb.get_sheet(4)
sheet_withouback=wb.get_sheet(0)


miou_with_row,miou_with_col=12,2
class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

           # 他说已经弃用
        self.loss = torch.nn.NLLLoss(weight)
        # self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        # return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    # Resize(256, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    # Resize(256, Image.NEAREST),
    ToLabel(),
    Relabel(255, NUM_CLASSES-1),   #ignore label to 19
])

def main(args,modelnow,datatype):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model=modelnow

    #model = torch.nn.DataParallel(model)
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
    print(args.datadir)

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")



    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # iouEvalVal = iouEval(NUM_CLASSES)


    # iouEvalVal = iouEval(NUM_CLASSES)
    epoch_loss_val=[]

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            #images=images*torch.tensor(2*math.pi)-torch.tensor(math.pi)


            images = images.cuda()
            labels = labels.cuda()

        # inputs = Variable(images)
        with torch.no_grad():
            inputs = Variable(images)

            # volatile flag makes it free backward or outputs for eval
            targets = Variable(labels)

            test_preds = model(inputs)

            inputs = inputs * torch.tensor(2 * math.pi) - torch.tensor(math.pi)
            flag = args.loadModel in ["DLPU.py", "VuRNet.py", "PhUn.py"]
            if flag:
                print("faffsafsafafafas")
                test_preds2 = test_preds
                order = torch.round(abs(test_preds2 - inputs) / (math.pi * 2))
                test_preds2 = order * torch.tensor(2 * math.pi) + inputs
            else:
                if args.loadModel == "segformer.py":
                    test_preds = F.interpolate(test_preds, size=256, mode='bilinear', align_corners=False)

                test_preds1 = test_preds.max(1)[1].unsqueeze(1).data
                # test_preds1=torch.round(test_preds)
                test_preds2 = inputs + test_preds1 * 2 * math.pi
            # iouEvalVal.addBatch(test_preds.max(1)[1].unsqueeze(1).data, labels)
            targets1 = inputs + targets * 2 * math.pi

            error_map = abs(torch.sub(test_preds2, targets1))
            # 转换为numpy数组
            error_np=error_map.squeeze().cpu().numpy()
            # 创建一个256x256的全白色图像
            image_np = 255 * np.ones((256, 256), dtype=np.uint8)
            # 将非零值的像素设置为黑色
            image_np[error_np != 0] = 0
            image = Image.fromarray(image_np)

        filenameSave = "D:/all_error_pic/mymethod_noise/" + args.loadModel[:-2]+"_"+datatype+"_"  + filename[0].split("input/")[1][
                                                                                :-4]   + ".png"
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        image.save(filenameSave)

        print (step, filenameSave)




    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")

    print("=======================================")


if __name__ == '__main__':
    model_seg = SegNet()
    model_erf=ERFNet(NUM_CLASSES)
    model_hip=HiPhaseNet(NUM_CLASSES)
    model_phase=PhaseNet2(NUM_CLASSES)
    model_redn=frrn()
    model_deep=DeepLabv3_plus()
    model_segfor=Segformer()
    # model_segfor=SegNet()
    model_dlpu=DLPU()
    model_vurnet=VuRNet()
    model_phun=PhUn()
    # model_1cha=ERFNet1cha()
    # model_erfphase=Net(num_classes=NUM_CLASSES)
    model_ee=EESANet(1,48,NUM_CLASSES)

    lie=1
    # loadmo = ["erfnet_1cha.py"]
    # modellist = [model_1cha]
    # pathlist = ["work"]
    loadmo=["erfnet.py","HiPhase.py","tiramisu.py","REDN.py","SegNet.py","deeplabv3p.py","segformer.py","EESANet.py","PhUn.py","DLPU.py","VuRNet.py"]
    modellist=[model_erf,model_hip,model_phase,model_redn,model_seg,model_deep,model_segfor,model_ee,model_phun,model_dlpu,model_vurnet]
    #pathlist=["work","HiPhase","phasenet","REDN","SegNet","deeplabv3p","segformer","eesanet","erfphase","phun","DLPU","vur_result"]
    pathlist = ["work", "HiPhase", "phasenet", "REDN", "SegNet", "deeplabv3p", "segformer", "eesanet","phun","DLPU","vur_net"]
    hangjia=0
    hangzer=[2,10,17,25]
    for index in range(0,1):
        # if index==6:
        #     continue

    # for index in range(4,len(loadmo)):
        lie=lie+1
        for datatype in ["gaus" ]:
        # for datatype in [ "blender"]:

            # if datatype == "blender":
            #     datadir =  datadir = "H:/real_phase/"

            for num in [ "100"]:
                parser = ArgumentParser()
                if datatype == "blender":
                    datadir = "H:/2022_1_5_work/work/new_blender/"
                    hangjia = 2
                if datatype == "zer":
                    datadir = "D:/2022_1_5_work/work/zer/100zer/"
                    hangjia = 0
                if datatype == "gaus":
                    datadir = "D:/2022_1_5_work/work/new_gaus/"
                    datadir = "E:/16gaus_noise/"
                    hangjia = 1

                for num in ["100"]:
                    parser = ArgumentParser()
                    if pathlist[index] in ["HiPhase", "phasenet", "deeplabv3p", "phun", "DLPU", "new_vurnet","vur_net"]:
                        dir = "H:/2022_1_5_work/" + pathlist[index] + "/" + datatype + "/" + num + datatype + "/result/"
                    else:
                        dir = "D:/2022_1_5_work/" + pathlist[index] +"/"+ datatype + "/" + num + datatype + "/result/"
                   # dir = "D:/2022_1_5_work/work/myloss/"+ datatype + "/" + num + datatype + "/result/"
                dir = "H:/2022_1_5_work/work/new_blender_result/" + "/new_vurnet/"                                                                                                                                                                                                                           + "/blender/100blender/result/"
                dir = "H:/2022_1_5_work/work/myloss/16_noise/gaus/100gaus/result/"

                print(dir)
                if num == "10":
                    hang = 2
                if num == "20":
                    hang = 10
                if num == "50":
                    hang = 17
                if num == "100":
                    hang = 25
                hang = hang + hangjia
                print("hang :"+str(hang))
                print("lie :"+str(lie))
                miou_with_row, miou_with_col = hang, lie
                if os.path.exists(dir):
                    parser.add_argument('--state')

                    parser.add_argument('--loadDir', default=dir)
                    parser.add_argument('--loadWeights', default="model_best.pth")
                    parser.add_argument('--loadModel', default=loadmo[index])
                    parser.add_argument('--subset', default="test")  # can be val or train (must have labels)
                    parser.add_argument('--datadir', default=datadir)
                    parser.add_argument('--num-workers', type=int, default=1)
                    parser.add_argument('--batch-size', type=int, default=1)
                    parser.add_argument('--cpu', action='store_true', default=False)

                    main(parser.parse_args(), modellist[index],datatype)







            #print("model:"+dir+"data:"+datadir+"hanglie:"+str([hang,lie]))