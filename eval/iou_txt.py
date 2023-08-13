# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################
import math
import re


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
from  erfphase import  erpahseNet
from  EESANet import  EESANet
from transform import Relabel, ToLabel, Colorize
from  segformer import  Segformer
from iouEval import iouEval, getColorEntry
import  xlrd
# import xlwt
import  os
# import  re
from xlutils.copy import  copy
from  VuRNet import  Net
from  erfnet_1cha import ERFNet1cha
from  SegNet import  SegNet_Blender

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

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # iouEvalVal = iouEval(NUM_CLASSES)
    # 指定输出文件路径
    output_path = 'G:\metric_map\mIoU\output'
    output_file = output_path + args.loadModel[:-3] + datatype + '.txt'

    output_folder = os.path.dirname(output_file)
    os.makedirs(output_folder, exist_ok=True)

    output_path_RMSE = 'G:\metric_map\RMSE\output'
    output_file_RMSE = output_path_RMSE + args.loadModel[:-3] + datatype + '.txt'

    output_folder_RMSE = os.path.dirname(output_file_RMSE)
    os.makedirs(output_folder_RMSE, exist_ok=True)

    output_path_SSIM = 'G:\metric_map\SSIM\output'
    output_file_SSIM = output_path_SSIM + args.loadModel[:-3] + datatype + '.txt'

    output_folder_SSIM = os.path.dirname(output_file_SSIM)
    os.makedirs(output_folder_SSIM, exist_ok=True)
    iouEvalVal = iouEval(NUM_CLASSES)
    epoch_loss_val=[]
    epoch_SSIM_val=[]
    epoch_loss_iou=[]
    iou_map={}
    ssim_map={}
    rmse_map={}

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            #images=images*torch.tensor(2*math.pi)-torch.tensor(math.pi)


            images = images.cuda()
            labels = labels.cuda()

        # inputs = Variable(images)
        with torch.no_grad():
            iouEvalVal = iouEval(NUM_CLASSES)
            inputs = Variable(images)
            # volatile flag makes it free backward or outputs for eval
            targets = Variable(labels)
            test_preds = model(inputs)
            if args.loadModel == "segformer.py":
               test_preds = F.interpolate(test_preds, size=256, mode='bilinear', align_corners=False)
            flag = args.loadModel in (["PhUn.py", "DLPU.py", "VuRNet.py"])
            if flag:
                # print("flag")
                test_preds2 = test_preds
            iouEvalVal.addBatch(test_preds.max(1)[1].unsqueeze(1).data, labels)
            inputs = inputs * torch.tensor(2 * math.pi) - torch.tensor(math.pi)
            test_preds1 = test_preds.max(1)[1].unsqueeze(1).data

            test_preds2 = inputs + test_preds1 * 2 * math.pi
            targets1 = inputs + targets * 2 * math.pi
            metric_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

            test_loss3 = torch.sqrt(metric_loss(test_preds2, targets1))
            test_loss1 = ssim(test_preds2, targets1)
            iouVal, iou_classes = iouEvalVal.getIoU()

            iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m' + '%'
            epoch_loss_iou.append(iouStr)
            epoch_SSIM_val.append(test_loss1.item())
            epoch_loss_val.append(test_loss3.item())









        #
        # iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("input/")[1]
        # iou_map[filenameSave] = iouStr
        # ssim_map[filenameSave]=test_loss1
        # rmse_map[filenameSave]=test_loss3
        # 打开文件以写入数据
        with open(output_file, 'a') as file:
            # 遍历数组元素，并将其写入文件
            clean_element = re.sub(r'\x1B\[[0-9;]*[mK]?', '', str(iouStr))
            file.write(str(clean_element) + '\n')
        # 打开文件以写入数据
        with open(output_file_RMSE, 'a') as file:
            # 遍历数组元素，并将其写入文件
            file.write(str(test_loss3.item())  + '\n')
        with open(output_file_SSIM, 'a') as file:
            # 遍历数组元素，并将其写入文件
            file.write(str(test_loss1.item())  + '\n')
        # 输出完成消息
        print('数组元素已成功写入文件：', str(filenameSave))













    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")

    print("=======================================")
    # iouVal, iou_classes = iouEvalVal.getIoU()
    #
    # iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m' + '%'
    # with open(output_file, 'a') as file:
    #     # 遍历数组元素，并将其写入文件
    #     clean_element = re.sub(r'\x1B\[[0-9;]*[mK]?', '', str(iouStr))
    #     file.write(str(clean_element)  + '\n')
    # # 打开文件以写入数据
    # with open(output_file_RMSE, 'a') as file:
    #     # 遍历数组元素，并将其写入文件
    #     file.write(str(sum(epoch_loss_val)/len(epoch_loss_val))  + '\n')
    # with open(output_file_SSIM, 'a') as file:
    #     # 遍历数组元素，并将其写入文件
    #     file.write(str(sum(epoch_SSIM_val)/len(epoch_SSIM_val))  + '\n')
    # 输出完成消息
    print('数组元素已成功写入文件：')
    # finalloss=sum(epoch_loss_val)/len(epoch_loss_val)
    # print("ken:"+str(len(epoch_loss_val)))
    # print("loss"+str(finalloss))
    # rmsesd=np.std(epoch_loss_val)
    # print("sd" + str(rmsesd))
    # sheet_miou.write(miou_with_row,miou_with_col,test_loss)
    # sheet_withouback.write(miou_with_row, miou_with_col, round(finalloss,4))
    # os.remove(xlsfilename)
    # wb.save(xlsfilename)

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
    model_vurnet=Net()
    model_phun=PhUn()
    # model_1cha=ERFNet1cha()
    model_erfphase=erpahseNet(num_classes=NUM_CLASSES)
    model_ee=EESANet(1,48,NUM_CLASSES)



    lie=1
    # loadmo = ["erfnet_1cha.py"]
    # modellist = [model_1cha]
    # pathlist = ["work"]
    loadmo=["erfnet.py","HiPhase.py","tiramisu.py","REDN.py","SegNet.py","deeplabv3p.py","segformer.py","EESANet.py","erfphase.py","PhUn.py","DLPU.py","VuRNet.py"]
    modellist=[model_erf,model_hip,model_phase,model_redn,model_seg,model_deep,model_segfor,model_ee,model_erfphase,model_phun,model_dlpu,model_vurnet]
    #pathlist=["erfnet","HiPhase","phasenet2","REDN","SegNet","deeplabv3p","segformer","eesanet","erfphase","phun","DLPU","new_vurnet"]
    pathlist = ["work", "hiphase", "phasenet", "redn", "segnet", "deeplabv3p", "segformer", "eesanet", "erfphase","phun","DLPU","new_vurnet"]
    hangjia=0
    hangzer=[2,10,17,25]
    # for index in range(0,1):
    for index in [3]:


    # for index in range(4,len(loadmo)):
        lie=lie+1
        for datatype in ["gaus","zer"]:
        # for datatype in [ "blender"]:

            if datatype == "blender":
                datadir = "H:/2022_1_5_work/work/new_blender/"
                hangjia = 2
            if datatype == "zer":
                datadir = "D:/2022_1_5_work/work/zer/100zer/"
                hangjia = 0
            if datatype == "gaus":
                datadir = "D:/2022_1_5_work/work/new_gaus/"
                hangjia = 1

            for num in [ "100"]:
                parser = ArgumentParser()
                if pathlist[index] in ["HiPhase","phasenet","deeplabv3p","phun","DLPU","new_vurnet"]:
                    dir = "H:/2022_1_5_work/" + pathlist[index] +"/" + datatype + "/" + num + datatype + "/result/"
                else:
                    dir = "D:/2022_1_5_work/" + pathlist[index] +"/" +datatype + "/" + num + datatype + "/result/"
                #dir="H:/2022_1_5_work/work/new_blender_result/"+ pathlist[index]+"/blender/100blender/result/"
               # dir="D:/2022_1_5_work/eesanet/"+"/newphaseloss/" + datatype + "/" + num + datatype + "/"
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



