# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from HiPhase import HiPhaseNet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry
from  REDN import  frrn
xlsfilename='C:/Users/Administrator/Desktop/paper_final.xls'
import  xlrd
import xlwt
import  os
import  re
from xlutils.copy import  copy
from  SegNet import  SegNet
from  tiramisu import  PhaseNet2
from segformer import Segformer
from  deeplabv3p import  DeepLabv3_plus
rb=xlrd.open_workbook_xls(xlsfilename)
wb=copy(rb)
from  EESANet import  EESANet
miou_row,miou_col=12,2
order_row,order_col=25,21
sheet_withouback=wb.get_sheet(2)

miou_with_row,miou_with_col=25,2

NUM_CHANNELS = 1
NUM_CLASSES = 17

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

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model =ERFNet(NUM_CLASSES)

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


    iouEvalVal = iouEval(NUM_CLASSES)
    iou_val=[]

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        iouEvalVal_epoch = iouEval(NUM_CLASSES)
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        targets=Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = F.interpolate(outputs, size=256, mode='bilinear', align_corners=False)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data,labels)
        iouEvalVal_epoch.addBatch(outputs.max(1)[1].unsqueeze(1).data,labels)

        filenameSave = filename[0].split("input/")[1]

        print (step, filenameSave)
        step_iou=iouEvalVal.getIoU()
        iou_val.append(step_iou)


    iouVal, iou_classes = iouEvalVal.getIoU()
    # output_path = 'G:\metric_map\mIoU\output'
    # output_file = output_path + args.loadModel[:-3] + datatype + '.txt'
    #
    # output_folder = os.path.dirname(output_file)
    # os.makedirs(output_folder, exist_ok=True)

    # 打开文件以写入数据
    # with open(output_file, 'w') as file:
    #     # 遍历数组元素，并将其写入文件
    #     for element in epoch_loss_val:
    #         file.write(str(element) + '\n')
    #
    # # 输出完成消息
    # print('数组元素已成功写入文件：', output_file)

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    print("TOTAL IOU: ", iouStr * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "0")
    print(iou_classes_str[1], "1")
    print(iou_classes_str[2], "2")
    print(iou_classes_str[3], "3")
    print(iou_classes_str[4], "4")
    print(iou_classes_str[5], "5")
    print(iou_classes_str[6], "6")
    print(iou_classes_str[7], "7")
    print(iou_classes_str[8], "8")
    print(iou_classes_str[9], "9")
    print(iou_classes_str[10], "10")
    print(iou_classes_str[11], "11")
    print(iou_classes_str[12], "12")
    print(iou_classes_str[13], "13")
    print(iou_classes_str[14], "14")
    print(iou_classes_str[15], "15")
    print(iou_classes_str[16], "16")
    # print(iou_classes_str[17], "17")
    # print(iou_classes_str[18], "18")
    # print(iou_classes_str[19], "19")
    # print(iou_classes_str[20], "20")
    # print(iou_classes_str[21], "21")
    # print(iou_classes_str[22], "22")
    # print(iou_classes_str[23], "23")
    # print(iou_classes_str[24], "24")
    # print(iou_classes_str[25], "25")
    # print(iou_classes_str[26], "26")
    # print(iou_classes_str[27], "27")
    # print(iou_classes_str[28], "28")
    # print(iou_classes_str[29], "29")
    # print(iou_classes_str[30], "30")
    # print(iou_classes_str[31], "31")
    # print(iou_classes_str[32], "32")
    # m15 = 0
    # for i in range(1, 17):
    #     tem = iou_classes_str[i] + '\033[0m'
    #     final_tem = re.sub('\x1b.*?m', '', tem)
    #     int_final = float(final_tem) / 100
    #     m15 = m15 + int_final
        # iou_tem=float(tem)
        # tem=iou_tem/100

        # sheet_order.write(order_row, 4+i, int_final)
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.4f}'.format(iouVal) + '\033[0m'
    iouStrfinal = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU: ", iouStrfinal, "%")
    # final_iou = re.sub('\x1b.*?m', '', iouStr)
    # sheet_order.write(order_row, order_col, final_iou)
    # sheet_miou.write(miou_with_row, miou_with_col,final_iou)
    # sheet_withouback.write(miou_with_row, miou_with_col, round(m15 /16, 4))
    # os.remove(xlsfilename)
    # wb.save(xlsfilename)


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
    parser.add_argument('--cpu', action='store_true',default=False)

    main(parser.parse_args())
