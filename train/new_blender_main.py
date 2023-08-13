# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math

import cv2
from PIL import Image, ImageOps
from argparse import ArgumentParser
import torch.nn as nn
from torch import autograd
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from HiPhase import HiPhaseNet

# from dataset import VOC12,cityscapes
from credataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard
from PIL import Image
import importlib
from iouEval import iouEval, getColorEntry
# from HiPhase import HiPhaseNet
# from deeplabv3p import DeepLabv3_plus
# from EESANet import EESANet
# from REDN import frrn
from shutil import copyfile
from erfnet import Net
# from EESANet import EESANet
# from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from erfnet_1cha import ERFNet1cha
from ssim import ssim

enco = True
NUM_CHANNELS = 1
NUM_CLASSES = 18  # pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


# def dowarp(unwarp):
#     dim0, dim1, dim2, dim3 = unwarp.shape
#     for k in range(dim0):
#         for i in range(dim2):
#             for j in range(dim3):
#                 element = unwarp[k][0][i][j].item()
#
#                 if element < -0.5:
#                     unwarp[k][0][i][j] = element + 1
#                     # print("改-")
#                 if element > 0.5:
#                     unwarp[k][0][i][j] = element - 1
#                     # print("改")
#     return  unwarp


# Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=False, height=256):
        self.enc = enc
        self.augment = augment
        self.height = height
        pass

    # def __call__(self, input, target,warx,wary):
    def __call__(self, input, target):
        # do something to both images
        # input =  Resize(self.height, Image.BILINEAR)(input)
        # warx= Resize(self.height, Image.BILINEAR)(warx)
        # wary= Resize(self.height, Image.BILINEAR)(wary)
        # target = Resize(self.height, Image.NEAREST)(target)
        # input_arr = np.array(input)
        # war_x=input_arr[:, :-1] - input_arr[ :, 1:]

        # if(self.augment):
        #     # Random hflip
        #     hflip = random.random()
        #     if (hflip < 0.5):
        #         input = input.transpose(Image.FLIP_LEFT_RIGHT)
        #         # warx=warx.transpose(Image.FLIP_LEFT_RIGHT)
        #         # wary=wary.transpose(Image.FLIP_LEFT_RIGHT)
        #         target = target.transpose(Image.FLIP_LEFT_RIGHT)
        #
        #     #Random translation 0-2 pixels (fill rest with padding
        #     transX = random.randint(-2, 2)
        #     transY = random.randint(-2, 2)
        #
        #     input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
        #     target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255)
        #     # warx=ImageOps.expand(warx, border=(transX,transY,0,0), fill=0)
        #     # wary=ImageOps.expand(wary, border=(transX,transY,0,0), fill=0)
        #     #pad label filling with 255
        #     input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
        #     # warx=warx.crop((0, 0, warx.size[0]-transX, warx.size[1]-transY))
        #     # wary=wary.crop((0, 0, wary.size[0]-transX, wary.size[1]-transY))
        #     target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        # warx=ToTensor()(warx)
        # wary = ToTensor()(wary)
        if (self.enc):
            target = Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        # target = Relabel(255, 19)(target)改成如下,与2022/2/26,如果不改会造成数组越界,进而只能跑encoder,跑不了decoder
        target = Relabel(255, NUM_CLASSES - 1)(target)

        # return input, target,warx,wary
        return input, target


metric_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        # 他说已经弃用
        self.loss = torch.nn.NLLLoss(weight)
        # self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        # return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


class residue(torch.nn.Module):
    def __init__(self):
        super(residue, self).__init__()

    def forward(warx, wary, unwarpped, enc=True):
        # print(inputs.max())
        # w_pad_x = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        # w_pad_y=nn.ZeroPad2d(padding=(0, 0, 0, 1))

        u_pad_x = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        u_pad_y = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        # w_grad_img_x = torch.mean(inputs[:, :, :, :-1] - inputs[:, :, :, 1:], 1, keepdim=True)

        # w_grad_img_y = torch.mean(inputs[:, :, :-1, :] - inputs[:, :, 1:, :], 1, keepdim=True)
        if enc == True:
            un_grad_img_x = unwarpped[:, :, :, 1:32] - unwarpped[:, :, :, 0:31]
            un_grad_img_y = unwarpped[:, :, 1:32, :] - unwarpped[:, :, 0:31, :]
        else:
            un_grad_img_x = unwarpped[:, :, :, 1:256] - unwarpped[:, :, :, 0:255]
            un_grad_img_y = unwarpped[:, :, 1:256, :] - unwarpped[:, :, 0:255, :]

        # w_grad_img_x = w_pad_x(w_grad_img_x)
        # print(w_grad_img_x.max())
        # print(w_grad_img_x.min())

        # w_grad_img_y = w_pad_y(w_grad_img_y)
        un_grad_img_x = u_pad_x(un_grad_img_x)
        un_grad_img_y = u_pad_y(un_grad_img_y)
        # tetss=w_grad_img_x.clone()
        # print(w_grad_img_x.max().item())
        # w_grad_img_x=dowarp(w_grad_img_x)
        # w_grad_img_y=dowarp(w_grad_img_y)
        # print(w_grad_img_x.equal(tetss))
        # print(1)

        resu = torch.mean(torch.abs(warx - un_grad_img_x) + torch.abs(wary - un_grad_img_y))

        return resu


def train(args, model, enc=False):
    best_acc = 0
    best_loss = float("inf")
    best_ssim = 0
    best_rmse = float("inf")
    best_cre = float("inf")

    # TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    # create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    # if (enc):
    #     weight[0]=1.1583
    #     weight[1]=1.3386
    #     weight[2]=1.1766
    #     weight[3]=1.0294
    #     weight[4]=0.9166
    #     weight[5]=0.8346
    #     weight[6]=0.7815
    #     weight[7]=0.7305
    #     weight[8]=0.7265
    #     weight[9]=0.7770
    #     weight[10]= 0.8158
    #     weight[11]=0.8961
    #     weight[12]=1.0000
    #     weight[13]=1.1446
    #     weight[14]=1.3207
    #     weight[15]= 1.1456
    #     weight[16]=78.3521
    #     weight[0] = 0.0023
    #     weight[1] = 0.0017
    #     weight[2] = 0.0013
    #     weight[3] = 0.0010
    #     weight[4] = 0.0008
    #     weight[5] = 0.0007
    #     weight[6] = 0.0005
    #     weight[7] = 0.0004
    #     weight[8] = 0.0004
    #     weight[9] = 0.0005
    #     weight[10] = 0.0007
    #     weight[11] = 0.0008
    #     weight[12] = 0.0011
    #     weight[13] = 0.0014
    #     weight[14] = 0.0020
    #     weight[15] = 0.0027
    #     weight[16] = 2.0823
    # else:
    # weight[0] = 1.1583
    # weight[1] = 1.3386
    # weight[2] = 1.1766
    # weight[3] = 1.0294
    # weight[4] = 0.9166
    # weight[5] = 0.8346
    # weight[6] = 0.7815
    # weight[7] = 0.7305
    # weight[8] = 0.7265
    # weight[9] = 0.7770
    # weight[10] = 0.8158
    # weight[11] = 0.8961
    # weight[12] = 1.0000
    # weight[13] = 1.1446
    # weight[14] = 1.3207
    # weight[15] = 1.1456
    # weight[16] = 78.3521
    # weight[0] = 0.0023
    # weight[1] = 0.0017
    # weight[2] = 0.0013
    # weight[3] = 0.0010
    # weight[4] = 0.0008
    # weight[5] = 0.0007
    # weight[6] = 0.0005
    # weight[7] = 0.0004
    # weight[8] = 0.0004
    # weight[9] = 0.0005
    # weight[10] = 0.0007
    # weight[11] = 0.0008
    # weight[12] = 0.0011
    # weight[13] = 0.0014
    # weight[14] = 0.0020
    # weight[15] = 0.0027
    # weight[16] = 2.0823

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(enc, augment=True, height=args.height)  # 1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)  # 1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'eval')

    loader_sample = RandomSampler(dataset_train, replacement=True, num_samples=10000, generator=None)
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                        sampler=loader_sample)
    loader_val_sample = RandomSampler(dataset_val, replacement=True, num_samples=1000, generator=None)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                            sampler=loader_val_sample)

    # if args.cuda:
    #     weight = weight.cuda()
    # criterion = CrossEntropyLoss2d(weight)
    criterion = CrossEntropyLoss2d()
    print(type(criterion))

    savedir = args.savedir

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write(
                "Epoch\t\tTrain-loss\t\tTest-loss\t\tcre_val\t\tssim_val\t\trmse_val\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    # TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)  ## scheduler 1
    # optimizer = Adam(model.parameters(), 1e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    # optimizer = Adam(model.parameters(), 1e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4) ## scheduler 2
    # optimizer=SGD(model.parameters(),momentum=0.9,lr=1e-02)
    start_epoch = 1
    if args.resume:
        # Must load weights, optimizer, epoch and best value.
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        # scheduler.step(epoch) 应该在optimizer.step()之后
        # scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []

        epoch_lossresdu = []
        epoch_loss_cri = []
        epoch_lossl1 = []
        epoch_lossssim = []
        epoch_lossrmse = []
        epoch_logssim = []

        doIouTrain = args.iouTrain
        doIouVal = args.iouVal

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        # for step, (images, labels,warx,wary) in enumerate(loader):
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            # print("label's size:")
            # print (labels.size())
            # print (np.unique(labels.numpy()))
            # print("labels: ", np.unique(labels[0].numpy()))
            # labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                # warx=warx.cuda()
                # wary=wary.cuda()

            inputs = Variable(images)

            targets = Variable(labels)
            # warxs=Variable(warx)
            # warys=Variable(wary)
            # erfnet前者
            if args.model == "erfnet" or args.model == "HiPhase":
                outputs = model(inputs, only_encode=enc)
            else:
                outputs = model(inputs)

            # outputs = model(inputs)

            # outputs = model(inputs)
            # order_float_18是HiPhase网络输出的通道为18的级次图
            # 这里*10是为了扩大数值范围使得softmax后更接近0或1
            # order_float = torch.nn.functional.softmax(outputs * 10, dim=1)
            # index生成[0,17]的级次序列和softmax出来的数据相乘
            index = []
            # for i in range(17):
            # for i in range(NUM_CLASSES):
            #     if(enc):
            #         index_ = torch.full([32, 32], i)
            #     else:
            #         index_ = torch.full([256, 256], i)
            #
            #
            #     index.append(index_)
            # index = torch.stack(index, 0).unsqueeze(0).cuda()
            # # 获得级次图
            # order1 = torch.sum(order_float * index, dim=1).unsqueeze(1)  # 级次图torch.Size([b,1,1024, 1024])
            # order1 = outputs.softmax(1).data
            # test_preds2 = inputs + test_preds1 * 2 * math.pi
            if enc == True:
                inputs = Resize(32, Image.NEAREST)(inputs)
            #     warx=Resize(32, Image.NEAREST)(warxs)
            #     wary=Resize(32, Image.NEAREST)(warys)

            # print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            # test=targets[:, 0]
            # loss = criterion(outputs, targets[:, 0])
            inputs = inputs * torch.tensor(2 * math.pi) - torch.tensor(math.pi)
            # loss1=residue.forward(warx,wary,order1*math.pi*2+inputs,enc)

            loss2 = criterion(outputs, targets[:, 0])
            if args.onlycre == True:
                loss = loss2
            else:
                a=1
                # loss1 = ssim(order1 * math.pi * 2 + inputs, inputs + targets * 2 * math.pi)
                # loss3 = torch.sqrt(metric_loss(order1 * math.pi * 2 + inputs, inputs + targets * 2 * math.pi))
                # loss = (-torch.log(loss1)) + loss3 + loss2

            # print("ssim：" + str(loss1))
            # print("交叉熵：" + str(loss2))
            # print("rmse" + str(loss3))
            # print("-logssim:"+str(-torch.log(loss1)))
            # print("loss1.size()")
            # print(loss1.size())
            # print("loss2.size()")
            # print(loss2.size())
            # print("loss3.size()")
            # print(loss3.size())
            # loss=100*(-torch.log(loss1))+loss3+10*loss2

            # loss=loss2

            loss.backward()
            optimizer.step()
            # 改在之后
            # scheduler.step(epoch)

            # epoch_loss.append(loss.data[0])
            epoch_loss.append(loss.item())
            # epoch_lossresdu.append(loss1.item())
            # epoch_loss_cri.append(loss2.item())
            # epoch_lossl1.append(loss3.item())

            # epoch_lossssim.append(loss1.item())
            # epoch_loss_cri.append(loss2.item())
            # epoch_lossrmse.append(loss3.item())
            # epoch_logssim.append(-torch.log(loss1).item())

            time_train.append(time.time() - start_time)

            if (doIouTrain):
                # start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                # print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            # print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                # image[0] = image[0] * .229 + .485
                # image[1] = image[1] * .224 + .456
                # image[2] = image[2] * .225 + .406
                # print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):  # merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                            f'target (epoch: {epoch}, step: {step})')
                print("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:

                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                if args.onlycre == False:
                    average_ssim = sum(epoch_lossssim) / len(epoch_lossssim)
                    average_rmse = sum(epoch_lossrmse) / len(epoch_lossrmse)
                    # average_ssimlog=sum(epoch_lossssimlog)/len(epoch_logssim)
                    # average_re=sum(epoch_lossresdu)/len(epoch_lossresdu)
                    average_cre = sum(epoch_loss_cri) / len(epoch_loss_cri)
                    print(f'loss cri: {average_cre:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                    # print(f' loss l1: {average_l1:0.4} (epoch: {epoch}, step: {step})',
                    #       "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                    print(f'loss ssim: {average_ssim:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
                    print(f'loss rmse: {average_rmse:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

                # average_l1=sum( epoch_lossl1)/len( epoch_lossl1)

                # print(f' loss re: {average_re:0.4} (epoch: {epoch}, step: {step})',
                #       "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

                # print(f'loss logssim: {average_ssimlog:0.4} (epoch: {epoch}, step: {step})',
                #       "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain) + '{:0.2f}'.format(iouTrain * 100) + '\033[0m'
            print("EPOCH IoU on TRAIN set: ", iouStr, "%")

            # Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []
        epoch_lossresdu_val = []
        epoch_loss_cri_val = []
        epoch_l1_val = []
        epoch_ssim_val = []
        epoch_rmse_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        # for step, (images, labels,warx,wary) in enumerate(loader_val):
        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                # warx=warx.cuda()
                # wary=wary.cuda()

            # 因为volatile已经禁用，修改
            # inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            # targets = Variable(labels, volatile=True)
            with torch.no_grad():
                inputs = Variable(images)  # volatile flag makes it free backward or outputs for eval
            with torch.no_grad():
                targets = Variable(labels)
            # with torch.no_grad():
            #     warxs = Variable(warx)
            # with torch.no_grad():
            #     warys = Variable(wary)
            if args.model == "erfnet" or args.model == "HiPhase":
                outputs = model(inputs, only_encode=enc)
            else:
                outputs = model(inputs)
            # outputs = model(inputs)
            order2 = outputs.max(1)[1].unsqueeze(1).data
            # test_preds2 = inputs + test_preds1 * 2 * math.pi
            if enc == True:
                inputs = Resize(32, Image.NEAREST)(inputs)
                # warx=Resize(32, Image.NEAREST)(warxs)
                # wary=Resize(32, Image.NEAREST)(warys)
            with torch.no_grad():
                inputs = inputs * torch.tensor(2 * math.pi) - torch.tensor(math.pi)
                # loss1 = residue.forward(warx,wary, order2 * math.pi * 2 + inputs,enc=enc)
                loss2 = criterion(outputs, targets[:, 0])
                loss_ssim = ssim(inputs + order2 * math.pi * 2, inputs + targets * math.pi * 2)
                loss_rmse = torch.sqrt(metric_loss(inputs + order2 * math.pi * 2, inputs + targets * math.pi * 2))
                # loss3 = metric_loss(order2 * math.pi * 2 + inputs, inputs + targets * 2 * math.pi)

                # loss = loss1 + loss3 + loss2
                # loss=loss2+loss3
                loss = 10 * loss2 + 100 * (-1 * torch.log(loss_ssim)) + loss_rmse

            epoch_loss_val.append(loss.item())
            epoch_loss_cri_val.append(loss2.item())
            epoch_ssim_val.append(loss_ssim.item())
            epoch_rmse_val.append(loss_rmse.item())

            # epoch_lossresdu_val.append(loss1.item())
            # epoch_loss_cri_val.append(loss2.item())
            # epoch_l1_val.append(loss3.item())

            time_val.append(time.time() - start_time)

            # Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                # start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                # print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):  # merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                                f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                            f'VAL target (epoch: {epoch}, step: {step})')
                print("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                # average_re=sum(epoch_lossresdu_val)/len(epoch_lossresdu_val)
                average_cre = sum(epoch_loss_cri_val) / len(epoch_loss_cri_val)
                average_rmse = sum(epoch_rmse_val) / len(epoch_rmse_val)
                average_ssim = sum(epoch_ssim_val) / len(epoch_ssim_val)
                # average_l1=sum( epoch_l1_val)/len( epoch_l1_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                # print(f'VAL loss re: {average_re:0.4} (epoch: {epoch}, step: {step})',
                #       "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                print(f'VAL loss cre: {average_cre:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                # print(f'VAL loss l1: {average_l1:0.4} (epoch: {epoch}, step: {step})',
                #       "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                print(f'VAL loss rmse: {average_rmse:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                print(f'VAL loss ssim: {average_ssim:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                # print(f'VAL loss logssim: {-math.log(average_ssim):0.4} (epoch: {epoch}, step: {step})',
                #       "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        average_epoch_cre_val = sum(epoch_loss_cri_val) / len(epoch_loss_cri_val)
        average_epoch_rmse_val = sum(epoch_rmse_val) / len(epoch_rmse_val)
        average_epoch_ssim_val = sum(epoch_ssim_val) / len(epoch_ssim_val)
        # scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
            print("EPOCH IoU on VAL set: ", iouStr, "%")

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)
        # remember best loss and save checkpoint

        current_loss = loss.item()
        current_ssim = loss_ssim.item()
        current_rmse = loss_rmse.item()
        current_cre = loss2.item()

        is_best_loss = current_loss < best_loss
        is_best_ssim = current_ssim > best_ssim
        is_best_rmse = current_rmse < best_rmse
        is_best_cre = current_cre < best_cre
        best_loss = min(current_loss, best_loss)
        best_cre = min(best_cre, current_cre)
        best_rmse = min(best_rmse, current_rmse)
        best_ssim = max(best_ssim, current_ssim)
        if enc:
            filenameCheckpointloss = savedir + '/checkpointloss_enc.pth.tar'
            filenameBestloss = savedir + '/modelloss_best_enc.pth.tar'
            filenameBestssim = savedir + '/modelssim_best_enc.pth.tar'
            filenameBestcre = savedir + '/modelcre_best_enc.pth.tar'
            filenameBestrmse = savedir + '/modelrmse_best_enc.pth.tar'


        else:
            filenameCheckpointloss = savedir + '/checkpointloss.pth.tar'
            filenameBestloss = savedir + '/modelloss_best.pth.tar'
            filenameBestssim = savedir + '/modelssim_best.pth.tar'
            filenameBestcre = savedir + '/modelcre_best.pth.tar'
            filenameBestrmse = savedir + '/modelrmse_best.pth.tar'
        save_checkpointloss({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'best_ssim': best_ssim,
            'best_cre': best_cre,
            'best_rmse': best_rmse,
            'optimizer': optimizer.state_dict(),
        }, is_best_loss, is_best_ssim, is_best_cre, is_best_rmse, filenameCheckpointloss, filenameBestloss,
            filenameBestssim, filenameBestrmse, filenameBestcre)
        # SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'

        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))
        if (is_best_loss):
            torch.save(model.state_dict(), filenameBestloss)
            print(f'save: {filenameBestloss} (epoch: {epoch})')
            if (not enc):
                filenamebest = f'{savedir}/modelloss_best.pth'
                with open(savedir + "/bestloss.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with loss= %.4f" % (epoch, best_loss))
            else:
                filenamebest = f'{savedir}/modelloss_best_enc.pth'
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with loss= %.4f" % (epoch, best_loss))
            torch.save(model.state_dict(), filenamebest)

        if (is_best_ssim):
            torch.save(model.state_dict(), filenameBestssim)
            print(f'save: {filenameBestssim} (epoch: {epoch})')
            if (not enc):
                filenamebest = f'{savedir}/modelssim_best.pth'
                with open(savedir + "/bestssim.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with ssim= %.4f" % (epoch, best_ssim))
            else:
                filenamebest = f'{savedir}/modelssim_best_enc.pth'
                with open(savedir + "/best_encoder_ssim.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with ssim= %.4f" % (epoch, best_ssim))

        if (is_best_rmse):
            torch.save(model.state_dict(), filenameBestrmse)
            print(f'save: {filenameBestrmse} (epoch: {epoch})')
            if (not enc):
                filenamebest = f'{savedir}/modelrmse_best.pth'
                with open(savedir + "/bestrmse.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with rmse %.4f" % (epoch, best_rmse))
            else:
                filenamebest = f'{savedir}/modelrmse_best_enc.pth'
                with open(savedir + "/best_encoder_rmse.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with rmse= %.4f" % (epoch, best_rmse))

        if (is_best_cre):
            torch.save(model.state_dict(), filenameBestcre)
            print(f'save: {filenameBestcre} (epoch: {epoch})')
            if (not enc):
                filenamebest = f'{savedir}/modelcre_best.pth'
                with open(savedir + "/bestcre.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with cre %.4f" % (epoch, best_cre))
            else:
                filenamebest = f'{savedir}/modelcre_best_enc.pth'
                with open(savedir + "/best_encoder_cre.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with cre= %.4f" % (epoch, best_cre))
        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
            epoch, average_epoch_loss_train, average_epoch_loss_val, average_epoch_cre_val, average_epoch_ssim_val,
            average_epoch_rmse_val, iouTrain, iouVal, usedLr))

    return (model)  # return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def save_checkpointloss(state, is_best_loss, is_best_ssim, is_best_cre, is_best_rmse, filenameCheckpointloss,
                        filenameBestloss, filenameBestssim, filenameBestrmse, filenameBestcre):
    torch.save(state, filenameCheckpointloss)
    if is_best_loss:
        print("Saving model as best loss")
        torch.save(state, filenameBestloss)

    if is_best_ssim:
        print("Saving model as best ssim")
        torch.save(state, filenameBestssim)

    if is_best_rmse:
        print("Saving model as best rmse")
        torch.save(state, filenameBestrmse)

    if is_best_cre:
        print("Saving model as best cres")
        torch.save(state, filenameBestcre)


def main(args):
    savedir = args.savedir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    # model = model_file.frrn()
    # model = model_file.EESANet(imagechan=1,chann=48,numclass=NUM_CLASSES)
    model = model_file.Net(NUM_CLASSES)
    # model=model_file.HiPhaseNet(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """

        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            return model

        # print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder

    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    # train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True)  # Train encoder
    # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0.
    # We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()  # because loaded encoder is probably saved in cuda
        else:
            # erfnet使用前者
            if args.model == "erfnet" or args.model == "HiPhase":
                pretrainedEnc = next(model.children()).encoder
            else:
                pretrainedEnc = next(model.children())  # deco

        # model = model_file.EESANet(imagechan=1,chann=48,numclass=NUM_CLASSES)
        model = model_file.Net(NUM_CLASSES)
        # model = model_file.frrn()
        # model = model_file.HiPhaseNet(NUM_CLASSES,pretrainedEnc)  #Add decoder to encoder
        # ERFNET 和 hiphase使用后者
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, enc=False)  # Train decoder
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=True)  # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', type=str, default="H:/2022_1_5_work/work/new_blender")
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=20)
    # encoder size 40，de 28
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)  # You can use this value to save model every X epochs
    # parser.add_argument('--savedir',default="E:/work/guas_data/result/")
    parser.add_argument('--savedir', default="H:/2022_1_5_work/work/new_blender_result/blender/100blender/result/")
    parser.add_argument('--decoder', action='store_true', default=True)
    parser.add_argument('--pretrainedEncoder')  # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true')  # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true')  # Use this flag to load last checkpoint for training
    parser.add_argument('--onlycre', action='store_true', default=True)
    main(parser.parse_args())
