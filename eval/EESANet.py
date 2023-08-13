# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

'''
2022/9/6/14 发现srb来增加减少通道，现在已经实现srb和maxpool,transconv
'''
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

    """3x3 convolution with padding"""
# class SRB(nn.Module):    srb_ot change channel version
#     def __init__(self, chann,outchan):
#         super().__init__()
#         self.conv3x3_1=nn.Conv2d(chann,chann,3,padding=1)
#         self.conv3x3_2=nn.Conv2d(chann,chann,3,padding=1)
#         self.conv3x3_3=nn.Conv2d(chann,chann,3,padding=1)
#         self.conv3x3_4=nn.Conv2d(chann,outchan,3,padding=1)
#         self.BN_1=nn.BatchNorm2d(num_features=chann)
#         self.BN_2=nn.BatchNorm2d(num_features=chann)
#         self.BN_3=nn.BatchNorm2d(num_features=chann)
#         self.BN_4=nn.BatchNorm2d(num_features=outchan)
#     def forward(self, input):
#         output1 = self.conv3x3_1(input)
#         output1 = self.BN_1(output1)
#         output1=F.leaky_relu(output1)
#         output2=self.conv3x3_2(output1)
#         output2=self.BN_2(output2)
#         output2=F.leaky_relu(output2)
#         output2=output1+output2
#         output3=self.conv3x3_3(output2)
#         output3=self.BN_3(output3)
#         output3=F.leaky_relu(output3)
#         output3=output3+output2
#         output4=self.conv3x3_4(output3)
#         output4=self.BN_4(output4)
#         output4=F.leaky_relu(output4)
#         output=output4+output3
#         return output
"""change chanel"""
class SRB(nn.Module):
    def __init__(self, chann,outchan):
        super().__init__()
        self.conv3x3_1=nn.Conv2d(chann,outchan,3,padding=1)
        self.conv3x3_2=nn.Conv2d(outchan,outchan,3,padding=1)
        self.conv3x3_3=nn.Conv2d(outchan,outchan,3,padding=1)
        self.conv3x3_4=nn.Conv2d(outchan,outchan,3,padding=1)
        self.BN_1=nn.BatchNorm2d(num_features=outchan)
        self.BN_2=nn.BatchNorm2d(num_features=outchan)
        self.BN_3=nn.BatchNorm2d(num_features=outchan)
        self.BN_4=nn.BatchNorm2d(num_features=outchan)
    def forward(self, input):
        output1 = self.conv3x3_1(input)
        output1 = self.BN_1(output1)
        output1=F.leaky_relu(output1)
        output2=self.conv3x3_2(output1)
        output2=self.BN_2(output2)
        output2=F.leaky_relu(output2)
        output2=output1+output2
        output3=self.conv3x3_3(output2)
        output3=self.BN_3(output3)
        output3=F.leaky_relu(output3)
        output3=output3+output2
        output4=self.conv3x3_4(output3)
        output4=self.BN_4(output4)
        output4=F.leaky_relu(output4)
        output=output4+output3
        return output

class PSA(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation='relu'):
        super(PSA, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # b
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # c
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # d
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)  # S

        out = self.gamma * out + x
        return out


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block5 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.atrous_block8 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear',align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block3 = self.atrous_block3(x)

        atrous_block5 = self.atrous_block5(x)

        atrous_block8 = self.atrous_block8(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block3,
                                              atrous_block5, atrous_block8], dim=1))
        return net


class EEB(nn.Module):
    def __init__(self,chann=1):
        super().__init__()
        self.conv3x3_1 = nn.Conv2d(chann, chann, 1)
        self.r=1
        self.bia=0
    def forward(self, x):
        output=self.conv3x3_1(x)*self.r+self.bia
        return  output

class conv(nn.Module) :
    def __init__(self, chann):
        super(conv, self).__init__()
        self.conv1=nn.Conv2d(chann, 2*chann, kernel_size=3, stride=1, padding=0, bias=True)
    def forward(self,x):
        x=self.conv1(x)
        return x

class maxpo(nn.Module):
    def __init__(self,kersize=2,stri=2):
        super(maxpo, self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=kersize,stride=stri)
    def forward(self,x):
        x=self.maxpool(x)
        return  x
class tranconv(nn.Module):
    def __init__(self,inputc,outc,kersize=2,stri=2):
        super(tranconv, self).__init__()
        self.trans=nn.ConvTranspose2d(in_channels=inputc,out_channels=outc,kernel_size=kersize,stride=stri)

    def forward(self,x):
        x=self.trans(x)
        return  x

class  EESANet(nn.Module):
    def __init__(self,imagechan,chann,numclass):
        super(EESANet, self).__init__()
        self.SRBenc1=SRB(chann=imagechan,outchan=chann)
        self.SRBenc2 = SRB(chann=chann,outchan=2*chann)
        self.SRBenc3 = SRB(chann=2*chann,outchan=4*chann)
        self.SRBenc4 = SRB(chann=4*chann,outchan=8*chann)
        self.SRBNeck = SRB(chann=8*chann,outchan=16*chann)
        self.SRBdec1 = SRB(chann=16*chann,outchan=8*chann)
        self.SRBdec2 = SRB(chann=8*chann,outchan=4*chann)
        self.SRBdec3 = SRB(chann=4*chann,outchan=2*chann)
        self.SRBdec4 = SRB(chann=2*chann,outchan=chann)
        self.MaxPoolen1=maxpo()
        self.MaxPoolen2 = maxpo()
        self.MaxPoolen3 = maxpo()
        self.MaxPoolneck = maxpo()
        self.ASPPneck=ASPP(in_channel=16*chann,depth=16*chann)
        self.PSAneck=PSA(in_dim=16*chann,activation='relu')
        self.transcov1=tranconv(inputc=16*chann,outc=8*chann)
        self.transcov2 = tranconv(inputc=8*chann,outc=4*chann)
        self.transcov3=tranconv(inputc=4*chann,outc=2*chann)
        self.transcov4= tranconv(inputc=2*chann,outc=chann)
        self.eeb=EEB()
        self.conv11=nn.Conv2d(chann+1,numclass,kernel_size=1)
        self.softmax = nn.Softmax(1)


    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        outc = self.SRBenc1(x)
        outc_maxpool=self.MaxPoolen1(outc)
        out2c=self.SRBenc2(outc_maxpool)
        out2c_maxpool=self.MaxPoolen2(out2c)
        out4c=self.SRBenc3(out2c_maxpool)
        out4c_maxpool=self.MaxPoolen3(out4c)
        out8c=self.SRBenc4(out4c_maxpool)
        out8c_maxpool=self.MaxPoolneck(out8c)
        out16c=self.SRBNeck(out8c_maxpool)
        outaspp=self.ASPPneck(out16c)
        outpsa=self.PSAneck(outaspp)
        out8c_tran=self.transcov1(outpsa)

        out8c_cat=torch.cat((out8c_tran,out8c),dim=1)
        out8c_dec=self.SRBdec1(out8c_cat)
        out4c_tran=self.transcov2(out8c_dec)

        out4c_cat=torch.cat((out4c_tran,out4c),dim=1)
        out4c_dec=self.SRBdec2(out4c_cat)
        out2c_tran=self.transcov3(out4c_dec)

        out2c_cat=torch.cat((out2c_tran,out2c),dim=1)
        out2c_dec=self.SRBdec3(out2c_cat)
        outc_tran=self.transcov4(out2c_dec)

        outc_cat=torch.cat((outc_tran,outc),dim=1)
        out_dec=self.SRBdec4(outc_cat)

        out_eeb=self.eeb(x)
        out_cat=torch.cat((out_eeb,out_dec),dim=1)

        out_conv11=self.conv11(out_cat)
        out=self.softmax(out_conv11)


        return  out





# if __name__ == '__main__':
#     model = EESANet(imagechan=1,chann=48,numclass=16).cuda()
#
#     model.eval()
#     image = torch.randn(1, 1, 128, 128).cuda()
#     # m=nn.MaxPool2d((2,1))
#     # transp=nn.ConvTranspose2d(16,8,2).cuda()
#     # res=m(image)
#     # dec=torch.randn(1,16,256,256).cuda()
#     # res=torch.cat((dec,image),dim=1)
#     with torch.no_grad():
#         output = model.forward(image)
#     print(output.size())

