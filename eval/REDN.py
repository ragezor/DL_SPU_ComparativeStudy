import torch.nn as nn
import torch.nn.functional as F
import functools
import torch

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(nn.BatchNorm2d(int(in_channels)),
                                          nn.ReLU(inplace=True),
                                          conv_mod)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """
    def __init__(self, prev_channels, out_channels, scale):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels

        self.conv1 = conv2DBatchNormRelu(prev_channels + 32, out_channels, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, k_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s*self.scale for _s in y_prime.shape[-2:]])
        #x = F.upsample(x, size=upsample_size, mode='nearest')
        x = F.interpolate(x, size=upsample_size, mode='nearest')
        z_prime = z + x

        return y_prime, z_prime

class RB(nn.Module):
    """
    Residual Unit for FRRN
    """
    def __init__(self, channels, kernel_size=3, strides=1):
        super(RB, self).__init__()

        self.conv1 = conv2DBatchNormRelu(channels, channels, k_size=1, stride=strides, padding=0)
        self.conv2 = conv2DBatchNormRelu(channels, channels, k_size=kernel_size, stride=strides, padding=1)
        self.conv3 = conv2DBatchNormRelu(channels, channels, k_size=1, stride=strides, padding=0)
        self.dropout = nn.Dropout2d(0.25)


    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return x + incoming



frrn_specs_dic = {
    'A':
        {
            'encoder': [[1, 48, 2],
                        [1, 96, 4],
                        [1, 192, 8],
                        [1, 192, 16]],

            'decoder': [[1, 96, 8],
                        [1, 96, 4],
                        [1, 48, 2]],
        },

    'B':
        {
            'encoder': [[3, 96, 2],
                        [4, 192, 4],
                        [2, 384, 8],
                        [2, 384, 16],
                        [2, 384, 32]],

            'decoder': [[2, 192, 16],
                        [2, 192, 8],
                        [2, 192, 4],
                        [2, 96, 2]],
        }, }


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=17, model_type='A'):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        #self.K = 64 * 512
        #self.loss = functools.partial(bootstrapped_cross_entropy2d, K=self.K)

        self.conv1 = conv2DBatchNormRelu(1, 32, 3, 1, 1)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(2):
            self.up_residual_units.append(RB(channels=32, kernel_size=3, strides=1))
            self.down_residual_units.append(RB(channels=32, kernel_size=3, strides=1))

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(32, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']

        # encoding
        prev_channels = 32
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        self.merge_conv = nn.Conv2d(prev_channels + 32, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        self.classif_conv = nn.Conv2d(32, self.n_classes,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(2):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        #z = self.split_conv(x)
        z = x

        prev_channels = 32
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            #y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear')
            y_upsampled = F.interpolate(y, size=upsample_size, mode='bilinear',align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([F.interpolate(y, scale_factor=2, mode='bilinear',align_corners=True), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(2):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x
class frrn_blender(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=18, model_type='A'):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        #self.K = 64 * 512
        #self.loss = functools.partial(bootstrapped_cross_entropy2d, K=self.K)

        self.conv1 = conv2DBatchNormRelu(1, 32, 3, 1, 1)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(2):
            self.up_residual_units.append(RB(channels=32, kernel_size=3, strides=1))
            self.down_residual_units.append(RB(channels=32, kernel_size=3, strides=1))

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(32, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]['encoder']

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]['decoder']

        # encoding
        prev_channels = 32
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels, out_channels=channels, scale=scale))
            prev_channels = channels

        self.merge_conv = nn.Conv2d(prev_channels + 32, 32,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    bias=True)

        self.classif_conv = nn.Conv2d(32, self.n_classes,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      bias=True)

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(2):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        #z = self.split_conv(x)
        z = x

        prev_channels = 32
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['encoding_frru', n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            #y_upsampled = F.upsample(y, size=upsample_size, mode='bilinear')
            y_upsampled = F.interpolate(y, size=upsample_size, mode='bilinear',align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = '_'.join(map(str, ['decoding_frru', n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([F.interpolate(y, scale_factor=2, mode='bilinear',align_corners=True), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(2):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x