import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import math


__all__ = ['FasNet']


class FasNet(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True, input_shape=(3, 224, 224), batch_norm=True):
        super(FasNet, self).__init__()
        # layers = []
        # in_channels = input_shape[0]
        # out_channels = 64
        # for i in range(2):
        #     conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #     if batch_norm:
        #         layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #     else:
        #         layers += [conv2d, nn.ReLU(inplace=True)]
        #
        # layers += [nn.MaxPool2d(stride=2)]
        #
        # in_channels = out_channels
        # out_channels = 128
        # for i in range(2):
        #     conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #     if batch_norm:
        #         layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #     else:
        #         layers += [conv2d, nn.ReLU(inplace=True)]
        #
        # layers += [nn.MaxPool2d(stride=2)]
        #
        # in_channels = out_channels
        # out_channels = 256
        # for i in range(3):
        #     conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #     if batch_norm:
        #         layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #     else:
        #         layers += [conv2d, nn.ReLU(inplace=True)]
        #
        # feature_1 = nn.Sequential(*layers)
        # layers += [nn.MaxPool2d(stride=2)]
        #
        # in_channels = out_channels
        # out_channels = 512
        # for i in range(3):
        #     conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #     if batch_norm:
        #         layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #     else:
        #         layers += [conv2d, nn.ReLU(inplace=True)]
        #
        # feature_2 = nn.Sequential(*layers)
        # layers += [nn.MaxPool2d(stride=2)]

        # in_channels = out_channels
        # out_channels = 512
        # for i in range(3):
        #     conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #     if batch_norm:
        #         layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #     else:
        #         layers += [conv2d, nn.ReLU(inplace=True)]
        #
        # feature_3 = nn.Sequential(*layers)

        # vgg_16 = [2, 2, 3, 3]
        # channels_num = [64, 'M', 128, 256, 512]
        #
        # layers = []
        # features = []
        # in_channels = 3
        # for k, stack_num in enumerate(vgg_16):
        #     out_channels = channels_num[k]
        #     for i in range(stack_num):
        #         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #         if batch_norm:
        #             layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        #         else:
        #             layers += [conv2d, nn.ReLU(inplace=True)]
        #     in_channels = out_channels
        #     if k > 1:
        #         features.append(nn.Sequential(*layers))
        #
        #     if k != len(vgg_16) - 1:
        #         layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #
        # feat_size_0 = self.get_conv_output_size(input_shape, features[-1])
        # feat_size_1 = self.get_conv_output_size(input_shape, features[-2])
        #
        # self.embed_0 = nn.Sequential(
        #     nn.Linear(feat_size_0[0] * feat_size_0[1] * feat_size_0[2], 256),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.embed_1 = nn.Sequential(
        #     nn.Linear(feat_size_1[0] * feat_size_1[1] * feat_size_1[2], 256),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.features = features
        #
        # self.classify = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes)
        # )



        # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.base = make_layers(cfg, batch_norm=batch_norm)

        output_shape = self.get_conv_output_size(input_shape, self.base)
        self.feature_1 = nn.Sequential(
            nn.Linear(output_shape[0] * output_shape[1] * output_shape[2], 256),
            nn.ReLU(inplace=True)
        )

        self.conv5_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        output_shape = self.get_conv_output_size(output_shape, self.conv5_block)
        self.feature_2 = nn.Sequential(
            nn.Linear(output_shape[0] * output_shape[1] * output_shape[2], 256),
            nn.ReLU(inplace=True)
        )

        self.classify = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def get_conv_output_size(shape, features):
        batch_size = 1  # Not important.
        input = Variable(torch.rand(batch_size, *shape), requires_grad=False)
        output_feat = features(input)
        output_size = output_feat.size()[1:]
        return output_size

    def forward(self, x):
        x = self.base(x)

        x_1 = x.view(x.size(0), -1)
        x_1 = self.feature_1(x_1)

        x_2 = self.conv5_block(x)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2 = self.feature_2(x_2)

        feature = torch.cat([x_1, x_2], 1)
        output = self.classify(feature)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

