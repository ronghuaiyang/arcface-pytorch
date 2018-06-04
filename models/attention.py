# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

from resbottleneck import ResBottleNeck


__all__ = ['AttentionNet']


class AttentionNet(nn.Module):

    def __init__(self, num_classes=1, init_weights=True, input_shape=(3, 64, 64), batch_norm=True):
        super(AttentionNet, self).__init__()

        sub_sample_cnt = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        sub_sample_cnt += 1

        self.attention_model1 = AttentionModule(64, 128, 2)
        sub_sample_cnt += 1

        self.attention_model2 = AttentionModule(128, 256, 2)
        sub_sample_cnt += 1

        self.attention_model3 = AttentionModule(256, 512, 2)
        sub_sample_cnt += 1

        sub_sample_ratio = 2 ** sub_sample_cnt
        # print('sub_sample_ratio', sub_sample_ratio)
        sub_sample_shape = (input_shape[1] / sub_sample_ratio, input_shape[2] / sub_sample_ratio)
        # print(sub_sample_shape)
        self.fc_feature = nn.Linear(512 * sub_sample_shape[0] * sub_sample_shape[1], 512)

        self.fc = nn.Linear(512, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.attention_model1(x)
        x = self.attention_model2(x)
        x = self.attention_model3(x)
        x = nn.AvgPool2d(kernel_size=x.size()[2:])(x)
        # x = x.view(x.size(0), -1)
        # x = nn.Dropout()(x)
        # x = self.fc_feature(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout()(x)
        output = self.fc(x)

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

    @staticmethod
    def get_output_size(shape, features):
        batch_size = 1  # Not important.
        input = torch.rand(batch_size, *shape)
        output_feat = features(input)
        output_size = output_feat.size()[1:]
        return output_size
    
    
class SoftMaskBranch(nn.Module):

    def __init__(self, channels):
        super(SoftMaskBranch, self).__init__()

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ResBottleNeck(channels, channels)
        self.upool = nn.Upsample(scale_factor=2)

        self.conv = nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=1),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mp1 = self.mpool(x)
        block1 = self.block(mp1)

        mp2 = self.mpool(block1)
        block2 = self.block(mp2)

        up2 = self.upool(block2)
        print(up2.size(), block1.size())
        up2 += block1
        block3 = self.block(up2)

        block4 = self.upool(block3)
        block4 += x

        out = self.conv(block4)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out


class TrunkBranch(nn.Module):

    def __init__(self, in_channels, block_num=2):
        super(TrunkBranch, self).__init__()

        self.resblock = ResBottleNeck(in_channels, in_channels)
        self.block_num = block_num

    def forward(self, x):
        for i in range(self.block_num):
            x = self.resblock(x)
        return x


class AttentionModule(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super(AttentionModule, self).__init__()

        self.resblock1 = ResBottleNeck(in_channels, out_channels, stride)
        self.trunk_branch = TrunkBranch(out_channels)
        self.soft_mask_branch = SoftMaskBranch(out_channels)
        self.resblock2 = ResBottleNeck(out_channels, out_channels)

    def forward(self, x):
        x = self.resblock1(x)
        trunk_out = self.trunk_branch(x)
        soft_mask_out = self.soft_mask_branch(x)
        out = (1 + soft_mask_out) * trunk_out
        out = self.resblock2(out)
        return out





