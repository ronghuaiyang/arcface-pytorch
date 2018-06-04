# -*- coding: utf-8 -*-
import torch.nn as nn


class ResBottleNeck(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResBottleNeck, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(input_channels, output_channels / 4, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels / 4)

        self.conv2 = nn.Conv2d(output_channels / 4, output_channels / 4, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels / 4)

        self.conv3 = nn.Conv2d(output_channels / 4, output_channels, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        self.bn4 = nn.BatchNorm2d(output_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(x)
            residual = self.bn4(residual)

        out += residual

        out = self.relu(out)
        return out