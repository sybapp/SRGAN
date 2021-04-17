import math
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.swish(residual)
        residual = self.conv2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class UpsampleBLock_OSRGAN(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock_OSRGAN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.swish = Swish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.swish(x)
        return x


class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.residual_blocks_num = 5
        self.upsample_block_num = int(math.log(scale_factor, 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        for i in range(self.residual_blocks_num):
            self.add_module('residual' + str(i + 1), ResidualBlock(64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        for i in range(self.upsample_block_num):
            self.add_module('upsample' + str(i + 1), UpsampleBLock(64, 2))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        cache = x.clone()

        for i in range(self.residual_blocks_num):
            cache = self.__getattr__('residual' + str(i + 1))(cache)

        x = self.conv2(cache) + x

        for i in range(self.upsample_block_num):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        batch_size = x.size(0)
        return torch.sigmoid(x.view(batch_size))


class Generator_OSRGAN(nn.Module):
    def __init__(self, scale_factor):
        super(Generator_OSRGAN, self).__init__()
        self.residual_blocks_num = 5
        self.upsample_block_num = int(math.log(scale_factor, 2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            Swish()
        )

        for i in range(self.residual_blocks_num):
            self.add_module('residual' + str(i + 1), RRDB(64))

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        for i in range(self.upsample_block_num):
            self.add_module('upsample' + str(i + 1), UpsampleBLock(64, 2))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        cache = x.clone()

        for i in range(self.n_residual_blocks):
            cache = self.__getattr__('residual' + str(i + 1))(cache)

        x = self.conv2(cache) + x

        for i in range(self.upsample_block_num):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)


class Discriminator_OSRGAN(nn.Module):
    def __init__(self):
        super(Discriminator_OSRGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            Swish(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            Swish(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            Swish(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            Swish(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            Swish(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            Swish(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            Swish(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            Swish(),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            Swish(),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.net(x)
        batch_size = x.size(0)
        return torch.sigmoid(x.view(batch_size))


if __name__ == '__main__':
    print(Generator_OSRGAN(8))
