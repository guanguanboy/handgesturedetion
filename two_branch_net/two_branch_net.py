import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class TwoBranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        b1_infrad = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2_infrad = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3_infrad = nn.Sequential(*resnet_block(64, 128, 2))
        b4_infrad = nn.Sequential(*resnet_block(128, 256, 2))
        b5_infrad = nn.Sequential(*resnet_block(256, 512, 2))

        self.rgb_net = nn.Sequential(b1, b2, b3, b4, b5)
        self.infrad_net = nn.Sequential(b1_infrad, b2_infrad, b3_infrad, b4_infrad, b5_infrad)

        self.prediction_head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

    def forward(self, rgb_input, infrad_input):
        rgb_feature =  self.rgb_net(rgb_input)
        infrad_feature = self.infrad_net(infrad_input)

        final_feature = rgb_feature + infrad_feature

        output = self.prediction_head(final_feature)

        return output


def test_TwoBranchNet():
    net = TwoBranchNet()

    rgb_input = torch.randn(2, 3, 224, 224)
    infrad = torch.randn(2, 3, 224, 224)

    output = net(rgb_input, infrad)
    print(output.shape)

if __name__ == "__main__":
    test_TwoBranchNet()