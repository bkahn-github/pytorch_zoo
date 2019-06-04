import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitation(nn.Module):
    """The channel-wise SE (Squeeze and Excitation) block from the [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.

    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939 and https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    
    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r=16):
        super(SqueezeAndExcitation, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class ChannelSqueezeAndSpatialExcitation(nn.Module):
    """The sSE (Channel Squeeze and Spatial Excitation) block from the 
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178
    
    Args:
        in_ch (int): The number of channels in the feature map of the input.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch):
        super(ChannelSqueezeAndSpatialExcitation, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(nn.Module):
    """The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation) block from the 
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.
    
    Implementation adapted from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Args:
        in_ch (int): The number of channels in the feature map of the input.
        r (int): The reduction ratio of the intermidiate channels.
                Default: 16.

    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_ch, r):
        super(ConcurrentSpatialAndChannelSqueezeAndChannelExcitation, self).__init__()

        self.SqueezeAndExcitation = SqueezeAndExcitation(in_ch, r)
        self.ChannelSqueezeAndSpatialExcitation = ChannelSqueezeAndSpatialExcitation(
            in_ch
        )

    def forward(self, x):
        cse = self.SqueezeAndExcitation(x)
        sse = self.ChannelSqueezeAndSpatialExcitation(x)

        x = torch.add(cse, sse)

        return x


class GaussianNoise(nn.Module):
    """A gaussian noise module.

    Args:
        stddev (float): The standard deviation of the normal distribution.
                        Default: 0.1.

    Shape:
        - Input: (batch, *)
        - Output: (batch, *) (same shape as input)
    """

    def __init__(self, stddev=0.1):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)

        return x + noise
