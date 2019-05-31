import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
class ChannelSqueezeAndExcitation(nn.Module):
    def __init__(self, in_ch, r):
        super(ChannelSqueezeAndExcitation, self).__init__()

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


class SpatialSqueezeAndExcitation(nn.Module):
    def __init__(self, in_ch):
        super(SpatialSqueezeAndExcitation, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class SpatialAndChannelSqueezeAndExcitation(nn.Module):
    def __init__(self, in_ch, r):
        super(SpatialAndChannelSqueezeAndExcitation, self).__init__()

        self.ChannelSqueezeAndExcitation = ChannelSqueezeAndExcitation(in_ch, r)
        self.SpatialSqueezeAndExcitation = SpatialSqueezeAndExcitation(in_ch)

    def forward(self, x):
        cse = self.ChannelSqueezeAndExcitation(x)
        sse = self.SpatialSqueezeAndExcitation(x)

        x = torch.add(cse, sse)

        return x


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(
            x
        )  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)

        return x + noise
