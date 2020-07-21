
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)

        # using only one group is equivalent to using layer norm but this way it is more convenient to implement
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        return F.elu(self.norm(self.conv(x)))


class Flatten(nn.Module):
    # flatten function wrapped as a torch module
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class FBEncoder(nn.Module):
    def __init__(self, spec_enc):
        super(FBEncoder, self).__init__()

        self.n_input_frames = 1
        self.enc = nn.Linear(78, spec_enc)
        self.norm = nn.LayerNorm(spec_enc)

        # we make these non trainable parameters of network such that they are stored with the model
        # they can be set to the mean/std of the dataset
        self.means = nn.Parameter(torch.zeros(78), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(78), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means), requires_grad=False)
        self.stds = nn.Parameter(torch.from_numpy(stds), requires_grad=False)

    def forward(self, x):

        x = self.reshape_input(x)

        x = (x - self.means)/self.stds

        return F.elu(self.norm(self.enc(x)))

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape
        return x.view(seq_len * bs, -1)


class CBEncoder(FBEncoder):
    def __init__(self, spec_enc):
        super(CBEncoder, self).__init__(spec_enc)

        self.n_input_frames = 40
        initial = 24

        self.enc = nn.Sequential(
            ConvBlock(1, initial, 3, 1, padding=1),
            ConvBlock(initial, initial, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial, initial * 2, 3, 1, padding=1),
            ConvBlock(initial * 2, initial * 2, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 2, initial * 4, 3, 1, padding=1),
            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            ConvBlock(initial * 4, initial * 4, 3, 1, padding=1),
            nn.MaxPool2d(2),

            ConvBlock(initial * 4, initial * 4, 1, 1),
            Flatten(),
            nn.Linear(initial * 4 * 4 * 2, spec_enc)
        )

        # we make these non trainable parameters of network such that they are stored with the model
        # they can be set to the mean/std of the dataset
        self.means = nn.Parameter(torch.zeros(78).unsqueeze(0).unsqueeze(-1), requires_grad=False)
        self.stds = nn.Parameter(torch.ones(78).unsqueeze(0).unsqueeze(-1), requires_grad=False)

    def set_stats(self, means, stds):
        self.means = nn.Parameter(torch.from_numpy(means).unsqueeze(0).unsqueeze(-1), requires_grad=False)
        self.stds = nn.Parameter(torch.from_numpy(stds).unsqueeze(0).unsqueeze(-1), requires_grad=False)

    def reshape_input(self, x):
        seq_len, bs, c, h, w = x.shape
        return x.view(seq_len * bs, c, h, w)

