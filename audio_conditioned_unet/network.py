
import torch

import torch.nn as nn
import torch.nn.functional as F

from audio_conditioned_unet import audio_encoder


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


def pad(x, target_shape):
    diffY = target_shape[2] - x.size()[2]
    diffX = target_shape[3] - x.size()[3]

    x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    return x


class FiLM(nn.Module):

    def __init__(self, zdim, maskdim):
        super(FiLM, self).__init__()

        self.gamma = nn.Linear(zdim, maskdim)   # s
        self.beta = nn.Linear(zdim, maskdim)    # t

    def forward(self, x, z):

        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta

        return x


class ConditionalUNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, spec_out=128, film=True, down_sample=True, up_sample=False,
                 up_in_channels=1, padding=1, no_skip=False):
        super(ConditionalUNetBlock, self).__init__()

        self.up_sample = up_sample
        self.down_sample = down_sample
        self.film = film

        self.no_skip = no_skip

        self.in_channels = in_channels

        if self.up_sample:
            self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                         nn.Conv2d(up_in_channels, in_channels, kernel_size=1, stride=1))

        if self.down_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)

        # using only one group is equivalent to using layer norm but this way it is more convenient to implement
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)

        if self.film:
            self.film_layer = FiLM(spec_out, out_channels)

    def forward(self, x, spec, residual=None):

        if self.up_sample:
            x = self.up_conv(x)

            if residual is not None:
                x = pad(x, residual.size())

                if not self.no_skip:
                    x = x + residual

        x = F.elu(self.norm1(self.conv1(x)))

        x = self.norm2(self.conv2(x))

        if self.film:
            x = self.film_layer(x, spec)

        x = F.elu(x)

        if self.down_sample:
            return x, self.max_pool(x)
        else:
            return x


class ConditionalUNet(nn.Module):

    def __init__(self, config):
        super(ConditionalUNet, self).__init__()

        self.config = config

        self.n_encoder_layers = self.config.get('n_encoder_layers', 4)
        self.n_filters_start = self.config.get('n_filters_start', 8)

        self.use_lstm = self.config.get("use_lstm", False)

        self.max_channel = 128

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.rnn_size = self.config.get('rnn_size', 512)
        self.rnn_layers = self.config.get('rnn_layer', 1)

        self.spec_enc = self.config.get('spec_enc', 512)

        self.perf_encoder = getattr(audio_encoder, config['audio_encoder'])(self.spec_enc)

        if self.use_lstm:
            self.rnn = nn.LSTM(self.spec_enc, hidden_size=self.rnn_size, num_layers=self.rnn_layers, batch_first=False)
        else:
            self.fc = nn.Linear(self.spec_enc, self.rnn_size)

        film_layers = config['film_layers']
        for i in range(1, self.n_encoder_layers+1):

            if i == 1:
                in_ = 1
                out_ = min(self.n_filters_start, self.max_channel)
            else:
                in_ = min(self.n_filters_start*(2**(i-2)), self.max_channel)
                out_ = min(self.n_filters_start*(2**(i-1)), self.max_channel)

            enc_block = ConditionalUNetBlock(in_, out_, self.rnn_size, film=i in film_layers)

            dec_block = ConditionalUNetBlock(out_, out_, self.rnn_size,
                                             film=2*(self.n_encoder_layers+1)-i in film_layers,
                                             up_in_channels=min(out_*2, self.max_channel), up_sample=True,
                                             down_sample=False)

            self.encoder.append(enc_block)
            self.decoder.append(dec_block)

        self.bottleneck_block = ConditionalUNetBlock(min(self.n_filters_start * (2 ** (self.n_encoder_layers - 1)), self.max_channel),
                                                     min(self.n_filters_start*(2**(self.n_encoder_layers)), self.max_channel),
                                                     self.rnn_size, film=self.n_encoder_layers+1 in film_layers,
                                                     down_sample=False)

        self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

        self.first_execution = True

        self.apply(initialize_weights)

    def forward(self, score, perf, hidden):

        x = score
        seq_len, bs, c, h, w = score.shape
        x = x.view(seq_len*bs, c, h, w)

        perf = self.perf_encoder(perf)

        if self.use_lstm:
            # use rnn for context vector
            perf = perf.view(seq_len, bs, -1)
            perf, hidden = self.rnn(perf, hidden)
            perf = perf.view(seq_len * bs, -1)
        else:
            # use fully connected layer as context vector (no temporal context)
            perf = F.elu(self.fc(perf))

        residuals = []
        for i in range(self.n_encoder_layers):

            res, x = self.encoder[i](x, perf)
            residuals.append(res)
            if self.first_execution:
                print('down', x.shape)

        x = self.bottleneck_block(x, perf)

        if self.first_execution:
            print('bottleneck', x.shape)

        # walk in reverse through the decoder
        for i in range(self.n_encoder_layers)[::-1]:

            x = self.decoder[i](x, perf, residuals[i])

            if self.first_execution:
                print('up', x.shape)

        x = self.conv_out(x)

        if self.first_execution:
            print('out', x.shape)
            self.first_execution = False

        x = torch.sigmoid(x)
        model_returns = {'segmentation': x, 'hidden': hidden}

        return model_returns
