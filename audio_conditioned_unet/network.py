
import torch.nn as nn
import torch.nn.functional as F


class UNetModular(nn.Module):

    def __init__(self, config):
        super(UNetModular, self).__init__()

        self.config = config

        self.n_encoder_layers = self.config.get('n_encoder_layers', 4)
        self.n_filters_start = self.config.get('n_filters_start', 8)
        self.spec_out = self.config.get('spec_out', 128)
        self.dropout = self.config.get('dropout', False)

        activation = self.config.get("activation", "relu")

        if activation == 'relu':
            activation_class = nn.ReLU
            activation_fnc = F.relu
        elif activation == 'elu':
            activation_class = nn.ELU
            activation_fnc = F.elu
        else:
            raise NotImplemented("Invalid activation")

        self.perf_encoder = SpecEncoder(self.spec_out, activation=activation_class, dropout=self.dropout)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(1, self.n_encoder_layers+1):

            if i == 1:
                in_ = 1
                out_ = self.n_filters_start
            else:
                in_ = self.n_filters_start*(2**(i-2))
                out_ = self.n_filters_start*(2**(i-1))

            enc_block = UNetBlock(in_, out_, self.spec_out, film=config.get('film{}'.format(i), 'False'),
                                  activation=activation_fnc)

            dec_block = UNetBlock(out_, out_, self.spec_out,
                                  film=config.get('film{}'.format(2*(self.n_encoder_layers+1)-i), False),
                                  up_in_channels=out_*2, up_sample=True, down_sample=False, activation=activation_fnc)

            self.encoder.append(enc_block)
            self.decoder.append(dec_block)

        self.bottleneck_block = UNetBlock(self.n_filters_start*(2**(self.n_encoder_layers-1)),
                                          self.n_filters_start*(2**(self.n_encoder_layers)),
                                          self.spec_out, film=config.get('film{}'.format(self.n_encoder_layers+1), False),
                                          down_sample=False, activation=activation_fnc)

        self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

        self.first_execution = True

        self.apply(initialize_weights)

    def forward(self, score, perf):

        x = score

        perf = self.perf_encoder(perf)

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

        return x


class UNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, spec_out=128, film=True, down_sample=True,
                 up_sample=False, up_in_channels=1, activation=F.relu):
        super(UNetBlock, self).__init__()

        self.up_sample = up_sample
        self.down_sample = down_sample
        self.activation = activation
        self.film = film

        if self.up_sample:
            self.up_conv = nn.ConvTranspose2d(up_in_channels, in_channels, kernel_size=2, stride=2)

        if self.down_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                               stride=1, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                               stride=1, padding=padding)

        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.film:
            self.film_layer = FiLM(spec_out, out_channels)

    def forward(self, x, spec, residual=None):

        if self.up_sample:
            x = self.up_conv(x)

            if residual is not None:
                x = pad(x, residual.size()) + residual

        x = self.activation(self.bn1(self.conv1(x)), inplace=True)

        x = self.bn2(self.conv2(x))

        if self.film:
            x = self.film_layer(x, spec)

        x = self.activation(x, inplace=True)

        if self.down_sample:
            return x, self.max_pool(x)
        else:
            return x


class SpecEncoder(nn.Module):
    def __init__(self, spec_out, activation=nn.ELU, dropout=False):
        super(SpecEncoder, self).__init__()

        initial = 16
        self.enc = nn.Sequential(
            nn.Conv2d(1, initial, 3, 1), nn.BatchNorm2d(initial), activation(inplace=True),
            nn.Conv2d(initial, initial, 3, 1), nn.BatchNorm2d(initial), activation(inplace=True),

            nn.Conv2d(initial, initial*2, 3, 2), nn.BatchNorm2d(initial*2), activation(inplace=True),
            nn.Conv2d(initial*2, initial*2, 3, 1), nn.BatchNorm2d(initial*2), activation(inplace=True),
            nn.Dropout2d(0.2) if dropout else PassThroughLayer(),

            nn.Conv2d(initial*2, initial*4, 3, 2), nn.BatchNorm2d(initial*4), activation(inplace=True),
            nn.Conv2d(initial*4, initial*6, 3, 2), nn.BatchNorm2d(initial*6), activation(inplace=True),

            nn.Conv2d(initial*6, initial*6, 1, 1), nn.BatchNorm2d(initial*6), activation(inplace=True),
            nn.Dropout2d(0.2) if dropout else PassThroughLayer(),

            Flatten(),
            nn.Linear((initial*6)*7*3, spec_out), nn.BatchNorm1d(spec_out), activation(inplace=True),

        )

    def forward(self, x):

        return self.enc(x)


class FiLM(nn.Module):

    def __init__(self, zdim, maskdim):
        super(FiLM, self).__init__()

        self.gamma = nn.Linear(zdim, maskdim)
        self.beta = nn.Linear(zdim, maskdim)

    def forward(self, x, z):

        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta

        return x


class Flatten(nn.Module):
    # flatten function wrapped as a torch module
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class PassThroughLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def pad(x, target_shape):
    diffY = target_shape[2] - x.size()[2]
    diffX = target_shape[3] - x.size()[3]

    x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    return x
