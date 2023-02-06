import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CLIPEnc(nn.Module):
    def __init__(self, hdims=[], ):
        super().__init__()
        layers = []
        for i in range(len(hdims)):
            if i == 0:
                block = nn.Sequential(
                    nn.Conv2d(3, hdims[0], kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(hdims[i-1], hdims[i], kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                )
            layers.append(block)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class TimmEnc(nn.Module):
    def __init__(self ):
        super().__init__()
        self.model = timm.create_model("resnet18", num_classes = 0, pretrained=True)
        self.fc = torch.nn.Linear(512, 12*4*4)


    def forward(self, x):
        emb = self.model(x)
        x = self.fc(emb)
        x = x.reshape(x.shape[0], 12, 4, 4)
        return x

class Encoder(nn.Module):
    def __init__(self, hdims=[], ):
        super().__init__()
        layers = []
        for i in range(len(hdims)):
            if i == 0:
                block = nn.Sequential(
                    nn.Conv2d(3, hdims[0], kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(hdims[i-1], hdims[i], kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                )
            layers.append(block)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class Decoder(nn.Module):
    def __init__(self, hdims=[], kernel_size=2, stride=2, padding=0):
        super().__init__()
        layers = []
        for i in reversed(range(len(hdims))):
            if i == 0:
                block = nn.Sequential(
                    nn.ConvTranspose2d(hdims[0], 3, kernel_size=kernel_size, stride=stride, padding=padding),
                )
            else:
                block = nn.Sequential(
                    nn.ConvTranspose2d(hdims[i], hdims[i-1], kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.Tanh()
                )
            layers.append(block)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class SyntheticImageGenerator(nn.Module):
    def __init__(self, num_classes, img_size=(32,32), num_seed_vec=16, num_decoder=8, hdims=[],
        kernel_size=2, stride=2, padding=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_decoder = num_decoder

        h = int(img_size[0] / 2**len(hdims))
        w = int(img_size[1] / 2**len(hdims))
        self.seed_shape = (hdims[-1], h, w)

        encoders_list = []
        decoders_list = []
        for _ in range(num_decoder):
            #encoders_list.append(Encoder(hdims))
            encoders_list.append(TimmEnc())
            decoders_list.append(Decoder(hdims, kernel_size, stride, padding))
        self.encoders = nn.ModuleList(encoders_list)
        self.decoders = nn.ModuleList(decoders_list)

        self.seed_vec = (nn.Parameter(0.1*torch.randn(
            (num_classes, num_seed_vec) + self.seed_shape), requires_grad=True))

    def broadcast_decoder(self):
        for c in range(1, self.num_decoder):
            for w, w0 in zip(self.decoders[c].parameters(), self.decoders[0].parameters()):
                w.data.copy_(w0.data)

    def get_all_cpu(self):
        x_list, y_list = [], []
        for c in range(self.num_classes):
            x, y = self.get_sample(c)
            x, y = x.detach().cpu(), y.detach().cpu()
            x_list.append(x)
            y_list.append(y)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return x, y

    def get_all(self):
        x_list, y_list = [], []
        for c in range(self.num_classes):
            x, y = self.get_sample(c)
            x_list.append(x)
            y_list.append(y)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        return x, y

    def get_sample(self, c):
        seed_vec = self.seed_vec[c]

        x_list = []
        for decoder in self.decoders:
            x = decoder(seed_vec)
            x_list.append(x)
        x = torch.cat(x_list, dim=0)
        y = torch.LongTensor([c]*x.shape[0]).to(x.device)
        return x, y

    def get_subsample_decoder(self, c, decoder_index=[0]):
        seed_vec = self.seed_vec[c]

        x_list = []
        for index in decoder_index:
            x = self.decoders[index](seed_vec)
            x_list.append(x)
        x = torch.cat(x_list, dim=0)
        return x

    def autoencoder(self, x):
        x_hat = self.decoders[0](self.encoders[0](x))
        return F.mse_loss(x_hat, x)
