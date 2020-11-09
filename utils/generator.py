import math 

import torch
import torch.nn as nn


class tconv_bn_relu(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, pad):
        self.tconv = nn.ConvTranspose2d(input_channel, 
                                        output_channel, 
                                        kernel_size=kernel_size, 
                                        stride=stride, 
                                        pad=pad, 
                                        bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class fc_bn_relu(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(fc_bn_relu, self).__init__()
        self.stage = nn.Sequential(
                    nn.Linear(input_channel, output_channel),
                    #nn.BatchNorm1d(output_channel),
                    nn.ReLU(True)
                )
    def forward(self, x):
        x = self.stage(x)
        return x


class Generator(nn.Module):
    def __init__(self, hc_dim, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hc_dim = hc_dim

        self.encoder = nn.Sequential(
                        fc_bn_relu(input_dim, output_dim//2),
                        fc_bn_relu(output_dim//2, output_dim//4),
                        fc_bn_relu(output_dim//4, output_dim//6)
                     )
        self.decoder = nn.Sequential(
                        fc_bn_relu(output_dim//6+self.hc_dim, output_dim//4),
                        fc_bn_relu(output_dim//4, output_dim//2),
                        fc_bn_relu(output_dim//2, output_dim)
                     )
        self._initialize_weights()
        
    def forward(self, x, hc):
        x = x.view(1, -1)
        #x = x.expand(*x.shape[:-1], self.input_dim)
        x = self.encoder(x)

        hc = hc.expand(*hc.shape[:-1], self.hc_dim)
        x = torch.cat((x, hc), 1)
        x = self.decoder(x)
        return x
    
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()  

# ===================== Conv ==============================

class ConvRelu(nn.Sequential):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad):
        super(ConvRelu, self).__init__()
        
        self.add_module("conv", nn.Conv2d(input_channel, output_channel, kernel, stride, pad, bias=False))
        self.add_module("relu", nn.ReLU())

class TConvRelu(nn.Sequential):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel,
                 stride,
                 pad):
        super(TConvRelu, self).__init__()
        self.add_module("tconv", nn.ConvTranspose2d(input_channel, output_channel, kernel, stride, pad, bias=False))
        self.add_module("relu", nn.ReLU())

class ConvGenerator(nn.Module):
    def __init__(self, hc_dim, input_dim, hidden_dim):
        super(ConvGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hc_dim = hc_dim

        self.encoder = nn.Sequential(
                        ConvRelu(self.input_dim, hidden_dim//4, 3, 1, 1),
                        ConvRelu(hidden_dim//4, hidden_dim//2, 3, 2, 1),
                        ConvRelu(hidden_dim//2, hidden_dim, 3, 1, 1),
                        
                     )
        self.decoder = nn.Sequential(
                        TConvRelu(hidden_dim+self.hc_dim, hidden_dim//2, 3, 1, 1),
                        TConvRelu(hidden_dim//2, hidden_dim//4, 4, 2, 1),
                        TConvRelu(hidden_dim//4, self.input_dim, 3, 1, 1),
                     )
        self._initialize_weights()
        
    def forward(self, x, hc):
        x = x.view(1, 1, *x.shape)
        x = self.encoder(x)

        hc = hc.expand(*hc.shape[:-1], self.hc_dim*x.size(-1)*x.size(-2))
        hc = hc.view(1, self.hc_dim, *x.shape[-2:])
        x = torch.cat((x, hc), 1)
        x = self.decoder(x)
        return x
    
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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()  

class SinGenerator(nn.Module):
    def __init__(self, hc_dim, input_dim, layer_nums=5, hidden_dim=32, min_hidden_dim=32):
        super(SinGenerator, self).__init__()
        N = hidden_dim
        self.hc_dim = hc_dim

        self.hc_head = nn.Sequential(
                        ConvRelu(hc_dim, N, 3, 1, 1),
                        ConvRelu(N, N, 3, 1, 1),
                        ConvRelu(N, N, 3, 1, 1))
        self.bb_head = nn.Sequential(
                        ConvRelu(input_dim, N, 3, 1, 1),
                        ConvRelu(N, N, 3, 1, 1))

        self.head = ConvRelu(N, N, 3, 1, 1)

        self.body = nn.Sequential()
        for i in range(layer_nums-2):
            N = int(hidden_dim / pow(2, i+1))
            block = ConvRelu(max(2*N, min_hidden_dim), max(N, min_hidden_dim), 3, 1, 1)
            self.body.add_module("block%d"%(i+1), block)
        self.tail = nn.Sequential(
                    nn.Conv2d(max(N, min_hidden_dim), input_dim, 3, 1, 1)
                )

        self._initialize_weights()

    def forward(self, x, hc, noise):
        y = x.view(1, 1, *x.shape)
        y += noise

        hc = hc.expand(*hc.shape[:-1], self.hc_dim*x.size(-1)*x.size(-2))
        hc = hc.view(1, self.hc_dim, *x.shape[-2:])
        hc_test = self.hc_head(hc)

        y = self.bb_head(y)
        #prior_y = torch.cat((y, hc_test), 1)
        prior_y = y + hc_test
        
        y_head = self.head(prior_y)
        y = self.body(y_head)
        y = y + prior_y
        y = self.tail(y)

        return y

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()  

def get_generator(CONFIG, arch_param_nums=None):
    generator = None
    if CONFIG.generator == "singan":
        generator = SinGenerator(CONFIG.hc_dim, 1)
    elif CONFIG.generator == "convgan":
        generator = ConvGenerator(CONFIG.hc_dim, 1, CONFIG.hc_dim)
    elif CONFIG.get_generator == "gan":
        generator = Generator(CONFIG.hc_dim, arch_param_nums, arch_param_nums)

    return generator




        

