import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


import pandas as pd
import numpy as np
from sklearn.utils import resample

from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d, ReLU, Softmax, CrossEntropyLoss, Sequential, Dropout, Conv2d, Linear

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
from brevitas.core.restrict_val import RestrictValueType
from tensor_norm import TensorNorm
from common import CommonWeightQuant, CommonActQuant



CNV_OUT_CH_POOL = [(21, False), (21, True), (21, False)]#, (128, True), (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(3549, 16), (16, 16)]
LAST_FC_IN_FEATURES = 16
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 6

class CNV_i(Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNV_i, self).__init__()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=in_bit_width,
                                                min_val=- 1.0, max_val=1.0 - 2.0 ** (-7), narrow_range=False,
                                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO))
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(QuantConv2d(kernel_size=KERNEL_SIZE, in_channels=in_ch, out_channels=out_ch,
                                                  bias=True, padding=4, weight_quant=CommonWeightQuant,
                                                  weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width))
            if is_pool_enabled:#
                self.conv_features.append(MaxPool2d(kernel_size=2))
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(in_features=in_features, out_features=out_features, bias=True,
                                                    weight_quant=CommonWeightQuant,weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(act_quant=CommonActQuant,bit_width=act_bit_width))
        
        self.linear_features.append(QuantLinear(in_features=LAST_FC_IN_FEATURES, out_features=num_classes,
                                                    bias=False, weight_quant=CommonWeightQuant,
                                                    weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)
    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)
    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x
    
    
    
class CNVdp(Module):
    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch):
        super(CNVdp, self).__init__()
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(QuantIdentity(act_quant=CommonActQuant,bit_width=in_bit_width, min_val=- 1.0,
                                                max_val=1.0 - 2.0 ** (-7), narrow_range=False,
                                                restrict_scaling_type=RestrictValueType.POWER_OF_TWO))
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(Conv2d(kernel_size=KERNEL_SIZE, in_channels=in_ch, out_channels=out_ch, 
                                             bias=True, padding=4))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(act_quant=CommonActQuant,bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(Linear(in_features=in_features,out_features=out_features, bias=True))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(act_quant=CommonActQuant,bit_width=act_bit_width))
        self.linear_features.append(Linear(in_features=LAST_FC_IN_FEATURES,out_features=num_classes, bias=False))
        self.linear_features.append(TensorNorm())
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)
    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, Conv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, Linear):
                mod.weight.data.clamp_(min_val, max_val)
    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x