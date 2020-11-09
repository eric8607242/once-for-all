import logging
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import ConvBNRelu, Flatten, conv_1x1_bn, MBConv
from utils.countmacs import MAC_Counter

class LookUpTable:
    """
    LookupTable which saved the macs of each block.
    Calculate the approximate macs of arch param(probability arch param or one hot arch param).
    """
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.cnt_layers = len(CONFIG.l_cfgs)

        self.basic_macs = self._calculate_basic_operations()
        self.depthwise_macs = self._calculate_depthwise_operations()
        self.pointwise_macs = self._calculate_pointwise_operations()

    def get_model_macs(self, arch_param):
        """
        Arg:
            arch_param(layer_nums, (split_block**2+1)*expansion_nums)
        """
        split_block_nums = self.CONFIG.split_blocks

        model_macs = []
        for l, (l_arch, l_cfg) in enumerate(zip(arch_param, self.CONFIG.l_cfgs)):
            expansion, output_channel, kernels, stride, split_block, se = l_cfg
            layer_macs = []

            skip_weight = torch.tensor(0.0).cuda()
            for b in range(expansion):
                split_arch_param = l_arch[b*split_block:(b+1)*split_block]
                layer_macs.append(self._get_depthwise_macs(split_arch_param, l, kernels))
                if "skip" in kernels:
                    skip_weight += split_arch_param[-1] / 6

            model_macs.append(torch.sum(torch.stack(layer_macs)))
            model_macs.append((1-skip_weight)*self.pointwise_macs[str(l)]["point_1"] if skip_weight.item() < 5/6 else torch.tensor(0.0).cuda())
            model_macs.append((1-skip_weight)*self.pointwise_macs[str(l)]["point_2"])

        return torch.sum(torch.stack(model_macs)) + self.basic_macs

    def _get_depthwise_macs(self, block_probability, layer_num, kernels):
        macs = 0
        for k, p in zip(kernels, block_probability):
            block_macs = self.depthwise_macs[str(layer_num)]["k{}".format(k)]

            macs += block_macs * p if p > 1e-2 else torch.tensor(0.0).cuda()
        return macs

    def _calculate_pointwise_operations(self, write_to_file=None):
        lookup_table_operation = {}

        input_channel = 16
        input_size = self.CONFIG.input_size \
                        if self.CONFIG.dataset[:5] == "cifar" else self.CONFIG.input_size//2
        for l, l_cfg in enumerate(self.CONFIG.l_cfgs):
            layers_macs = {}

            expansion, output_channel, kernels, stride, split_block, se = l_cfg
            mid_channel = input_channel * expansion

            point_wise_1_macs = 0
            if input_channel != mid_channel:
                point_wise_1 = nn.Sequential(
                                ConvBNRelu(input_channel, 
                                           mid_channel,
                                           kernel=1,
                                           stride=1,
                                           pad=0,
                                           activation="relu",
                                           group=1)
                                )
                point_wise_1_macs = self._calculate_macs(point_wise_1, input_channel, input_size)

            point_wise_2 = nn.Sequential(
                            ConvBNRelu(mid_channel,
                                       output_channel,
                                       kernel=1,
                                       stride=1,
                                       pad=0,
                                       activation=None,
                                       group=1)
                            )

            point_wise_2_macs = self._calculate_macs(point_wise_2, mid_channel, input_size if stride==1 else input_size//2)

            layers_macs["point_1"] = point_wise_1_macs
            layers_macs["point_2"] = point_wise_2_macs

            input_channel = output_channel
            input_size = input_size if stride == 1 else input_size//2
            lookup_table_operation[str(l)] = layers_macs

        return lookup_table_operation

    def _calculate_basic_operations(self, write_to_file=None):
        model_macs = 0

        if self.CONFIG.dataset[:5] == "cifar":
            last_input_size = 4
            first = nn.Sequential(ConvBNRelu(input_channel=3, 
                                             output_channel=32, 
                                             kernel=3, 
                                             stride=1,
                                             pad=3//2, 
                                             activation="relu"),
                                  MBConv(input_channel=32,
                                         output_channel=16,
                                         expansion=1,
                                         kernels=[3],
                                         stride=1,
                                         activation="relu",
                                         split_block=1,
                                         se=False))
        elif self.CONFIG.dataset[:8] == "imagenet":
            last_input_size = 7
            first = nn.Sequential(ConvBNRelu(input_channel=3, 
                                             output_channel=32, 
                                             kernel=3, 
                                             stride=2,
                                             pad=3//2, 
                                             activation="relu"),
                                  MBConv(input_channel=32,
                                         output_channel=16,
                                         expansion=1,
                                         kernels=[3],
                                         stride=1,
                                         activation="relu",
                                         split_block=1,
                                         se=False))
        model_macs += self._calculate_macs(first, 3, self.CONFIG.input_size) 

        last_stage = nn.Sequential(
                    conv_1x1_bn(320, 1280),
                    Flatten(),
                    nn.Linear(1280, self.CONFIG.classes)
                )
        model_macs += self._calculate_macs(last_stage, 320, last_input_size)
        return model_macs


    def _calculate_depthwise_operations(self, write_to_file=None):
        lookup_table_operation = {}

        input_channel = 16
        input_size = self.CONFIG.input_size \
                        if self.CONFIG.dataset[:5] == "cifar" else self.CONFIG.input_size//2
        for l, l_cfg in enumerate(self.CONFIG.l_cfgs):
            layers_macs = {}

            expansion, output_channel, kernels, stride, split_block, se = l_cfg

            for k in kernels:
                if k == "skip":
                    model = nn.Sequential()
                    model_macs = 0
                else:
                    block_input_channel = input_channel
                    model = ConvBNRelu(block_input_channel,
                                       block_input_channel,
                                       kernel=k,
                                       stride=stride,
                                       pad=(k//2),
                                       activation="relu",
                                       group=block_input_channel)
                    model_macs = self._calculate_macs(model, block_input_channel, input_size)
                layers_macs["k{}".format(k)] = model_macs

            input_channel = output_channel
            input_size = input_size if stride == 1 else input_size//2
            lookup_table_operation[str(l)] = layers_macs

        return lookup_table_operation
    
    def _calculate_macs(self, model, input_channel, input_size):
        counter = MAC_Counter(model, [1, input_channel, input_size, input_size])
        macs = counter.print_summary()["total_gmacs"]*1000
        return macs

    def calculate_block_probability(self, arch_param, tau):
        """
        Encode arch param to probability for generator training
        """
        arch_param = arch_param.view(len(self.CONFIG.l_cfgs), self.CONFIG.split_blocks*self.CONFIG.kernels_nums)
        p_arch_param = torch.zeros_like(arch_param)

        for l_num, (l_cfg, l, p_l) in enumerate(zip(self.CONFIG.l_cfgs, arch_param, p_arch_param)):
            expansion, output_channel, kernels, stride, split_block, se = l_cfg

            for b in range(expansion):
                if b == 0 and l_num in self.CONFIG.static_layers:
                    end_index = (b+1)*split_block-1
                    split_arch_param = l[b*split_block:(b+1)*split_block-1]
                else:
                    end_index = (b+1)*split_block
                    split_arch_param = l[b*split_block:(b+1)*split_block]
                p_l[b*split_block:end_index] = \
                        F.gumbel_softmax(split_arch_param, tau=tau)


        return p_arch_param
    
    def get_validation_arch_param(self, arch_param):
        """
        Encode arch param to one-hot arch param for discrete validation
        """
        arch_param = arch_param.view(len(self.CONFIG.l_cfgs), self.CONFIG.split_blocks*self.CONFIG.kernels_nums)
        val_arch_param = torch.zeros_like(arch_param)

        for l_num, (l_cfg, l, v_l) in enumerate(zip(self.CONFIG.l_cfgs, arch_param, val_arch_param)):
            expansion, output_channel, kernels, stride, split_block, se = l_cfg

            for b in range(expansion):
                if b == 0 and l_num in self.CONFIG.static_layers:
                    end_index = (b+1)*split_block-1
                    split_arch_param = l[b*split_block:end_index]
                else:
                    end_index = (b+1)*split_block
                    split_arch_param = l[b*split_block:end_index]
                v_l[b*split_block:end_index] = \
                        self._get_one_hot_vector(split_arch_param)

        return val_arch_param

    def _get_one_hot_vector(self, arch_param):
        arch_param = F.softmax(arch_param)
        one_hot_vector = torch.zeros_like(arch_param)
        one_hot_vector[arch_param.argmax()] = 1

        return one_hot_vector

    def decode_arch_param(self, arch_param):
        # Correct the index if choose skip operation
        del_index = 0
        layers_config = copy.deepcopy(self.CONFIG.l_cfgs)
        for l_num, (l_cfg, l_arch_param) in enumerate(zip(self.CONFIG.l_cfgs, arch_param)):
            expansion, output_channel, kernels, stride, split_block, se = l_cfg

            config_kernel = []
            expansion_num = 0
            skip_num = 0
            for b in range(expansion):
                if b == 0 and l_num in self.CONFIG.static_layers:
                    end_index = (b+1)*split_block-1
                    split_arch_param = l_arch_param[b*split_block:(b+1)*split_block-1]
                else:
                    end_index = (b+1)*split_block
                    split_arch_param = l_arch_param[b*split_block:(b+1)*split_block]

                choose_kernel = split_arch_param.argmax()
                if kernels[choose_kernel.item()] == "skip":
                    skip_num += 1
                    continue
                config_kernel.append(kernels[choose_kernel.item()])
                expansion_num += 1
            
            #logging.info(skip_num)
            if skip_num == expansion:
                logging.info("Delete")
                del layers_config[l_num-del_index]
                del_index += 1
                continue
            layers_config[l_num-del_index][0] = expansion_num
            layers_config[l_num-del_index][2] = config_kernel
        return layers_config

    def encode_arch_param(self, layers_config):
        """
        Encode layers_config to arch_param
        """
        arch_param = torch.zeros(len(self.CONFIG.l_cfgs), self.CONFIG.split_blocks*self.CONFIG.kernels_nums)
        for a in range(len(arch_param)):
            a_l = [0, 0, 0, 1] * 6
            arch_param[a] = torch.tensor(a_l)

        for i, l in enumerate(layers_config):
            expansion, output_channel, kernels, stride, split_block, se = l
            for e in range(expansion):
                if kernels[e] == 3:
                    kernel_encode = torch.tensor([1, 0, 0, 0])
                elif kernels[e] == 5:
                    kernel_encode = torch.tensor([0, 1, 0, 0])
                elif kernels[e] == 7:
                    kernel_encode = torch.tensor([0, 0, 1, 0])

                arch_param[i, e*4:e*4+4] = kernel_encode

        return arch_param

    def encode_skip_connection(self, arch_param):
        del_indexs = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17]
        del_index = random.smaple(del_indexs, 1)

        arch_param[del_index] = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.float)
        macs = self.get_model_macs(arch_param.cuda())

        return arch_param, macs


        


