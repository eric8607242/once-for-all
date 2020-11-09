import os

import logging
import json

import numpy as np
import torch

from utils.util import min_max_normalize

class BackbonePool:
    def __init__(self, lookup_table, arch_param_nums, generator, model, trainer, loader, CONFIG, bias=1):
        self.arch_param_nums = arch_param_nums 
        self.CONFIG = CONFIG

        logging.info("============================= Backbone pool ================================")
        if os.path.exists(self.CONFIG.path_to_backbone_pool):
            logging.info("Load backbone pool from {}".format(self.CONFIG.path_to_backbone_pool))
            self.backbone_pool, self.backbone_pool_avg = self._load_backbone_pool()
        else:
            logging.info("Generate backbone pool")
            self.backbone_pool, self.backbone_pool_avg = self._generate_backbone_pool(generator, model, trainer, loader, lookup_table, bias)

    def get_backbone(self, macs):
        backbone_keys = np.array([int(k) for k in self.backbone_pool.keys()])
        backbone_diff = np.absolute(backbone_keys - macs)

        backbone_index = backbone_diff.argmin()
        backbone = self.backbone_pool[str(backbone_keys[backbone_index])]

        return torch.Tensor(backbone)

    def get_backbone_keys(self):
        return self.backbone_pool.keys()

    def _load_backbone_pool(self):
        backbone_pool = None
        backbone_pool_avg = None

        with open(self.CONFIG.path_to_backbone_pool) as f:
            backbone_pool = json.load(f)
        with open(self.CONFIG.path_to_backbone_pool_avg) as f:
            backbone_pool_avg = json.load(f)
        return backbone_pool, backbone_pool_avg

    def save_backbone_pool(self, path_to_backbone_pool, backbone_pool=None, avg=False):
        if backbone_pool is None:
            backbone_pool = self.backbone_pool
        if avg:
            backbone_pool = self.backbone_pool_avg

        with open(path_to_backbone_pool, "w") as f:
            json.dump(backbone_pool, f)

    def _generate_backbone_pool(self, generator, model, trainer, loader, lookup_table, bias=1):
        backbone_pool = {}
        backbone_pool_avg = {}

        low_macs = self.CONFIG.low_macs
        high_macs = self.CONFIG.high_macs
        pool_interval = (high_macs - low_macs)//(self.CONFIG.pool_size+1)

        for mac in range(low_macs+pool_interval, high_macs-1, pool_interval):
            top1_avg = 0
            gen_mac, arch_param = self.generate_arch_param(lookup_table)
            
            layers_config = lookup_table.decode_arch_param(arch_param)
            arch_param = lookup_table.encode_arch_param(layers_config)
            
            # Skip encoding =======================
            #arch_param, gen_mac = lookup_table.encode_skip_connection(arch_param)
            # =====================================
            arch_param, hardware_constraint = trainer.set_arch_param(generator, model, hardware_constraint=gen_mac, arch_param=arch_param)
            if gen_mac < mac + bias and gen_mac > mac - bias:
                top1_avg, _ = trainer.generator_validate(generator, model, loader, 0, 0, hardware_constraint=gen_mac, arch_param=arch_param, sample=False)
            while gen_mac > mac + bias or gen_mac < mac - bias or len(layers_config) < 19:
                top1_avg = 0
                gen_mac, arch_param = self.generate_arch_param(lookup_table)
                
                layers_config = lookup_table.decode_arch_param(arch_param)
                arch_param = lookup_table.encode_arch_param(layers_config)
                
                # Skip encoding ====================
                #arch_param, gen_mac = lookup_table.encode_skip_connection(arch_param)
                # ==================================
                arch_param, hardware_constraint = trainer.set_arch_param(generator, model, hardware_constraint=mac, arch_param=arch_param)
                if gen_mac < mac + bias and gen_mac > mac - bias:
                    top1_avg, _ = trainer.generator_validate(generator, model, loader, 0, 0, hardware_constraint=gen_mac, arch_param=arch_param, sample=False)
            print(layers_config)

            backbone_pool_avg[str(mac)] = top1_avg
            # =============
            backbone_pool[str(mac)] = arch_param.tolist()
            logging.info("Target mac {} : Backbone generate {}".format(mac, gen_mac))

        self.save_backbone_pool(self.CONFIG.path_to_backbone_pool, backbone_pool=backbone_pool)
        self.save_backbone_pool(self.CONFIG.path_to_backbone_pool_avg, backbone_pool=backbone_pool_avg)

        return backbone_pool, backbone_pool_avg

    def update_backbone_pool(self, new_backbone, macs, top1_avg):
        backbone_keys = np.array([int(k) for k in self.backbone_pool.keys()])
        backbone_diff = np.absolute(backbone_keys - macs)

        backbone_index = backbone_diff.argmin()
        backbone_key = backbone_keys[backbone_index]

        # check new_backbone in backbone_pool keys
        if np.absolute(backbone_key - macs) < 1 and self.backbone_pool_avg[str(backbone_key)] < top1_avg:
            logging.info("--------------------Backbone pool update. Macs : {}, Top1-avg : {}".format(macs, top1_avg))
            self.backbone_pool_avg[str(backbone_key)] = top1_avg
            self.backbone_pool[str(backbone_key)] = new_backbone.detach().cpu().tolist()

    def generate_arch_param(self, lookup_table, p=False, skip=False):
        layers_num = len(self.CONFIG.l_cfgs)
        arch_param = torch.empty(layers_num, self.arch_param_nums//len(self.CONFIG.l_cfgs))
        if skip:
            layers_expansion = np.random.randint(low=0, high=self.CONFIG.expansion+1, size=(layers_num))
            while 1 in layers_expansion:
                layers_expansion = np.random.randint(low=0, high=self.CONFIG.expansion+1, size=(layers_num))
        else:
            layers_expansion = np.random.randint(low=2, high=self.CONFIG.expansion+1, size=(layers_num))

        #else:
        #    layers_expansion = np.random.randint(low=1, high=self.CONFIG.expansion+1, size=(layers_num))

        for i in range(len(arch_param)):
            architecture = [0 for i in range(self.CONFIG.kernels_nums-1)]+[1]
            arch_param[i] = torch.tensor(architecture*self.CONFIG.split_blocks)
            for e in range(layers_expansion[i]):
                expansion_param = [0 for i in range(self.CONFIG.kernels_nums)]
                expansion_param[np.random.randint(0, self.CONFIG.kernels_nums-1)] = 1
                arch_param[i][e*self.CONFIG.kernels_nums:(e+1)*self.CONFIG.kernels_nums] = \
                        torch.tensor(expansion_param)

        arch_param = lookup_table.get_validation_arch_param(arch_param) \
            if not p else lookup_table.calculate_block_probability(arch_param, tau=5)

        mac = lookup_table.get_model_macs(arch_param.cuda())
        return mac, arch_param
