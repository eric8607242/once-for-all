import time
import copy
import json
import logging
import random 
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from evaluate import evaluate_generator, save_generator_evaluate_metric
from utils.util import AverageMeter, save, accuracy, min_max_normalize, bn_calibration
from utils.optim import cal_hc_loss
from utils.countmacs import MAC_Counter


class Trainer:
    def __init__(self, g_optimizer, writer, device, accuracy_predictor, flops_table, CONFIG):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.losses = AverageMeter()
        self.hc_losses = AverageMeter()

        self.writer = writer
        self.device = device

        self.criterion = criterion
        self.g_optimizer = g_optimizer

        self.CONFIG = CONFIG

        self.epochs = self.CONFIG.epochs
        self.warmup_epochs = self.CONFIG.warmup_epochs
        self.search_epochs = self.CONFIG.search_epochs

        self.hardware_pool = [i for i in range(self.CONFIG.low_macs, self.CONFIG.high_macs, 5)]
        self.hardware_index = 0
        random.shuffle(self.hardware_pool)

        self.noise_weight = self.CONFIG.noise_weight

        # ================== OFA ====================
        self.accuracy_predictor = accuracy_predictor
        self.flops_table = flops_table

        self.backbone = self.calculate_one_hot(torch.randn(8*21)).cuda()


    def search_train_loop(self, generator):
        self.epochs = self.warmup_epochs + self.search_epochs
        # Training generator
        best_loss = 10000.0
        best_top1 = 0
        tau = 5
        for epoch in range(self.warmup_epochs, self.search_epochs):
            logging.info("Start to train for search epoch {}".format(epoch))
            logging.info("Tau: {}".format(tau))
            self._generator_training_step(generator, val_loader, epoch, tau, info_for_logger="_gen_train_step")

            # ================ Train ============================================
            for i in range():
                # Training generator
                arch_param, hardware_constraint = self.set_arch_param(generator, tau=tau)

                # ============== evaluation flops ===============================
                gen_flops = self.flops_table.predict_arch_param_efficiency(arch_param)
                hc_loss = cal_hc_loss(gen_flops.cuda(), hardware_constraint.item(), self.CONFIG.alpha, self.CONFIG.loss_penalty)
                # ===============================================================
                self.g_optimizer.zero_grad()

                # ============== predict top1 accuracy ==========================
                top1_avg = self.accuracy_predictor(arch_param)
                ce_loss = -1 * top1_avg
                # ===============================================================
                loss = ce_loss + hc_loss
                logging.info("HC loss : {}".format(hc_loss))
                loss.backward()

                self.g_optimizer.step()
                self.g_optimizer.zero_grad()
            # ====================================================================


            # ============== Valid ===============================================
            hardware_constraint, arch_param = self._get_arch_param(generator, hardware_constraint, valid=True)
            arch_param = self.calculate_one_hot(arch_param)
            arch_param, hardware_constraint = self.set_arch_param(generator, model, hardware_constraint=hardware_constraint, arch_param=arch_param)
            # ============== evaluation flops ===============================
            gen_flops = self.flops_table.predict_arch_param_efficiency(arch_param)

            hc_loss = cal_hc_loss(gen_flops.cuda(), hardware_constraint.item(), self.CONFIG.alpha, self.CONFIG.loss_penalty)
            # ===============================================================

            # ============== predict top1 accuracy ==========================
            top1_avg = self.accuracy_predictor(arch_param)
            logger.info("Valid : Top-1 avg : {}".format(top1_avg))
            # ===============================================================

            # ====================================================================
            

            # ============== Evaluate ============================================
            total_loss = 0
            evaluate_metric = {"gen_flops":[], "true_flops":[]}
            for flops in range(self.CONFIG.low_macs, self.CONFIG.high_macs, 10):
                hardware_constraint = torch.tensor(flops, dtpye=torch.float32)
                hardware_constraint = hardware_constraint.view(-1, 1)
                hardware_constraint = hardware_constraint.to(self.device)

                normalize_hardware_constraint = min_max_normalize(self.CONFIG.high_macs, self.CONFIG.low_macs, hardware_constraint) 

                noise = torch.randn(*self.backbone.shape)
                noise = noise.to(device)
                noise *= 0

                arch_param = generator(self.backbone, normalize_hardware_constraint, noise)
                # ============== evaluation flops ===============================
                gen_flops = self.flops_table.predict_arch_param_efficiency(arch_param)
                hc_loss = cal_hc_loss(gen_flops.cuda(), hardware_constraint.item(), self.CONFIG.alpha, self.CONFIG.loss_penalty)
                # ===============================================================

                evaluate_metric["gen_flops"].append(gen_flops)
                evaluate_metric["true_flops"].append(flops)

                total_loss += hc_loss.item()
            kendall_tau, _ = stats.kendalltau(evaluate_metric["gen_flops"], evaluate_metric["true_flops"])
            # ====================================================================

            logging.info("Total loss : {}".format(total_loss))
            if best_loss > total_loss:
                logging.info("Best loss by now: {} Tau : {}.Save model".format(total_loss, kendall_tau))
                best_loss = total_loss
                save_generator_evaluate_metric(evaluate_metric, self.CONFIG.path_to_generator_eval)
                save(generator, self.g_optimizer, self.CONFIG.path_to_save_generator)
            if top1_avg > best_top1 and total_loss < 0.4:
                logging.info("Best top1-avg by now: {}.Save model".format(top1_avg))
                best_top1 = top1_avg
                save(generator, self.g_optimizer, self.CONFIG.path_to_best_avg_generator)
            save(generator, self.g_optimizer, "./logs/generator/{}.pth".format(total_loss))

            tau *= self.CONFIG.tau_decay
            self.noise_weight = self.noise_weight * self.CONFIG.noise_decay if self.noise_weight > 0.0001 else 0
            logging.info("Noise weight : {}".format(self.noise_weight))
        logging.info("Best loss: {}".format(best_loss))
        save(generator, self.g_optimizer, self.CONFIG.path_to_fianl_generator)


    def _get_arch_param(self, generator, hardware_constraint=None, valid=False):
        # ====================== Strict fair sample
        if hardware_constraint is None:
            hardware_constraint = torch.tensor(self.hardware_pool[self.hardware_index]+random.random()-0.5, dtype=torch.float32).view(-1, 1)
            #hardware_constraint = torch.tensor(self.hardware_pool[self.hardware_index], dtype=torch.float32).view(-1, 1)
            self.hardware_index += 1
            if self.hardware_index == len(self.hardware_pool):
                self.hardware_index = 0
                random.shuffle(self.hardware_pool)
        else:
            hardware_constraint = torch.tensor(hardware_constraint, dtype=torch.float32).view(-1, 1)
        # ======================

        hardware_constraint = hardware_constraint.to(self.device)
        logging.info("Target macs : {}".format(hardware_constraint.item()))


        normalize_hardware_constraint = min_max_normalize(self.CONFIG.high_macs, self.CONFIG.low_macs, hardware_constraint)

        noise = torch.randn(*self.backbone.shape)
        noise = noise.to(self.device)
        noise *= self.noise_weight

        arch_param = generator(self.backbone, normalize_hardware_constraint, noise)

        return hardware_constraint, arch_param

    def set_arch_param(self, generator, hardware_constraint=None, arch_param=None, tau=None):
        """Sample the sub-network from supernet by arch_param(generate from generator or user specific)
        """
        if tau is not None:
            hardware_constraint, arch_param = self._get_arch_param(generator)
            arch_param = self.calculate_block_probability(arch_param, tau)

        arch_param = arch_param.to(self.device)
        return arch_param, hardware_constraint

    def calculate_block_probability(self, arch_param, tau):
        arch_param = arch_param.view(21, 8)
        p_arch_param = torch.zeros_like(arch_param)

        for i in range(20):
            p_arch_param[i, :4] = F.gumbel_softmax(arch_param[i, :4], tau=tau)
            p_arch_param[i, 4:] = F.gumbel_softmax(arch_param[i, 4:], tau=tau)

        p_arch_param[-1, :5] = F.gumbel_softmax(arch_param[-1, :5], tau=tau)
        return p_arch_param

    def calculate_one_hot(self, arch_param):
        arch_param = arch_param.view(21, 8)
        p_arch_param = torch.zeros_like(arch_param)

        for i in range(20):
            p_arch_param[i, arch_param[i, :4].argmax()] = 1
            p_arch_param[i, arch_param[i, 4:].argmax()] = 1

        p_arch_param[-1, arch_param[-1, :5].argmax()] = 1
        return p_arch_param
    
