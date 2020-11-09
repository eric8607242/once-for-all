import time
import random
import logging
import argparse

import torch
import torch.nn as nn

import pandas as pd
import scipy.stats as stats

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient, min_max_normalize
from utils.model import Model
from utils.backbone_pool import BackbonePool
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import cal_hc_loss, get_optimizer, get_lr_scheduler
from utils.lookup_table_builder import LookUpTable

def evaluate_generator_top1(trainer, generator, model, test_loader, lookup_table, backbone_pool, device, CONFIG):
    avg_metric = {"gen_macs":[], "avg":[]}
    for macs in range(CONFIG.low_macs, CONFIG.high_macs, 10):
        top1_avg, gen_macs = trainer.generator_validate(generator, model, test_loader, 0, 0, hardware_constraint=macs, sample=True)

        avg_metric["gen_macs"].append(gen_macs)
        avg_metric["avg"].append(top1_avg)

    return avg_metric


def evaluate_generator(generator, backbone_pool, lookup_table, CONFIG, device, val=True):
    """
    Evaluate kendetall and hardware constraint loss of generator
    """
    total_loss = 0

    evaluate_metric = {"gen_macs":[], "true_macs":[]}
    for mac in range(CONFIG.low_macs, CONFIG.high_macs, 10):
        hardware_constraint = torch.tensor(mac, dtype=torch.float32)
        hardware_constraint = hardware_constraint.view(-1, 1)
        hardware_constraint = hardware_constraint.to(device)

        backbone = backbone_pool.get_backbone(hardware_constraint.item())
        backbone = backbone.to(device)

        normalize_hardware_constraint = min_max_normalize(CONFIG.high_macs, CONFIG.low_macs, hardware_constraint)

        noise = torch.randn(*backbone.shape)
        noise = noise.to(device)
        noise *= 0

        arch_param = generator(backbone, normalize_hardware_constraint, noise)
        arch_param = lookup_table.get_validation_arch_param(arch_param)

        layers_config = lookup_table.decode_arch_param(arch_param)
        print(layers_config)

        gen_mac = lookup_table.get_model_macs(arch_param)
        hc_loss = cal_hc_loss(gen_mac.cuda(), hardware_constraint.item(), CONFIG.alpha, CONFIG.loss_penalty)

        evaluate_metric["gen_macs"].append(gen_mac.item())
        evaluate_metric["true_macs"].append(mac)

        total_loss += hc_loss.item()
    tau, _ = stats.kendalltau(evaluate_metric["gen_macs"], evaluate_metric["true_macs"])

    return evaluate_metric, total_loss, tau

def evaluate_arch_param(evaluate_trainer, supernet, generator, train_loader, test_loader, parameter_metric, backbone_pool, lookup_table, device, CONFIG):
    avg_metric = {"gen_macs":[], "avg":[], "supernet_avg":[]}
    for a in range(parameter_metric.shape[0]):
        arch_param = torch.tensor(parameter_metric.iloc[a].values.reshape(19, -1), dtype=torch.float)
        layers_config = lookup_table.decode_arch_param(arch_param)

        model = Model(layers_config, CONFIG.dataset, CONFIG.classes)
        model = model.to(device)
        if (device.type == "cuda" and CONFIG.ngpu >= 1) :
            model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

        macs = cal_model_efficient(model, CONFIG)
        arch_param, hardware_constraint = evaluate_trainer.set_arch_param(generator, supernet, hardware_constraint=macs, arch_param=arch_param)
        supernet_avg, _ = evaluate_trainer.generator_validate(generator, supernet, test_loader, 0, 0, hardware_constraint=hardware_constraint, arch_param=arch_param, sample=False)
        
        avg_accuracy = 0
        """
        for t in range(CONFIG.train_time):
            model = Model(layers_config, CONFIG.dataset, CONFIG.classes)
            model = model.to(device)
            if (device.type == "cuda" and CONFIG.ngpu >= 1) :
                model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

            optimizer = get_optimizer(model, CONFIG.optim_state)
            scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

            trainer = Trainer(criterion, None, optimizer, None, scheduler, writer, device, lookup_table, backbone_pool, CONFIG)
            top1_avg = trainer.train_loop(train_loader, test_loader, model)

            avg_accuracy += top1_avg
        """

        avg_accuracy /= CONFIG.train_time
        avg_metric["gen_macs"].append(macs)
        avg_metric["avg"].append(avg_accuracy)
        avg_metric["supernet_avg"].append(supernet_avg)
    return avg_metric
    
def evaluate_supernet(trainer, generator, model, train_loader, test_loader, lookup_table, backbone_pool, device, CONFIG, bias=10):
    avg_metric = {"gen_macs":[], "avg":[], "skip_flag":[]}
    for target_macs in range(CONFIG.low_macs, CONFIG.high_macs, 10):
        for i in range(20):
            skip_flag = 0

            skip = False
            #if i < 5:
                #skip = True
            gen_mac, arch_param = backbone_pool.generate_arch_param(lookup_table, skip=skip)

            while gen_mac > target_macs + bias or gen_mac < target_macs - bias:
                gen_mac, arch_param = backbone_pool.generate_arch_param(lookup_table, skip=skip)

            mac = torch.tensor(gen_mac)
            mac = mac.to(device)

            layers_config = lookup_table.decode_arch_param(arch_param)
            print(layers_config)
            
            if len(layers_config) < 19:
                skip_flag = 1

            arch_param, hardware_constraint =trainer.set_arch_param(generator, model, hardware_constraint=mac, arch_param=arch_param)
            top1_avg, _ = trainer.generator_validate(generator, model, test_loader, 0, 0, hardware_constraint=hardware_constraint, arch_param=arch_param, sample=False)
            
            avg_metric["gen_macs"].append(mac.item())
            avg_metric["avg"].append(top1_avg)
            avg_metric["skip_flag"].append(skip_flag)
    return avg_metric

def evaluate_lookup_table(lookup_table, backbone_pool, CONFIG, evaluate_nums=10):
    for i in range(evaluate_nums):
        gen_mac, arch_param = backbone_pool.generate_arch_param(lookup_table)
        gen_mac = lookup_table.get_model_macs(arch_param.cuda())
        layers_config = lookup_table.decode_arch_param(arch_param)

        model = Model(layers_config, CONFIG.dataset, CONFIG.classes)

        cal_model_efficient(model, CONFIG)

def generate_architecture_parameter(target_macs=100, generate_num=10, bias=1):
    parameter_metric = []
    for i in range(generate_num):
        gen_mac, arch_param = backbone_pool.generate_arch_param(lookup_table)

        while gen_mac > target_macs + bias or gen_mac < target_macs - bias:
            gen_mac, arch_param = backbone_pool.generate_arch_param(lookup_table)

        parameter = arch_param.reshape(-1)
        parameter_metric.append(parameter.tolist())

    return parameter_metric
    
def save_generator_evaluate_metric(evaluate_metric, path_to_generator_evaluate):
    df_metric = pd.DataFrame(evaluate_metric)
    df_metric.to_csv(path_to_generator_evaluate, index=False)

if __name__ == "__main__":
    # reconstruct
    from utils.trainer import Trainer
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--evaluate-supernet", action="store_true", default=False, help="whether to evaluate supernet")
    parser.add_argument("--evaluate-generator", action="store_true", default=False, help="whether to evaluate generator")
    parser.add_argument("--evaluate-arch-param", action="store_true", default=False, help="whether to evaluate arch_param")
    parser.add_argument("--evaluate-lookup_table", action="store_true", default=False, help="whether to evaluate lookup_table")
    parser.add_argument("--loading-architectures", action="store_true", default=False, help="whether to load the architecture")
    parser.add_argument("--generate-architecture-parameter", action="store_true", default=False, help="generate architecture")
    parser.add_argument("--target-macs", type=int, help="target macs")
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    get_logger(CONFIG.log_dir)
    writer = get_writer(CONFIG.write_dir)

    #set_random_seed(CONFIG.seed)

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG)

    model = Supernet(CONFIG)
    lookup_table = LookUpTable(CONFIG)

    arch_param_nums = model.get_arch_param_nums()
    #generator = ConvGenerator(CONFIG.hc_dim, 1, CONFIG.hidden_dim)
    generator = get_generator(CONFIG, arch_param_nums)

    criterion = cross_encropy_with_label_smoothing

    if CONFIG.generator_pretrained is not None:
        logging.info("Loading model")
        model.load_state_dict(torch.load(CONFIG.model_pretrained)["model"])
        generator.load_state_dict(torch.load(CONFIG.generator_pretrained)["model"])

    generator.to(device)
    model.to(device)
    if (device.type == "cuda" and CONFIG.ngpu >= 1):
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))
        
    backbone_pool = BackbonePool(lookup_table, arch_param_nums, None, None, None, None, CONFIG)

    trainer = Trainer(criterion, None, None, None, None, writer, device, lookup_table, backbone_pool, CONFIG)


    architecture_list = None
    if args.loading_architectures:
        parameter_metric = pd.read_csv(CONFIG.path_to_save_architecture)

    if args.evaluate_supernet:
        avg_metric = evaluate_supernet(trainer, generator, model, train_loader, val_loader, lookup_table, backbone_pool, device, CONFIG)
        save_generator_evaluate_metric(avg_metric, CONFIG.path_to_supernet_eval)
    
    if args.evaluate_generator:
        evaluate_metric, total_loss, tau = evaluate_generator(generator, backbone_pool, lookup_table, CONFIG, device)
        avg_metric = evaluate_generator_top1(trainer, generator, model, val_loader, lookup_table, backbone_pool, device, CONFIG)

        save_generator_evaluate_metric(evaluate_metric, CONFIG.path_to_generator_eval)
        save_generator_evaluate_metric(avg_metric, CONFIG.path_to_generator_eval_avg)
            
    if args.evaluate_arch_param:
        avg_metric = evaluate_arch_param(trainer, model, generator, train_loader, val_loader, parameter_metric, backbone_pool, lookup_table, device, CONFIG)
        save_generator_evaluate_metric(avg_metric, CONFIG.path_to_save_evaluate_architecture)

    if args.evaluate_lookup_table:
        evaluate_lookup_table(lookup_table, backbone_pool, CONFIG)

    if args.generate_architecture_parameter:
        parameter_metric = generate_architecture_parameter()
        save_generator_evaluate_metric(parameter_metric, CONFIG.path_to_save_architecture)
    


        

