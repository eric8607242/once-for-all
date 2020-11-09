import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.supernet import Supernet
from utils.generator import get_generator
from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cross_encropy_with_label_smoothing, cal_model_efficient
from utils.backbone_pool import BackbonePool
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import get_optimizer, get_lr_scheduler, CrossEntropyLossSoft
from utils.lookup_table_builder import LookUpTable
from utils.trainer import Trainer

# once for all
from ofa.tutorial.accuracy_predictor import AccuracyPredictor
from ofa.tutorial.flops_table import FLOPsTable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    parser.add_argument("--title", type=str, help="experiment title", required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    get_logger(CONFIG.log_dir)
    writer = get_writer(args.title, CONFIG.write_dir)

    logging.info("=================================== Experiment title : {} Start ===========================".format(args.title))



    set_random_seed(CONFIG.seed)

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG)


    generator = get_generator(CONFIG, 21*8)

    generator.to(device)

    # ============ OFA ================
    accuracy_predictor = AccuracyPredictor(
                pretrained=True,
                device=device
            )
    print(accuracy_predictor.model)
    flops_table = FLOPsTable(device=device)

    # =================================

    g_optimizer = get_optimizer(generator, CONFIG.g_optim_state)

    start_time = time.time()
    trainer = Trainer(g_optimizer, writer, device, accuracy_predictor, flops_table, CONFIG)
    trainer.search_train_loop(generator)
    logging.info("Total search time: {:.2f}".format(time.time() - start_time))

    logging.info("=================================== Experiment title : {} End ===========================".format(args.title))
