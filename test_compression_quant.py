import os
import torch

from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.dataset_loader import val_loader
from models.model_presets import imagenet_pretrained


if __name__ == '__main__':
    # Test Configs
    line_size = 8

    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = "compression.csv"

    content = []

    for name, model in imagenet_pretrained.items():
        pass