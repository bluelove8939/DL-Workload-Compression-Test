import os
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.model_presets import imagenet_quant_pretrained, imagenet_pretrained
from models.tools.pruning import prune_layer, remove_prune_model
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.training import train, validate
from simulation.cycle_sim import AcceleratorCycleSim, AcceleratorConfig
from simulation.testbenches import testbenches, quantized_testbenches, layer_filter, quant_layer_filter


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, 'extractions_quant_activations'),
                    help='Directory of model extraction files', dest='extdir')
comp_args, _ = parser.parse_known_args()

args.batch_size = 1  # batch size is 1 (activation for only one image)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device ({__name__})")


# Dataset configuration
dataset_dirname = args.data

dset_dir_candidate = [
    args.data,
    os.path.join('C://', 'torch_data', 'imagenet'),
    os.path.join('E://', 'torch_data', 'imagenet'),
]

for path in dset_dir_candidate:
    if not os.path.isdir(dataset_dirname):
        dataset_dirname = path

train_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

test_dataset = datasets.ImageFolder(
        os.path.join(dataset_dirname, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=True)
else:
    train_sampler = None
    val_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler)


if __name__ == '__main__':
    save_dirpath = os.path.join(os.curdir, 'model_output')
    os.makedirs(save_dirpath, exist_ok=True)

    log_dirpath = os.path.join(os.curdir, 'logs')
    os.makedirs(log_dirpath, exist_ok=True)

    performance_log_filename = 'accelerator_cycles_quant.csv'
    performance_logs = ['model name,layer name,cycles,total']

    # Reconfig the environement if using quantized model
    quant = True
    device = 'cpu'

    for model_type, model_config in imagenet_quant_pretrained.items():
        print("\nAccelerator Simulation Configs:")
        print(f"- full modelname: {model_type}\n")

        model = model_config.generate().to(device)

        config = AcceleratorConfig(ve_num=128, vector_size=64, fifo_capacity=128,
                                   fetch_cycle=1, index_cycle=1, mac_cycle=1)
        sim = AcceleratorCycleSim(config=config, quant=quant, device=device)
        sim.register_model(model, model_name=model_type, layer_filter=layer_filter if not quant else quant_layer_filter)

        model.eval()
        for img, tag in val_loader:
            img, tag = img.to(device), tag.to(device)
            model(img)
            break

        for (model_name, layer_name), (cycles, total) in sim.get_performance().items():
            performance_logs.append(f"{model_type},{layer_name},{cycles},{total}")

    with open(os.path.join(log_dirpath, performance_log_filename), 'wt') as performance_file:
        performance_file.write('\n'.join(performance_logs))
