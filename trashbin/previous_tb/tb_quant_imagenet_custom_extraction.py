import os
import argparse
import platform
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.tools.progressbar import progressbar
from models.tools.imagenet_utils.args_generator import args
from models.model_presets import imagenet_pretrained
from models.tools.extractor import ModelExtractor, weight_trace, bias_trace
from models.tools.quanitzation import QuantWrapper


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, '../extractions_quant'), help='Directory of model extraction files', dest='extdir')
comp_args, _ = parser.parse_known_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")


# Dataset configuration
dataset_dirname = args.data
if not os.path.isdir(dataset_dirname):
    dataset_dirname = os.path.join(os.sep, 'Users', 'user', "Desktop", "study", "data", "imagenet")

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

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler)


if __name__ == '__main__':
    save_dirpath = os.path.join(os.curdir, '../../model_output')
    os.makedirs(save_dirpath, exist_ok=True)

    extracted_resultfiles = []

    for model_type, model_config in imagenet_pretrained.items():
        full_modelname = f"{model_type}_custom_quant_Imagenet"
        save_modelname = f"{model_type}_custom_quant_Imagenet.pth"
        save_fullpath = os.path.join(save_dirpath, save_modelname)

        print("Test Configs:")
        print(f"- full modelname: {full_modelname}")
        print(f"- save modelname: {save_modelname}")
        print(f"- save fullpath:  {save_fullpath}\n")

        model = QuantWrapper(model=model_config.generate())
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)

        for i, (images, target) in enumerate(train_loader):
            if i >= 10:
                break
            output = model(images)
            print(f"\r{progressbar(i, len(train_loader), scale=50)} [{i}/{len(train_loader)}]", end='')

        quant_model = torch.quantization.convert(model_prepared)
        # lr = 0.0001
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # loss_fn = nn.CrossEntropyLoss().to(device)

        # qmodule = QuantizationModule(train_loader, loss_fn=loss_fn, optimizer=optimizer)
        # quant_model = qmodule.quantize(model, default_qconfig='fbgemm', citer=0, verbose=1)
        # torch.save(quant_model.state_dict(), save_fullpath)

        save_extraction_dir = os.path.join(comp_args.extdir, full_modelname)
        os.makedirs(save_extraction_dir, exist_ok=True)

        print(f"extracting '{full_modelname}' at {save_extraction_dir}")

        extractor_module = ModelExtractor(target_model=quant_model, output_modelname=full_modelname, verbose=False)
        extractor_module.reset()
        # extractor_module.add_param_trace(weight_trace)
        # extractor_module.add_param_trace(bias_trace)
        # extractor_module.extract_params()
        # extractor_module.save_params(savepath=save_extraction_dir, only_quant=True)


        print(f"extracting '{full_modelname}' completed")