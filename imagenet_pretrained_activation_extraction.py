import os
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.tools.imagenet_utils.args_generator import args
from models.model_presets import imagenet_pretrained
from models.tools.extractor import ModelExtractor, weight_trace, bias_trace


parser = argparse.ArgumentParser(description='Extraction Configs')
parser.add_argument('-dir', '--directory', default=os.path.join(os.curdir, 'extractions_activations'),
                    help='Directory of model extraction files', dest='extdir')
comp_args, _ = parser.parse_known_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device ({__name__})")


# Dataset configuration
dataset_dirname = args.data
if not os.path.isdir(dataset_dirname):
    dataset_dirname = os.path.join('C://', 'data', 'imagenet')

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
    save_dirpath = os.path.join(os.curdir, 'model_output')
    os.makedirs(save_dirpath, exist_ok=True)

    extractor_module = ModelExtractor(verbose=True)
    extracted_resultfiles = []

    for model_type, model_config in imagenet_pretrained.items():
        # if model_type != 'ResNet18' and model_type != 'ResNet34':
        #     continue

        full_modelname = f"{model_type}_Imagenet"
        save_modelname = f"{model_type}_Imagenet.pth"
        save_fullpath = os.path.join(save_dirpath, save_modelname)

        print("\nTest Configs:")
        print(f"- full modelname: {full_modelname}")
        print(f"- save modelname: {save_modelname}")
        print(f"- save fullpath:  {save_fullpath}\n")

        model = model_config.generate().to(device)
        torch.save(model.state_dict(), save_fullpath)

        save_extraction_dir = os.path.join(comp_args.extdir, full_modelname + '_output')
        os.makedirs(save_extraction_dir, exist_ok=True)

        print(f"extracting '{full_modelname}' at {save_extraction_dir}")

        extractor_module.target_model = model
        extractor_module.output_modelname = full_modelname
        extractor_module.reset()
        # for layer_name, layer_obj in model_config.traced_layers(extractor_module.target_model).items():
        #     extractor_module.register_hook(layer_obj, layer_name)
        extractor_module.register_hook_by_type(torch.nn.modules.Conv2d)
        extractor_module.register_hook_by_type(torch.nn.modules.ReLU)
        extractor_module.register_hook_by_type(torch.nn.modules.ReLU6)
        extractor_module.register_hook_by_type(torch.nn.modules.BatchNorm2d)
        extractor_module.extract_activation(test_loader, max_iter=3)
        extractor_module.save_activation(savepath=save_extraction_dir)

        print(f"extracting '{full_modelname}' completed\n\n")
