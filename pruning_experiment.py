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

    for model_type, model_config in imagenet_pretrained.items():
        # if model_type != 'ResNet18' and model_type != 'ResNet34':
        #     continue

        full_modelname = f"{model_type}_Imagenet"
        save_modelname = f"{model_type}_Imagenet.pth"
        save_fullpath = os.path.join(save_dirpath, save_modelname)

        step = 0.1
        target_amount = 0.7
        threshold = 1
        iter_max = 10

        print("\nTest Configs:")
        print(f"- step: {step}")
        print(f"- target amount: {target_amount}")
        print(f"- threshold: {threshold}")
        print(f"- max iter: {iter_max}")
        print(f"- full modelname: {full_modelname}")
        print(f"- save modelname: {save_modelname}")
        print(f"- save fullpath:  {save_fullpath}\n")

        model = model_config.generate().to(device)
        torch.save(model.state_dict(), save_fullpath)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), 0.0005,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # normal_acc, _ = validate(val_loader=val_loader, model=model, criterion=criterion, args=args,
        #                          pbar_header='normal')

        iter_cnt = 0
        current_amount = 0

        # while current_amount < target_amount:
        #     current_amount += (1 - current_amount) * step
        #
        #     prune_layer(model, step=0.1)
        #
        #     while True:
        #         pruned_acc, _ = validate(val_loader=val_loader, model=model, criterion=criterion, args=args,
        #                                  pbar_header=f'tune{iter_cnt:2d} acc')
        #
        #         if (normal_acc - pruned_acc) < threshold:
        #             print(f'pruning succeed at iter {iter_cnt}')
        #             break
        #         if iter_cnt >= iter_max:
        #             print(f'pruning failed at iter {iter_cnt}')
        #             break
        #
        #         train(train_loader, model, criterion, optimizer, epoch=1, args=args, at_prune=False,
        #               pbar_header='prune tuning')
        #
        #         iter_cnt += 1

        prune_layer(model, step=0.3)
        # pruned_acc, _ = validate(val_loader=val_loader, model=model, criterion=criterion, args=args,
        #                                                           pbar_header=f'tune{iter_cnt:2d} acc')
        #
        # print(f"model name: {full_modelname:20s} normal: {normal_acc:.2f} pruned: {pruned_acc:.2f}")

        remove_prune_model(model)

        full_pmodelname = full_modelname + '_pruned_30'
        save_pmodelname = f"{full_pmodelname}.pth"
        save_pfullpath = os.path.join(save_dirpath, save_pmodelname)

        torch.save(model.state_dict(), save_pfullpath)
