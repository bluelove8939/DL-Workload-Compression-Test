import os
import copy

import torch
from torch.utils.data import DataLoader

from models.model_presets import imagenet_pretrained
from models.tools.pruning import grouped_prune_model, remove_prune_model

from models.tools.imagenet_utils.training import validate, train
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.dataset_loader import val_loader, train_dataset, train_sampler


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # Setups
    dirname = os.path.join(os.curdir, 'model_output')
    pth_filename_fmt = "{name}_pruned_tuned_pamt_{pamt}.pth"
    txt_filename_fmt = "{name}_pruned_tuned_pamt_{pamt}.txt"

    os.makedirs(dirname, exist_ok=True)
    
    for name, config in imagenet_pretrained.items():
        print(f"Pruning model: {name}...")

        # Generate model without quantization
        model = config.generate().to(device)
        # normal_statedict = copy.deepcopy(model.state_dict())

        # Pruning setup
        pamt = 0.5
        lr = 0.0001
        momentum = 0
        epoch = 1
        batch_size = 32

        tuning_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # Pruning model
        pmodel = copy.deepcopy(model).to(device)
        grouped_prune_model(model=pmodel, step=pamt)

        # Fine-tuning
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(pmodel.parameters(), lr=lr, momentum=momentum)

        train(
            train_loader=tuning_loader, model=pmodel, optimizer=optimizer, criterion=criterion, device=device,
            epoch=epoch, args=args, pbar_header='tuning')
        remove_prune_model(pmodel)

        # Accuracy check
        r_top1_acc, r_top5_acc, _ = validate(
            val_loader=val_loader, model=model, criterion=criterion, args=args, device=device, at_prune=False,
            pbar_header='normal', ret_top5=True)
        p_top1_acc, p_top5_acc, _ = validate(
            val_loader=val_loader, model=pmodel, criterion=criterion, args=args, device=device, at_prune=False,
            pbar_header='pruned', ret_top5=True)

        # Save state dictionary
        os.makedirs(dirname, exist_ok=True)
        torch.save(pmodel.state_dict(), os.path.join(dirname, pth_filename_fmt.format(name=name, pamt=pamt)))

        # Save information
        with open(os.path.join(os.curdir, 'utils', 'prune_res_template.txt'), 'rt') as fmtfile:
            fmt = fmtfile.read()

            content = fmt.format(
                dirname=dirname, filename=txt_filename_fmt.format(name=name, pamt=pamt),
                r_top1_acc=r_top1_acc, r_top5_acc=r_top5_acc,
                p_top1_acc=p_top1_acc, p_top5_acc=p_top5_acc,
                pamt=pamt, epoch=epoch,
                criterion=type(criterion).__name__, optimizer=type(optimizer).__name__,
                optim_parameters=f'lr={lr}, momentum={momentum:.1f}',
            )

            with open(os.path.join(dirname, txt_filename_fmt.format(name=name, pamt=pamt)), 'wt') as outfile:
                outfile.write(content)