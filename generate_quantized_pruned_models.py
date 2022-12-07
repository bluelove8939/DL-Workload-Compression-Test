import os
import torch
import argparse

from models.model_presets import imagenet_pretrained, generate_from_chkpoint
from models.tools.quanitzation import QuantizationModule

from models.tools.imagenet_utils.training import validate
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.dataset_loader import train_loader, val_loader


if __name__ == "__main__":
    # Pruning
    pamt = 0.5  # target pruning amount

    # Setups
    dirname = os.path.join(os.curdir, 'model_output')
    pth_filename_fmt = "{name}_quantized_tuned_citer_{citer}_pruned_pamt_{pamt:.1f}.pth"
    txt_filename_fmt = "{name}_quantized_tuned_citer_{citer}_pruned_pamt_{pamt:.1f}.txt"

    os.makedirs(dirname, exist_ok=True)

    for name, config in imagenet_pretrained.items():
        if name != 'AlexNet':
            continue

        print(f"Quantizing model: {name}...")

        # Generate model without quantization
        model = generate_from_chkpoint(
            model_primitive=config.generate(),
            chkpoint_path=os.path.join(os.curdir, 'model_output', f'{name}_pruned_tuned_pamt_{pamt}.pth'),
        )

        # Quantization setup
        citer = 10
        learning_rate = 0.0001
        crit_name = "CrossEntropyLoss"

        tuning_dataloader = train_loader
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Quantize model
        qmod = QuantizationModule(tuning_dataloader=tuning_dataloader, criterion=criterion, optimizer=optimizer)
        qmodel = qmod.quantize(model=model, citer=citer, verbose=1)  # calibration

        # Accuracy check
        r_top1_acc, r_top5_acc, _ = validate(
            val_loader=val_loader, model=model, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='normal', ret_top5=True)
        q_top1_acc, q_top5_acc, _ = validate(
            val_loader=val_loader, model=qmodel, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='quant ', ret_top5=True)

        # Save state dictionary
        os.makedirs(dirname, exist_ok=True)
        torch.save(qmodel.state_dict(), os.path.join(dirname, pth_filename_fmt.format(name=name, citer=citer, pamt=pamt)))

        # Save information
        with open(os.path.join(os.curdir, 'utils', 'quant_res_template.txt'), 'rt') as fmtfile:
            fmt = fmtfile.read()

            content = fmt.format(
                dirname=dirname, filename=txt_filename_fmt.format(name=name, citer=citer, pamt=pamt),
                r_top1_acc=r_top1_acc, r_top5_acc=r_top5_acc,
                q_top1_acc=q_top1_acc, q_top5_acc=q_top5_acc,
                citer=citer,
                criterion=type(criterion).__name__, optimizer=type(optimizer).__name__,
                optim_parameters=f'learning_rate={learning_rate}',
            )

            with open(os.path.join(dirname, txt_filename_fmt.format(name=name, citer=citer, pamt=pamt)), 'wt') as outfile:
                outfile.write(content)