import os
import torch
import argparse

from models.model_presets import imagenet_pretrained
from models.tools.quanitzation import QuantizationModule

from models.tools.imagenet_utils.training import validate
from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.dataset_loader import train_loader, val_loader


parser = argparse.ArgumentParser(description='Quantized Model Generation Configs')
parser.add_argument('--normal-validate', default=False, action='store_true',
                    help='Calculate accuracy of unquantized models', dest='normal_validate')
quant_args, _ = parser.parse_known_args()


if __name__ == "__main__":
    # Setups
    dirname = os.path.join('/home', 'shared', 'Quantized_Models')
    pth_filename_fmt = "{name}_quantized_tuned_citer_{citer}.pth"
    txt_filename_fmt = "{name}_quantized_tuned_citer_{citer}.txt"

    tuning_dataloader = train_loader
    
    for name, config in imagenet_pretrained.items():
        if name in ('AlexNet', 'VGG16'):
            continue

        print(f"Quantizing model: {name}...")

        # Generate model without quantization
        model = config.generate()

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
        if quant_args.normal_validate:
            r_top1_acc, r_top5_acc, _ = validate(
                val_loader=val_loader, model=model, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='normal', ret_top5=True)
        else:
            r_top1_acc, r_top5_acc = 0, 0
        q_top1_acc, q_top5_acc, _ = validate(
            val_loader=val_loader, model=qmodel, criterion=criterion, args=args, device='cpu', at_prune=False, pbar_header='quant ', ret_top5=True)

        # Save state dictionary
        os.makedirs(dirname, exist_ok=True)
        torch.save(qmodel.state_dict(), os.path.join(dirname, pth_filename_fmt.format(name, citer)))

        # Save informations
        with open(os.path.join(os.curdir, 'utils', 'quant_res_template.txt'), 'rt') as fmtfile:
            fmt = fmtfile.read()

            content = fmt.format(
                dirname=dirname, filename=txt_filename_fmt.format(name=name, citer=citer),
                r_top1_acc=r_top1_acc, r_top5_acc=r_top5_acc,
                q_top1_acc=q_top1_acc, q_top5_acc=q_top5_acc,
                citer=citer,
                criterion=type(criterion).__name__,
                optimizer=type(optimizer).__name__,
                optim_parameters=f'learning_rate={learning_rate}',
            )

            with open(os.path.join(dirname, txt_filename_fmt.format(name=name, citer=citer)), 'wt') as outfile:
                outfile.write(content)