import os
import torch
import numpy as np

from models.tools.imagenet_utils.dataset_loader import val_loader
from models.model_presets import imagenet_pretrained, generate_from_chkpoint
from simulation.compression_sim import CompressionTestbench


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"
    filepath_fmt = os.path.join(os.curdir, 'model_output', "{name}_pruned_tuned_pamt_0.5.pth")

    compr_tb = CompressionTestbench()

    for name, config in imagenet_pretrained.items():
        if not os.path.isfile(filepath_fmt.format(name=name)):
            continue

        # Generate model
        model = generate_from_chkpoint(
            model_primitive=config.generate(),
            chkpoint_path=filepath_fmt.format(name=name), )
        compr_tb.register_activation_compression(model=model, model_name=name)

        images, target = next(iter(val_loader))

        model.eval()
        model(images)

    print(compr_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,' + ','.join(compr_tb.algo_names)]
        for key, val in compr_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))