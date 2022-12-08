import os
import torch
from torch.utils.data import DataLoader

from models.tools.imagenet_utils.dataset_loader import val_dataset, val_sampler
from models.model_presets import imagenet_pretrained
from simulation.compression_sim import CompressionTestbench


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"

    compr_tb = CompressionTestbench(linesize=8)

    for name, config in imagenet_pretrained.items():
        # Generate model
        model = config.generate()
        compr_tb.register_activation_compression(model=model, model_name=name)

        # Inference one batch
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, sampler=val_sampler)
        images, _ = next(iter(val_loader))

        model = model.to(device)
        images = images.to(device)

        model.eval()
        model(images)

    print(compr_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,' + ','.join(compr_tb.algo_names)]
        for key, val in compr_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))