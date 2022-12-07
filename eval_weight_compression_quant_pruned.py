import os
import torch
import numpy as np

from models.model_presets import generate_from_quant_chkpoint, imagenet_pretrained
from simulation.compression_sim import CompressionTestbench


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"
    filepath_fmt = os.path.join(os.curdir, 'model_output', "{name}_quantized_tuned_citer_10_pruned_pamt_0.5.pth")

    compr_tb = CompressionTestbench(quant=True)

    for name, config in imagenet_pretrained.items():
        if not os.path.isfile(filepath_fmt.format(name=name)):
            continue

        # Generate model
        model = generate_from_quant_chkpoint(
            model_primitive=config.generate(),
            chkpoint_path=filepath_fmt.format(name=name),)
        compr_tb.register_weight_compression(model=model, model_name=name)

        dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

        model.eval()
        model(dummy_image)

    print(compr_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,' + ','.join(compr_tb.algo_names)]
        for key, val in compr_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))