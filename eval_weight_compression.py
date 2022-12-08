import os
import torch
import numpy as np

from models.model_presets import imagenet_pretrained
from simulation.compression_sim import WeightCompressionSim
from simulation.testbenches import testbench_filter


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"

    compr_tb = WeightCompressionSim(linesize=8)

    for name, config in imagenet_pretrained.items():
        # Generate model
        model = config.generate()
        compr_tb.register_model(model=model, model_name=name, testbench_filter=testbench_filter)

        dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

        model.eval()
        model(dummy_image)

    print(compr_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,' + ','.join(compr_tb.algo_names)]
        for key, val in compr_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))