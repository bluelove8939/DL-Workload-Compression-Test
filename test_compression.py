import os
import torch

from models.tools.imagenet_utils.args_generator import args
from models.tools.imagenet_utils.dataset_loader import val_loader
from models.tools.lowering import ifm_lowering, weight_lowering, ConvLayerInfo
from models.model_presets import imagenet_pretrained

from compression.algorithms import zeroval_compression, bdizv_compression, bitplane_compression, csc_compression


class CompressionTestbench(object):
    def __init__(self) -> None:
        self.algorithms = {
            'ZVC':   zeroval_compression,
            'BDIZV': bdizv_compression,
            'BPC':   bitplane_compression,
            'CSC':   csc_compression,
        }
        self.algo_names = ['ZVC', 'BDIZV', 'BPC', 'CSC']

        self.result = {}

    def generate_weight_compr_hook(self, model_name, submodule_name) -> callable:
        def weight_compr_hook(submodule, input_tensor, output_tensor):
            key = (model_name, submodule_name)
            res = []

            for aname in self.algo_names:
                pass

        return weight_compr_hook

    def register_model(self, model, model_name) -> None:
        for sm_name, sm in model.named_modules():
            if 'conv' in type(sm).__name__.lower():
                sm.register_forward_hook(
                    compr_tb.generate_weight_compr_hook(model_name=model_name, submodule_name=sm_name))


if __name__ == '__main__':
    # Test Configs
    line_size = 8

    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = "compression.csv"

    compr_tb = CompressionTestbench()

    for name, config in imagenet_pretrained.items():
        # Generate model
        model = config.generate()
        compr_tb.register_model(model=model, model_name=name)