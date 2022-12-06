import os
import torch
import numpy as np

# from models.tools.imagenet_utils.args_generator import args
# from models.tools.imagenet_utils.dataset_loader import val_loader
from models.tools.lowering import ifm_lowering, weight_lowering, ConvLayerInfo
from models.model_presets import imagenet_quant_pretrained

from compression.algorithms import zeroval_compression, bdizv_compression, bitplane_compression, csc_compression


def sparsity(arr: np.ndarray):
    return (arr.size - np.count_nonzero(arr)) / arr.size


class CompressionQuantTestbench(object):
    def __init__(self) -> None:
        self.algorithms = {
            'ZVC':   zeroval_compression,
            'BDIZV': bdizv_compression,
            'BPC':   bitplane_compression,
            'CSC':   csc_compression,
        }
        self.algo_names = ['ZVC', 'BDIZV', 'BPC', 'CSC']

        self.result = {}

    def generate_weight_compr_hook(self, model_name, submodule_name, submodule_info) -> callable:
        def weight_compr_hook(submodule, input_tensor, output_tensor):
            # print(f"compressing weight: {model_name:15s} {submodule_name:30s} {submodule_info}")

            key = (model_name, submodule_name)
            compr_siz = [0] * len(self.algo_names)

            lowered_weight = weight_lowering(weight=submodule.weight().int_repr().detach(), layer_info=submodule_info).detach().cpu().numpy()
            total_siz = lowered_weight.dtype.itemsize*8 * lowered_weight.size

            print(f"compressing weight: {model_name:15s} {submodule_name:30s} {sparsity(lowered_weight)}")

            for aidx, aname in enumerate(self.algo_names):
                if aname == 'CSC':
                    compr_siz[aidx] = total_siz / len(self.algorithms[aname](lowered_weight, wordwidth=lowered_weight.dtype.itemsize*8))
                else:
                    for weight_vec in lowered_weight:
                        compr_siz[aidx] += len(self.algorithms[aname](weight_vec, wordwidth=weight_vec.dtype.itemsize*8))
                    compr_siz[aidx] = total_siz / compr_siz[aidx]

            self.result[key] = compr_siz

        return weight_compr_hook

    def register_model(self, model, model_name) -> None:
        layer_info_dict = ConvLayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if sm_name in layer_info_dict.keys():
                sm.register_forward_hook(
                    self.generate_weight_compr_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"

    compr_tb = CompressionQuantTestbench()

    for name, config in imagenet_quant_pretrained.items():
        # Generate model
        model = config.generate()
        compr_tb.register_model(model=model, model_name=name)

        dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

        model.eval()
        model(dummy_image)

    print(compr_tb.result)

    with open(os.path.join(log_dirname, log_filename), 'wt') as log_file:
        content = ['model name,layer name,' + ','.join(compr_tb.algo_names)]
        for key, val in compr_tb.result.items():
            content.append(f"{','.join(key)},{','.join(list(map(lambda x:f'{x:.4f}', val)))}")
        log_file.write('\n'.join(content))