import numpy as np

from compression.algorithms import zeroval_compression, bdizv_compression, bitplane_compression, csc_compression
from models.tools.lowering import ConvLayerInfo, weight_lowering, ifm_lowering


def sparsity(arr: np.ndarray):
    return (arr.size - np.count_nonzero(arr)) / arr.size


class CompressionTestbench(object):
    def __init__(self, quant: bool=False, linesize: int=-1) -> None:
        self.quant = quant
        self.linesize = linesize

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
            print(f'weight compression hook called for {model_name}.{submodule_name}')
            key = (model_name, submodule_name)
            compr_siz = [0] * len(self.algo_names)

            print('- lowering weight tensor')
            weight_tensor = submodule.weight if not self.quant else submodule.weight().detach().int_repr()
            lowered_weight = weight_lowering(weight=weight_tensor, layer_info=submodule_info).detach().cpu().numpy()
            total_siz = lowered_weight.dtype.itemsize*8 * lowered_weight.size

            for aidx, aname in enumerate(self.algo_names):
                print(f'- start compressing with {aname}')

                if aname == 'CSC':
                    compr_siz[aidx] = total_siz / len(self.algorithms[aname](lowered_weight, wordwidth=lowered_weight.dtype.itemsize*8))
                else:
                    for weight_vec in lowered_weight:
                        if self.linesize != -1:
                            for st_idx in range(0, len(weight_vec), self.linesize):
                                ed_idx = min(st_idx + self.linesize, len(weight_vec))
                                compr_siz[aidx] += len(self.algorithms[aname](weight_vec[st_idx:ed_idx],
                                                                              wordwidth=weight_vec.dtype.itemsize * 8))
                        else:
                            compr_siz[aidx] += len(self.algorithms[aname](weight_vec, wordwidth=weight_vec.dtype.itemsize*8))
                    compr_siz[aidx] = total_siz / compr_siz[aidx]

            self.result[key] = compr_siz

            print(
                f"compressing weight: {model_name:15s} {submodule_name:30s} "
                f"sparsity: {sparsity(lowered_weight) * 100:3.2f}%, "
                f"result: {', '.join(map(lambda x: f'{x[0]}:{x[1]:.4f}', zip(self.algo_names, compr_siz)))}"
            )

        return weight_compr_hook

    def register_weight_compression(self, model, model_name) -> None:
        layer_info_dict = ConvLayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if sm_name in layer_info_dict.keys():
                sm.register_forward_hook(
                    self.generate_weight_compr_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))

    def generate_activation_compr_hook(self, model_name, submodule_name, submodule_info) -> callable:
        def activation_compr_hook(submodule, input_tensor, output_tensor):
            print(f'activation compression hook called for {model_name}.{submodule_name}')
            key = (model_name, submodule_name)
            compr_siz = [0] * len(self.algo_names)

            if self.quant:
                input_tensor = input_tensor[0].detach().int_repr()
            else:
                input_tensor = input_tensor[0]

            print('- lowering input tensor')
            lowered_input = ifm_lowering(ifm=input_tensor, layer_info=submodule_info, verbose=True).detach().cpu().numpy()
            total_siz = lowered_input.dtype.itemsize*8 * lowered_input.size

            for aidx, aname in enumerate(self.algo_names):
                print(f'- start compressing with {aname}')
                if aname == 'CSC':
                    compr_siz[aidx] = total_siz / len(self.algorithms[aname](lowered_input, wordwidth=lowered_input.dtype.itemsize*8))
                else:
                    for activation_vec in lowered_input:
                        if self.linesize != -1:
                            for st_idx in range(0, len(activation_vec), self.linesize):
                                ed_idx = min(st_idx + self.linesize, len(activation_vec))
                                compr_siz[aidx] += len(self.algorithms[aname](activation_vec[st_idx:ed_idx], wordwidth=activation_vec.dtype.itemsize*8))
                        else:
                            compr_siz[aidx] += len(self.algorithms[aname](activation_vec, wordwidth=activation_vec.dtype.itemsize*8))
                    compr_siz[aidx] = total_siz / compr_siz[aidx]

            self.result[key] = compr_siz

            print(
                f"- compression result: {model_name:15s} {submodule_name:30s} "
                f"sparsity: {sparsity(lowered_input) * 100:3.2f}%, "
                f"result: {', '.join(map(lambda x: f'{x[0]}:{x[1]:.4f}', zip(self.algo_names, compr_siz)))}"
            )

        return activation_compr_hook

    def register_activation_compression(self, model, model_name) -> None:
        layer_info_dict = ConvLayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if sm_name in layer_info_dict.keys():
                sm.register_forward_hook(
                    self.generate_activation_compr_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))