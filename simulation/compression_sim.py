import numpy as np
import torch.nn

from simulation.sim_metaclass import Sim
from compression.algorithms import compression_algorithms
from models.tools.lowering import ConvLayerInfo, weight_lowering, ifm_lowering


def sparsity(arr: np.ndarray):
    return (arr.size - np.count_nonzero(arr)) / arr.size


class WeightCompressionSim(Sim):
    def __init__(self, quant: bool=False, linesize: int=-1,
                 algo_names = ('ZVC', 'BDIZV', 'BPC', 'CSC')) -> None:
        super(WeightCompressionSim, self).__init__()

        self.quant = quant
        self.linesize = linesize
        self.algo_names = algo_names

        self.result = {}

    def generate_hook(self, model_name, submodule_name, submodule_info) -> callable:
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
                    compr_siz[aidx] = total_siz / len(compression_algorithms[aname](lowered_weight, wordwidth=lowered_weight.dtype.itemsize*8))
                else:
                    for weight_vec in lowered_weight:
                        if self.linesize != -1:
                            for st_idx in range(0, len(weight_vec), self.linesize):
                                ed_idx = min(st_idx + self.linesize, len(weight_vec))
                                compr_siz[aidx] += len(compression_algorithms[aname](weight_vec[st_idx:ed_idx],
                                                                              wordwidth=weight_vec.dtype.itemsize * 8))
                        else:
                            compr_siz[aidx] += len(compression_algorithms[aname](weight_vec, wordwidth=weight_vec.dtype.itemsize*8))
                    compr_siz[aidx] = total_siz / compr_siz[aidx]

            self.result[key] = compr_siz

            print(
                f"compressing weight: {model_name:15s} {submodule_name:30s} "
                f"sparsity: {sparsity(lowered_weight) * 100:3.2f}%, "
                f"result: {', '.join(map(lambda x: f'{x[0]}:{x[1]:.4f}', zip(self.algo_names, compr_siz)))}"
            )

        return weight_compr_hook


class ActivationCompressionSim(Sim):
    def __init__(self, quant: bool=False, linesize: int=-1,
                 algo_names = ('ZVC', 'BDIZV', 'BPC', 'CSC')) -> None:
        super(ActivationCompressionSim, self).__init__()

        self.quant = quant
        self.linesize = linesize
        self.algo_names = algo_names

        self.result = {}

    def generate_hook(self, model_name, submodule_name, submodule_info) -> callable:
        def activation_compr_hook(submodule, input_tensor, output_tensor):
            print(f'activation compression hook called for {model_name}.{submodule_name}')
            key = (model_name, submodule_name)
            compr_siz = [0] * len(self.algo_names)

            if self.quant:
                input_tensor = input_tensor[0].detach().int_repr()
            else:
                input_tensor = input_tensor[0]

            print('- lowering input tensor')
            lowered_input = ifm_lowering(ifm=input_tensor, layer_info=submodule_info, verbose=False).detach().cpu().numpy()
            total_siz = lowered_input.dtype.itemsize*8 * lowered_input.size

            for aidx, aname in enumerate(self.algo_names):
                print(f'- start compressing with {aname}')
                if aname == 'CSC':
                    compr_siz[aidx] = total_siz / len(compression_algorithms[aname](lowered_input, wordwidth=lowered_input.dtype.itemsize*8))
                else:
                    for activation_vec in lowered_input:
                        if self.linesize != -1:
                            for st_idx in range(0, len(activation_vec), self.linesize):
                                ed_idx = min(st_idx + self.linesize, len(activation_vec))
                                compr_siz[aidx] += len(compression_algorithms[aname](activation_vec[st_idx:ed_idx], wordwidth=activation_vec.dtype.itemsize*8))
                        else:
                            compr_siz[aidx] += len(compression_algorithms[aname](activation_vec, wordwidth=activation_vec.dtype.itemsize*8))
                    compr_siz[aidx] = total_siz / compr_siz[aidx]

            self.result[key] = compr_siz

            print(
                f"- compression result: {model_name:15s} {submodule_name:30s} "
                f"sparsity: {sparsity(lowered_input) * 100:3.2f}%, "
                f"result: {', '.join(map(lambda x: f'{x[0]}:{x[1]:.4f}', zip(self.algo_names, compr_siz)))}"
            )

        return activation_compr_hook