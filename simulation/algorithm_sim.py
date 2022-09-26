from typing import Callable
import torch
import numpy as np
from scipy.sparse import csr_matrix

from compression.binary_array import array2binary
from compression.algorithms import bdizv_compression

from models.tools.lowering import weight_lowering, ifm_lowering, ConvLayerInfo


# supported_algorithms : Dict[str, Callable] = {
#     'BDIZV': bdizv_compression,
#     'BDI1B': bdi1b_compression,
#     'BDIZV_non_optimal': bdizv_compression,
# }


class PerformanceSim(object):
    def __init__(self, quant : bool=True):
        self.total_op = 0      # number of total operations
        self.removed_op = 0    # number of removed operations
        self.quant = quant     # indicating whether target model is quantized (default: True)

    def get_performance(self):
        return self.total_op / (self.total_op - self.removed_op)

    def register_model(self, model: torch.nn.Module):
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226))

        def generate_hook(layer_name : str) -> Callable:
            def algo_sim_check_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                if 'conv' not in type(layer).__name__.lower():
                    pass

                ifm = input_tensor[0].int_repr() if self.quant else input_tensor[0]
                weight = model.state_dict()[layer_name + '.weight']

                lowered_ifm = ifm_lowering(ifm=ifm, layer_info=layer_info[layer_name])
                lowered_weight = weight_lowering(weight=weight, layer_info=layer_info[layer_name])

                print(f"Calculating performance with layer: {layer_name}")

                iw, ivecw = lowered_ifm.shape
                ww, wvecw = lowered_weight.shape

                if ivecw != wvecw:
                    raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

                self.total_op += iw * ww * ivecw

                imask = lowered_ifm != 0
                wmask = lowered_weight != 0

                for im_vec in imask:
                    for wm_vec in wmask:
                        self.removed_op += int(torch.logical_and(im_vec, wm_vec).count_nonzero())

            return algo_sim_check_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                sublayer.register_forward_hook(generate_hook(layer_name))

if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    sim = PerformanceSim(quant=True)
    sim.register_model(model)

    dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

    model.eval()
    model(dummy_image)

    print(sim.total_op, sim.removed_op)
    print(1 - sim.removed_op / sim.total_op)