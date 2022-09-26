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
    def __init__(self, layer_info, quant : bool=True):
        self.total_op = 0             # number of total operations
        self.removed_op = 0           # number of removed operations
        self.quant = quant            # indicating whether target model is quantized (default: True)
        self.layer_info = layer_info  # information of convolution layers

    def generate_hook(self, layer_name : str) -> Callable:
        def hook(layer, input_tensor, output_tensor):
            if 'conv' not in type(layer).__name__.lower():
                pass

            lowered_ifm = ifm_lowering(ifm=input_tensor, layer_info=ConvLayerInfo.generate_from_layer(layer))
        return hook

if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    def print_children(layer : torch.nn.Module, prefix : str or None=None):
        prefix = ('' if prefix is None else prefix)

        for sublayer_name, sublayer in layer.named_children():
            # if ('conv' in type(sublayer).__name__.lower()) or \
            #         ('relu' in type(sublayer).__name__.lower()):
            #     print(prefix + sublayer_name, sublayer)

            if 'conv' in type(sublayer).__name__.lower():
                print(prefix + sublayer_name, sublayer, sublayer.weight().shape)

            print_children(sublayer, prefix= prefix + sublayer_name + '.')

    print_children(model)

    # def generate_input_shape_hook(layer_name):
    #     def hook(model, input_tensor, output_tensor):
    #         q_input_tensor = input_tensor[0].int_repr()
    #         sparsity = 1 - torch.count_nonzero(q_input_tensor) / torch.numel(q_input_tensor)
    #
    #         print(model)
    #
    #         print(f"name: {layer_name:25s}  sparsity: {sparsity:.4f}")
    #     return hook
    #
    # for lname, layer in model.named_modules():
    #     if 'conv' in type(layer).__name__.lower():
    #         layer.register_forward_hook(generate_input_shape_hook(lname))
    #
    # W, H = 224, 224  # size of input image
    # dummy_image = torch.tensor(np.zeros(shape=(1, 3, H, W), dtype=np.dtype('float32')))
    #
    # model.eval()
    # model(dummy_image)