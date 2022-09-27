from typing import Callable, Dict
import torch
import numpy as np
from scipy.sparse import csr_matrix

from compression.binary_array import array2binary
from compression.algorithms import bdizv_compression, bdi1b_compression, csr_compression, csc_compression, zeroval_compression

from models.tools.lowering import weight_lowering, ifm_lowering, ConvLayerInfo


class AcceleratorSim(object):
    supported_algorithms: Dict[str, Callable] = {
        'BDIZV': bdizv_compression,
        'BDI'  : bdi1b_compression,
        'ZVC'  : zeroval_compression,
        'CSR'  : csr_compression,
        'CSC'  : csc_compression,
    }

    def __init__(self, quant : bool=True, device : str='cpu'):
        self.total_op = 0    # number of total operations
        self.removed_op = 0  # number of removed operations

        self.total_siz = 0   # total uncompressed size
        self.compr_siz = {}  # compressed size

        self.quant = quant    # indicating whether target model is quantized (default: True)
        self.device = device  # pytorch device

    def get_performance(self):
        return self.total_op / (self.total_op - self.removed_op)

    def get_compression_ratio(self):
        return {algo_name : (self.total_siz / compr_siz) for algo_name, compr_siz in self.compr_siz.items()}

    def register_model(self, model: torch.nn.Module):
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device=self.device)

        def generate_hook(layer_name : str) -> Callable:
            def algo_sim_check_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                # Generation of lowered input feature map and weight data
                print(f"Generating lowered data with layer: {layer_name}", end='')

                if 'conv' not in type(layer).__name__.lower():
                    pass

                ifm = input_tensor[0]
                weight = model.state_dict()[layer_name + '.weight']

                if self.quant:
                    ifm = ifm.int_repr()
                    weight = weight.int_repr()

                lowered_ifm = ifm_lowering(ifm=ifm, layer_info=layer_info[layer_name]).detach().cpu().numpy()
                lowered_weight = weight_lowering(weight=weight, layer_info=layer_info[layer_name]).detach().cpu().numpy()

                # Performance Test
                print(f"\rCalculating performance with layer: {layer_name}", end='')
                iw, ivecw = lowered_ifm.shape
                ww, wvecw = lowered_weight.shape

                if ivecw != wvecw:
                    raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

                self.total_op += iw * ww * ivecw

                imask = lowered_ifm != 0
                wmask = lowered_weight != 0

                for im_vec in imask:
                    for wm_vec in wmask:
                        self.removed_op += int(np.count_nonzero(np.logical_and(im_vec, wm_vec)))

                # Compression Test
                print(f"\rCalculating compression ratio with layer: {layer_name}", end='')

                self.total_siz += (len(lowered_ifm.tobytes()) + len(lowered_weight.tobytes())) * 8

                for algo_name, algo_method in AcceleratorSim.supported_algorithms.items():
                    if algo_name not in self.compr_siz.keys():
                        self.compr_siz[algo_name] = 0

                    if algo_name == 'CSR' or algo_name == 'CSC':
                        self.compr_siz[algo_name] += len(algo_method(lowered_ifm, lowered_ifm.dtype.itemsize))
                        self.compr_siz[algo_name] += len(algo_method(lowered_weight, lowered_weight.dtype.itemsize))
                    else:
                        for i_vec in lowered_ifm:
                            self.compr_siz[algo_name] += len(algo_method(i_vec, i_vec.dtype.itemsize))

                        for w_vec in lowered_weight:
                            self.compr_siz[algo_name] += len(algo_method(w_vec, w_vec.dtype.itemsize))

                print(f"\rSimulation finished with layer: {layer_name}", end='\n')

            return algo_sim_check_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                sublayer.register_forward_hook(generate_hook(layer_name))


if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    sim = AcceleratorSim(quant=True)
    sim.register_model(model)

    dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

    model.eval()
    model(dummy_image)

    print(sim.get_performance())
    print(sim.get_compression_ratio())