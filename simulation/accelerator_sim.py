from typing import Callable, Dict, List
import torch
import numpy as np
from scipy.sparse import csr_matrix

from compression.binary_array import array2binary
from compression.algorithms import bdizv_compression, bdi1b_compression, csr_compression, csc_compression, zeroval_compression

from models.tools.lowering import weight_lowering, ifm_lowering, ConvLayerInfo


class AcceleratorConfig(object):
    def __init__(self, ve_num: int=128, mac_cycle: int=1, scheduler: bool=False):

        self.ve_num = ve_num        # number of VEs
        self.mac_cycle = mac_cycle  # mac cycle

        self.scheduler = scheduler


class CycleSim(object):
    def __init__(self, ac_config: AcceleratorConfig, quant: bool=True, device: str='cpu'):
        self.ac_config = ac_config  # accelerator configuration
        self.cycle_result = {}      # cycle result

        self.quant = quant    # indicating whether target model is quantized (default: True)
        self.device = device  # pytorch device

        self.ve_queue: np.ndarray or None = None

    def get_cycle(self):
        return self.cycle_result

    def _sim_init(self):
        self.ve_queue = np.array([0] * self.ac_config.ve_num)

    def register_model(self, model: torch.nn.Module, model_name: str='default'):
        model_name = type(model).__name__ if model_name == 'default' else model_name
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device=self.device)

        def generate_hook(model_name: str, layer_name : str) -> Callable:
            def cycle_sim_check_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                # Generation of lowered input feature map and weight data
                print(f"Generating lowered data with layer: {layer_name}", end='')

                result_key = (model_name, layer_name)

                if 'conv' not in type(layer).__name__.lower():
                    pass

                ifm = input_tensor[0]
                weight = model.state_dict()[layer_name + '.weight']

                if self.quant:
                    ifm = ifm.int_repr()
                    weight = weight.int_repr()

                lowered_ifm = ifm_lowering(ifm=ifm, layer_info=layer_info[layer_name]).detach().cpu().numpy()
                lowered_weight = weight_lowering(weight=weight,
                                                 layer_info=layer_info[layer_name]).detach().cpu().numpy()

                iw, ivecw = lowered_ifm.shape
                ww, wvecw = lowered_weight.shape

                if ivecw != wvecw:
                    raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

                # Run cycle test
                print(f"\rRun cycle test: {layer_name}", end='')

                self._sim_init()  # initialize VE queue
                op_cycle = 0

                imask = lowered_ifm != 0
                wmask = lowered_weight != 0

                for ipivot in range(0, iw, self.ac_config.ve_num):
                    for w_idx in range(0, ww, 1):
                        for ve_idx in range(self.ac_config.ve_num if (iw - ipivot >= self.ac_config.ve_num) else (iw - ipivot)):
                            im_vec = imask[ipivot + ve_idx]
                            wm_vec = wmask[w_idx]
                            self.ve_queue[ve_idx] += np.count_nonzero(np.logical_and(im_vec, wm_vec))

                        op_cycle += np.min(self.ve_queue)
                        self.ve_queue -= np.min(self.ve_queue)

                sparse_cycle = (op_cycle + np.max(self.ve_queue)) * self.ac_config.mac_cycle
                dense_cycle = ((iw // self.ac_config.ve_num) + (1 if iw % self.ac_config.ve_num else 0)) * ww * ivecw * self.ac_config.mac_cycle
                self.cycle_result[result_key] = (sparse_cycle, dense_cycle)

                print(f"\rSimulation finished with layer: {layer_name:30s}  "
                      f"sparse: {sparse_cycle:6d}  "
                      f"dense: {dense_cycle:6d}", end='\n')

            return cycle_sim_check_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                sublayer.register_forward_hook(generate_hook(model_name, layer_name))


class AcceleratorSim(object):
    supported_algorithms: Dict[str, Callable] = {
        'BDIZV': bdizv_compression,
        'BDI'  : bdi1b_compression,
        'ZVC'  : zeroval_compression,
        'CSR'  : csr_compression,
        'CSC'  : csc_compression,
    }
    algo_names = tuple(sorted(supported_algorithms.keys()))

    def __init__(self, quant : bool=True, device : str='cpu'):
        self.performance_result = {}  # key: (model type, layer name)  value: (total, valid, gain)
        self.weight_compression_result = {}
        self.ifm_compression_ratio = {}

        self.quant = quant    # indicating whether target model is quantized (default: True)
        self.device = device  # pytorch device

    def get_performance(self):
        return self.performance_result

    def get_weight_compression_ratio(self):
        return self.weight_compression_result

    def get_ifm_compression_ratio(self):
        return self.ifm_compression_ratio

    def register_model(self, model: torch.nn.Module, model_name: str='default'):
        model_name = type(model).__name__ if model_name == 'default' else model_name
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device=self.device)

        def generate_hook(model_name: str, layer_name : str) -> Callable:
            def algo_sim_check_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                # Generation of lowered input feature map and weight data
                print(f"Generating lowered data with layer: {layer_name}", end='')

                result_key = (model_name, layer_name)

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
                # self.performance_result[result_key] = [0, 0, 0]

                iw, ivecw = lowered_ifm.shape
                ww, wvecw = lowered_weight.shape

                if ivecw != wvecw:
                    raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

                total_op = iw * ww * ivecw
                valid_op = 0

                imask = lowered_ifm != 0
                wmask = lowered_weight != 0

                for im_vec in imask:
                    for wm_vec in wmask:
                        valid_op += int(np.count_nonzero(np.logical_and(im_vec, wm_vec)))

                self.performance_result[result_key] = (total_op, valid_op, total_op / (total_op - valid_op + 1e-10))

                # Compression Test
                print(f"\rCalculating compression ratio with layer: {layer_name}", end='')

                ifm_total_siz = len(lowered_ifm.tobytes()) * 8
                weight_total_siz = len(lowered_weight.tobytes()) * 8
                weight_compr_siz = {}
                ifm_compr_siz = {}

                for algo_name, algo_method in AcceleratorSim.supported_algorithms.items():
                    if algo_name not in weight_compr_siz.keys():
                        weight_compr_siz[algo_name] = 0

                    if algo_name not in ifm_compr_siz.keys():
                        ifm_compr_siz[algo_name] = 0

                    if algo_name == 'CSR' or algo_name == 'CSC':
                        ifm_compr_siz[algo_name] += len(algo_method(lowered_ifm, lowered_ifm.dtype.itemsize*8))
                        weight_compr_siz[algo_name] += len(algo_method(lowered_weight, lowered_weight.dtype.itemsize*8))
                    else:
                        for i_vec in lowered_ifm:
                            ifm_compr_siz[algo_name] += len(algo_method(i_vec, i_vec.dtype.itemsize*8))

                        for w_vec in lowered_weight:
                            weight_compr_siz[algo_name] += len(algo_method(w_vec, w_vec.dtype.itemsize*8))

                self.ifm_compression_ratio[result_key] = [ifm_total_siz / ifm_compr_siz[name] for name in AcceleratorSim.algo_names]
                self.weight_compression_result[result_key] = [weight_total_siz / weight_compr_siz[name] for name in AcceleratorSim.algo_names]

                print(f"\rSimulation finished with layer: {layer_name}", end='\n')

            return algo_sim_check_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                sublayer.register_forward_hook(generate_hook(model_name, layer_name))


if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    sim = AcceleratorSim(quant=False,)
    sim.register_model(model)

    dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

    model.eval()
    model(dummy_image)

    print(sim.get_performance())
    print(sim.get_compression_ratio())