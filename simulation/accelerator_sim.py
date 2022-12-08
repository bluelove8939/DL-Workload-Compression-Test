from typing import Callable
import torch
import numpy as np

from simulation.sim_metaclass import Sim
from models.tools.lowering import weight_lowering, ifm_lowering, ConvLayerInfo


class AcceleratorConfig(object):
    def __init__(self, ve_num: int=128, mac_cycle: int=1, scheduler: bool=False):

        self.ve_num = ve_num        # number of VEs
        self.mac_cycle = mac_cycle  # mac cycle

        self.scheduler = scheduler


class CycleSim(Sim):
    def __init__(self, ac_config: AcceleratorConfig, quant: bool=True, device: str='cpu'):
        super(CycleSim, self).__init__()

        self.ac_config = ac_config  # accelerator configuration
        self.result = {}      # cycle result

        self.quant = quant    # indicating whether target model is quantized (default: True)
        self.device = device  # pytorch device

        self.ve_queue: np.ndarray or None = None

    def _sim_init(self):
        self.ve_queue = np.array([0] * self.ac_config.ve_num)

    def generate_hook(self, model_name: str, submodule_name: str, submodule_info: ConvLayerInfo) -> Callable:
        def cycle_sim_check_hook(layer: torch.nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
            print(f'cycle evaluation hook called for {model_name}.{submodule_name}')

            result_key = (model_name, submodule_name)

            if 'conv' not in type(layer).__name__.lower():
                pass

            print('- lowering input feature map and weight tensor')
            ifm = input_tensor[0]
            weight = model.state_dict()[submodule_name + '.weight']

            if self.quant:
                ifm = ifm.int_repr()
                weight = weight.int_repr()

            lowered_ifm = ifm_lowering(ifm=ifm, layer_info=submodule_info).detach().cpu().numpy()
            lowered_weight = weight_lowering(weight=weight, layer_info=submodule_info).detach().cpu().numpy()

            iw, ivecw = lowered_ifm.shape
            ww, wvecw = lowered_weight.shape

            if ivecw != wvecw:
                raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

            # Run cycle test
            print(f"- start measuring the operation cycle")

            self._sim_init()  # initialize VE queue
            op_cycle = 0

            imask = lowered_ifm != 0
            wmask = lowered_weight != 0

            for ipivot in range(0, iw, self.ac_config.ve_num):
                for w_idx in range(0, ww, 1):
                    for ve_idx in range(
                            self.ac_config.ve_num if (iw - ipivot >= self.ac_config.ve_num) else (iw - ipivot)):
                        im_vec = imask[ipivot + ve_idx]
                        wm_vec = wmask[w_idx]
                        self.ve_queue[ve_idx] += np.count_nonzero(np.logical_and(im_vec, wm_vec))

                    op_cycle += np.min(self.ve_queue)
                    self.ve_queue -= np.min(self.ve_queue)

            sparse_cycle = (op_cycle + np.max(self.ve_queue)) * self.ac_config.mac_cycle
            dense_cycle = ((iw // self.ac_config.ve_num) + (
                1 if iw % self.ac_config.ve_num else 0)) * ww * ivecw * self.ac_config.mac_cycle
            self.result[result_key] = (sparse_cycle, dense_cycle, dense_cycle / (sparse_cycle + 1e-10))

            print(
                f"cycle evaluation: {model_name:15s} {submodule_name:30s} "
                f"sparse cycle: {sparse_cycle}, dense cycle: {dense_cycle}, gain: {dense_cycle / (sparse_cycle + 1e-10) * 100:.2f}%"
            )

        return cycle_sim_check_hook


class PerformanceSim(Sim):
    def __init__(self, quant : bool=True, device : str='cpu'):
        super(PerformanceSim, self).__init__()

        self.result = {}  # key: (model type, layer name)  value: (total, valid, gain)

        self.quant = quant    # indicating whether target model is quantized (default: True)
        self.device = device  # pytorch device

        self.compr_gran = 8

    def generate_hook(self, model_name, submodule_name, submodule_info) -> callable:
        def performance_eval_hook(layer: torch.nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
            print(f'performance evaluation hook called for {model_name}.{submodule_name}')

            result_key = (model_name, submodule_name)

            if 'conv' not in type(layer).__name__.lower():
                pass

            print('- lowering input feature map and weight tensor')
            ifm = input_tensor[0]
            weight = model.state_dict()[submodule_name + '.weight']

            if self.quant:
                ifm = ifm.int_repr()
                weight = weight.int_repr()

            lowered_ifm = ifm_lowering(ifm=ifm, layer_info=submodule_info).detach().cpu().numpy()
            lowered_weight = weight_lowering(weight=weight, layer_info=submodule_info).detach().cpu().numpy()

            iw, ivecw = lowered_ifm.shape
            ww, wvecw = lowered_weight.shape

            if ivecw != wvecw:
                raise Exception(f'Lowering algorithm may have an error {lowered_ifm.shape} {lowered_weight.shape}')

            # Performance Test
            print(f"- start measuring the number of ineffectual operations")

            total_op = iw * ww * ivecw
            valid_op = 0

            imask = lowered_ifm != 0
            wmask = lowered_weight != 0

            for im_vec in imask:
                for wm_vec in wmask:
                    valid_op += int(np.count_nonzero(np.logical_and(im_vec, wm_vec)))

            self.result[result_key] = (total_op, valid_op, total_op / (valid_op + 1e-10))

            print(
                f"performance evaluation: {model_name:15s} {submodule_name:30s} "
                f"total: {total_op}, valid: {valid_op}, ratio: {total_op / (valid_op + 1e-10) * 100:.2f}%"
            )

        return performance_eval_hook


if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    sim = PerformanceSim(quant=False,)
    sim.register_model(model, 'ResNet50')

    dummy_image = torch.tensor(np.zeros(shape=(1, 3, 226, 226), dtype=np.dtype('float32')))

    model.eval()
    model(dummy_image)

    print(sim.result)