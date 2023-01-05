import os
import sys
import numpy as np

from simulation.sim_metaclass import Sim
from models.tools.lowering import weight_lowering, ifm_lowering

# This script requires SystemPy library
# SystemPy is not available right now because it is on development
# Please contact to su8939@skku.edu if you have any issue running this script

try:
    from simulation.accelerators.run_cycle_simulation import run_systolic_array_only_cycles, run_compressed_accelerator
    from simulation.accelerators.accelerator_configs import CompressedAcceleratorConfig, SystolicArrayWSConfig
except ImportError:
    # raise Exception("[ERROR] This script requires SystemPy library. "
    #                 "SystemPy is not available right now because it is on development. "
    #                 "Please contact to su8939@skku.edu if you have any issue running this script. ")

    sys.path.append(os.path.join(os.curdir, '..', 'SystemPy'))

    from simulation.accelerators.run_cycle_simulation import run_systolic_array_only_cycles, run_compressed_accelerator
    from simulation.accelerators.accelerator_configs import CompressedAcceleratorConfig, SystolicArrayWSConfig


class CompressedAcceleratorCycleSim(Sim):
    def __init__(self, ca_config: CompressedAcceleratorConfig, sa_config: SystolicArrayWSConfig,
                 wgt_tile_shape=(32, 32), act_tile_shape=(32, 32), sampling_factor=10, quant=True):

        super(CompressedAcceleratorCycleSim, self).__init__()

        # self.engine_num    = engine_num     # number of engines
        # self.pe_num        = pe_num         # number of PEs
        # self.mult_num      = mult_num       # number of multipliers
        # self.chunk_size    = chunk_size     # size of a chunk
        # self.fifo_capacity = fifo_capacity  # capacity of FIFO inside the VE
        self.ca_config = ca_config
        self.sa_config = sa_config

        # self.sa_shape = sa_shape  # shape of systolic array

        self.wgt_tile_shape = wgt_tile_shape    # shape of a weight tile
        self.act_tile_shape = act_tile_shape    # shape of a activation tile
        self.sampling_factor = sampling_factor  # number of tiles to sample

        self.quant = quant
        self.result = {}

    def generate_hook(self, model_name, submodule_name, submodule_info) -> callable:
        def layer_cycles_hook(submodule, input_tensor, output_tensor):
            print(f'cycle accurate test hook called for {model_name}.{submodule_name}')
            key = (model_name, submodule_name)
            cycles = [0, 0]

            input_tensor = input_tensor[0]
            weight_tensor = submodule.weight

            if self.quant:
                input_tensor = input_tensor.detach().int_repr()
                weight_tensor = weight_tensor().detach().int_repr()

            if submodule_info.is_linear():
                input_tensor = input_tensor.numpy().T
                weight_tensor = weight_tensor.numpy()
            elif submodule_info.is_conv():
                print('- lowering input and weight tensor')
                input_tensor = ifm_lowering(ifm=input_tensor, layer_info=submodule_info, verbose=False).detach().cpu().numpy().T
                weight_tensor = weight_lowering(weight=weight_tensor, layer_info=submodule_info).detach().cpu().numpy()
            else:
                raise Exception(f"[ERROR] Invalid submodule information: {submodule_info}")

            if self.wgt_tile_shape is None or self.act_tile_shape is None:
                cycles[0] += run_compressed_accelerator(weight_tensor, input_tensor, config=self.ca_config, tile_index=-1, verbose=True)
                cycles[1] += run_systolic_array_only_cycles(weight_tensor, input_tensor, config=self.sa_config, tile_index=-1, verbose=True)
            else:
                ih, iw = input_tensor.shape
                wh, ww = weight_tensor.shape
                # th, tw = self.tile_shape
                wth, wtw = self.wgt_tile_shape
                ith, itw = self.act_tile_shape

                if wtw < self.ca_config.pe_num:
                    print(f"- [WARNING] weight tile width is smaller than the number of PEs")

                if ith < self.ca_config.pe_num:
                    print(f"- [WARNING] activation tile height is smaller than the number of PEs")

                if wtw != ith:
                    raise Exception(f"tile shape mismatch -> weight tile: {self.wgt_tile_shape}  activation tile: {self.act_tile_shape}")

                # zeropad input and weight tensors
                if ih % ith:
                    input_tensor = np.pad(input_tensor, ((0, ith - (ith % ih)), (0, 0)), 'constant', constant_values=0)
                if iw % itw:
                    input_tensor = np.pad(input_tensor, ((0, 0), (0, itw - (itw % iw))), 'constant', constant_values=0)
                if wh % wth:
                    weight_tensor = np.pad(weight_tensor, ((0, wth - (wth % wh)), (0, 0)), 'constant', constant_values=0)
                if ww % wtw:
                    weight_tensor = np.pad(weight_tensor, ((0, 0), (0, wtw - (wtw % ww))), 'constant', constant_values=0)

                # tiled matrix multiplication
                ih, iw = input_tensor.shape
                wh, ww = weight_tensor.shape
                sample_cnt = 0
                total_tmul = (iw // itw) * (wh // wth) * (ww // wtw)

                # print(f"- calculating tiled multiplication with systolic array (tile shape: {self.sa_config.sa_shape})")
                # cycles[1] += run_systolic_array_only_cycles(weight_tensor, input_tensor, config=self.sa_config, tile_index=sample_cnt, verbose=True)
                # print(f"- systolic array cycles: {cycles[1]}")

                print(f"- calculating tiled multiplication with compressed accelerator (total tile multiplications: {total_tmul})\n"
                      f"- weight shape: {weight_tensor.shape}  input shape:  {input_tensor.shape}")

                if ww != ih:
                    print(f"- [WARNING] shape mismatch  weight: {weight_tensor.shape}  input: {input_tensor.shape}")

                for iidx in range(0, iw // itw, 4):
                    for widx in range(0, wh // wth, 1):
                        for tidx in range(0, ww // wtw, 2):
                            if self.sampling_factor != 0 and sample_cnt > self.sampling_factor:
                                break

                            input_tile = input_tensor[tidx*ith:(tidx+1)*ith, iidx*itw:(iidx+1)*itw]
                            weight_tile = weight_tensor[widx*wth:(widx+1)*wth, tidx*wtw:(tidx+1)*wtw]

                            sa_cycle = run_systolic_array_only_cycles(weight_tile, input_tile, config=self.sa_config, tile_index=sample_cnt, verbose=True)
                            cycles[1] += sa_cycle

                            ca_cycle = run_compressed_accelerator(weight_tile, input_tile, config=self.ca_config, tile_index=sample_cnt+1, verbose=True)
                            cycles[0] += ca_cycle
                            sample_cnt += 1

                            # print(f"  tile {sample_cnt}  "
                            #       f"weight sparsity: {(1 - np.count_nonzero(weight_tile) / np.size(weight_tile))*100:5.2f}%  "
                            #       f"input sparsity: {(1 - np.count_nonzero(input_tile) / np.size(input_tile))*100:5.2f}%  "
                            #       f"cycles: {ca_cycle}({cycles[0] // sample_cnt})  "
                            #       f"expected gain: {cycles[1] / ((cycles[0] // sample_cnt) * total_tmul):.6f}")


                cycles[0] = (cycles[0] // sample_cnt) * total_tmul
                cycles[1] = (cycles[1] // sample_cnt) * total_tmul

            self.result[key] = cycles

            print(
                f"- cycle simulation: {model_name:15s} {submodule_name:30s}  "
                f"compressed accelerator: {cycles[0]}  "
                f"systolic array: {cycles[1]}  "
                f"performance gain: {cycles[1] / cycles[0]:.6f}\n"
            )

        return layer_cycles_hook