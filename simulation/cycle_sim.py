import os
import sys
import numpy as np

from simulation.sim_metaclass import Sim
from models.tools.lowering import weight_lowering, ifm_lowering

# This script requires SystemPy library
# SystemPy is not available right now because it is on development
# Please contact to su8939@skku.edu if you have any issue running this script

try:
    from simulation.accelerators.compressed_accelerator_prev import CompressedAccelerator, auto_config_matrices, restore_activation_mat
    from simulation.accelerators.systolic_array_only_cycles import systolic_array_cycles_ws
except ImportError:
    # raise Exception("[ERROR] This script requires SystemPy library. "
    #                 "SystemPy is not available right now because it is on development. "
    #                 "Please contact to su8939@skku.edu if you have any issue running this script. ")

    sys.path.append(os.path.join(os.curdir, '..', 'SystemPy'))

    from simulation.accelerators.compressed_accelerator_prev import CompressedAccelerator, auto_config_matrices, \
        restore_activation_mat
    from simulation.accelerators.systolic_array_only_cycles import systolic_array_cycles_ws


class CompressedAcceleratorCycleSim(Sim):
    def __init__(self, engine_num, pe_num, mult_num, chunk_size, fifo_capacity,
                 sa_shape=(8, 8), wgt_tile_shape=(32, 32), act_tile_shape=(32, 32), sampling_factor=10, quant=True):

        super(CompressedAcceleratorCycleSim, self).__init__()

        self.engine_num    = engine_num     # number of engines
        self.pe_num        = pe_num         # number of PEs
        self.mult_num      = mult_num       # number of multipliers
        self.chunk_size    = chunk_size     # size of a chunk
        self.fifo_capacity = fifo_capacity  # capacity of FIFO inside the VE

        self.sa_shape = sa_shape  # shape of systolic array

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
                cycles[0] += self._run_compressed_accelerator(input_tensor, weight_tensor)
                cycles[1] += systolic_array_cycles_ws(arr_shape=self.sa_shape, act_shape=input_tensor.shape, wgt_shape=weight_tensor.shape)
            else:
                ih, iw = input_tensor.shape
                wh, ww = weight_tensor.shape
                # th, tw = self.tile_shape
                wth, wtw = self.wgt_tile_shape
                ith, itw = self.act_tile_shape

                if wtw < self.pe_num:
                    print(f"- [WARNING] weight tile width is smaller than the number of PEs")

                if ith < self.pe_num:
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

                print(f"- calculating tiled multiplication with systolic array (tile shape: {self.sa_shape})")
                cycles[1] += systolic_array_cycles_ws(arr_shape=self.sa_shape, act_shape=input_tensor.shape, wgt_shape=weight_tensor.shape)

                print(f"- calculating tiled multiplication with compressed accelerator (total tile multiplications: {total_tmul})\n"
                      f"- weight shape: {weight_tensor.shape}  input shape:  {input_tensor.shape}")

                if ww != ih:
                    print(f"- [WARNING] shape mismatch  weight: {weight_tensor.shape}  input: {input_tensor.shape}")

                for iidx in range(iw // itw):
                    for widx in range(wh // wth):
                        for tidx in range(ww // wtw):
                            if self.sampling_factor != 0 and sample_cnt > self.sampling_factor:
                                break

                            input_tile = input_tensor[tidx*ith:(tidx+1)*ith, iidx*itw:(iidx+1)*itw]
                            weight_tile = weight_tensor[widx*wth:(widx+1)*wth, tidx*wtw:(tidx+1)*wtw]

                            # ith, itw = input_tile.shape
                            # if itw < self.pe_num:
                            #     input_tile = np.pad(input_tile, ((0, 0), (0, self.pe_num - itw)), 'constant', constant_values=0)

                            ca_cycle = self._run_compressed_accelerator(input_tile, weight_tile, tile_index=sample_cnt)

                            cycles[0] += ca_cycle
                            sample_cnt += 1

                cycles[0] = ((cycles[0] // sample_cnt) * total_tmul) // self.engine_num

            self.result[key] = cycles

            print(
                f"- cycle simulation: {model_name:15s} {submodule_name:30s}  "
                f"compressed accelerator: {cycles[0]}  "
                f"systolic array: {cycles[1]}  "
                f"performance gain: {cycles[1] / cycles[0]:.6f}\n"
            )

        return layer_cycles_hook

    def _run_compressed_accelerator(self, input_tensor, weight_tensor, tile_index=-1):
        # act_shape = input_tensor.shape
        # wgt_shape = weight_tensor.shape
        #
        # print((f'\r[tile {tile_index}]  ' if tile_index != -1 else '') + f'- calculating cycle of systolic array (SystolicArray)', end='')
        # sa_cycle = systolic_array_cycles(arr_shape=self.sa_shape, act_shape=act_shape, wgt_shape=wgt_shape)

        print((f'\r[tile {tile_index}]  ' if tile_index != -1 else '') + f'- simulating with compressed accelerator (CompressedAccelerator)', end='')
        weight_queue, weight_masks, weight_controls, activation_queue_arr, activation_masks_arr = auto_config_matrices(
            weight=weight_tensor, activation=input_tensor, pe_num=self.pe_num, chunk_size=self.chunk_size,
        )

        print((f'\r[tile {tile_index}]  ' if tile_index != -1 else '') + f'- weight mappings: {len(weight_masks)}  activation mappings: {len(activation_masks_arr[0])}', end='')
        ca_unit = CompressedAccelerator(
            mult_num=self.mult_num, pe_num=self.pe_num, chunk_size=self.chunk_size, fifo_capacity=self.fifo_capacity).compile(verbose=False)
        ca_unit.run(reset_n=1)
        ca_unit.run(reset_n=0)
        ca_unit.run(reset_n=1)

        w_m_valid, a_m_valid = 1, [1] * self.pe_num
        w_d_valid, a_d_valid = 1, [1] * self.pe_num

        ca_cycle = 0
        target_output_num = np.count_nonzero(weight_controls)
        output_num = np.zeros(shape=self.pe_num)

        while True:
            ca_unit.run(
                clk=0,
                control=0 if len(weight_controls) == 0 else weight_controls[0],
                w_d_valid=w_d_valid,
                w_m_valid=w_m_valid,
                a_d_valid_arr=a_d_valid,
                a_m_valid_arr=a_m_valid,
                w_d_in=np.zeros(shape=self.chunk_size) if len(weight_queue) == 0 else weight_queue[0],
                w_m_in=0 if len(weight_masks) == 0 else weight_masks[0],
                a_d_in_arr=[np.zeros(shape=self.chunk_size) if len(aq) == 0 else aq[0] for aq in activation_queue_arr],
                a_m_in_arr=[0 if len(am) == 0 else am[0] for am in activation_masks_arr],
            )
            ca_unit.run(clk=1)

            for ps_idx, ps_valid in enumerate(ca_unit.ps_valid_arr):
                if ps_valid.get() == 1:
                    output_num[ps_idx] += 1

            sys.stdout.write(
                f"\r[tile {tile_index}]  "
                f"cycle: {ca_cycle}  "
                f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  " + ' '*20)

            if ca_unit.w_d_in_required == 1 and len(weight_queue):
                del weight_queue[0]
                w_d_valid = 1 if len(weight_queue) else 0
            else:
                w_d_valid = 0

            if ca_unit.w_m_in_required == 1 and len(weight_masks):
                del weight_masks[0]
                del weight_controls[0]
                w_m_valid = 1 if len(weight_masks) else 0
            else:
                w_m_valid = 0

            for vidx in range(self.pe_num):
                if ca_unit.a_d_in_required_arr[vidx] == 1 and len(activation_queue_arr[vidx]):
                    del activation_queue_arr[vidx][0]
                    a_d_valid[vidx] = 1 if len(activation_queue_arr[vidx]) else 0
                else:
                    a_d_valid[vidx] = 0

                if ca_unit.a_m_in_required_arr[vidx] == 1 and len(activation_masks_arr[vidx]):
                    del activation_masks_arr[vidx][0]
                    a_m_valid[vidx] = 1 if len(activation_masks_arr[vidx]) else 0
                else:
                    a_m_valid[vidx] = 0

            # if ca_unit.is_idle() and len(weight_masks) == 0:
            #     break

            if np.count_nonzero(np.array(output_num) != target_output_num) == 0:
                break

            ca_cycle += 1

        print("\r", end='')

        return ca_cycle


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from models.model_presets import generate_from_quant_chkpoint, imagenet_pretrained
    from models.tools.imagenet_utils.dataset_loader import val_dataset, val_sampler

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    model = generate_from_quant_chkpoint(
        model_primitive=imagenet_pretrained['AlexNet'].generate(),
        chkpoint_path=os.path.join(os.curdir, '..', 'model_output', 'AlexNet_quantized_tuned_citer_10_pruned_pamt_0.5.pth'))

    sim = CompressedAcceleratorCycleSim(pe_num=16, mult_num=2, chunk_size=4, fifo_capacity=8, sa_shape=(8, 8),
                                        tile_shape=(32, 32), sampling_factor=10, quant=True)
    sim.register_model(model=model, model_name='AlexNet')

    # Inference one batch
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, sampler=val_sampler)
    images, _ = next(iter(val_loader))

    model = model.to('cpu')
    images = images.to('cpu')

    model.eval()
    model(images)

    for (model_name, layer_name), (ca_cycle, sa_cycle) in sim.result.items():
        print(
            f"cycle simulation: {model_name:15s} {layer_name:30s}  "
            f"compressed accelerator: {ca_cycle}  "
            f"systolic array: {sa_cycle}"
        )