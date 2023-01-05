import sys
import numpy as np

from simulation.accelerators.accelerator_configs import SystolicArrayWSConfig, CompressedAcceleratorConfig
from simulation.accelerators.systolic_array_only_cycles import systolic_array_cycles_ws
from simulation.accelerators.compressed_accelerator import auto_config_matrices


def run_systolic_array_only_cycles(weight_tensor, input_tensor, config: SystolicArrayWSConfig, tile_index: int=-1, verbose: bool=False):
    return systolic_array_cycles_ws(arr_shape=config.sa_shape, wgt_shape=weight_tensor.shape, act_shape=input_tensor.shape)


def run_compressed_accelerator(weight_tensor, input_tensor, config: CompressedAcceleratorConfig, tile_index: int=-1, verbose: bool=False):
    if verbose:
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=np.inf)

    weight_queue, weight_masks, weight_controls, activation_queue_arr, activation_masks_arr = auto_config_matrices(
        weight=weight_tensor, activation=input_tensor, pe_num=config.pe_num, chunk_size=config.chunk_size,
    )

    ca_unit = config.generate()
    ca_unit.run(reset_n=1)
    ca_unit.run(reset_n=0)
    ca_unit.run(reset_n=1)

    w_m_valid, a_m_valid = 1, [1] * config.pe_num
    w_d_valid, a_d_valid = 1, [1] * config.pe_num

    ca_cycle = 0
    target_output_num = np.count_nonzero(weight_controls)
    output_num = np.zeros(shape=config.pe_num)

    while True:
        ca_unit.run(
            clk=0,
            control=0 if len(weight_controls) == 0 else weight_controls[0],
            w_d_valid=w_d_valid,
            w_m_valid=w_m_valid,
            a_d_valid_arr=a_d_valid,
            a_m_valid_arr=a_m_valid,
            w_d_in=np.zeros(shape=config.chunk_size) if len(weight_queue) == 0 else weight_queue[0],
            w_m_in=0 if len(weight_masks) == 0 else weight_masks[0],
            a_d_in_arr=[np.zeros(shape=config.chunk_size) if len(aq) == 0 else aq[0] for aq in activation_queue_arr],
            a_m_in_arr=[0 if len(am) == 0 else am[0] for am in activation_masks_arr],
        )
        ca_unit.run(clk=1)

        for ps_idx, ps_valid in enumerate(ca_unit.ps_valid_arr):
            if ps_valid.get() == 1:
                output_num[ps_idx] += 1
        if verbose:
            sys.stdout.write(
                (f"\r[tile {tile_index}]  " if tile_index != -1 else '\r') +
                f"[{np.sum(np.array(output_num)) / (target_output_num*len(output_num))*100:5.2f}%]  "
                f"cycle: {ca_cycle}  "
                # f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  " + ' ' * 20
            )

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

        for vidx in range(config.pe_num):
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

        if np.count_nonzero(np.array(output_num) != target_output_num) == 0:
            break

        ca_cycle += 1

    ca_cycle //= config.engine_num

    if verbose:
        print("\r", end='')

    return ca_cycle


if __name__ == '__main__':
    weight_sparsity = 0.7
    input_sparsity = 0.7
    testcase = 5

    for t in range(testcase):
        sa_config = SystolicArrayWSConfig(sa_shape=(8, 8))
        ca_config = CompressedAcceleratorConfig(engine_num=2, pe_num=32, mult_num=1, chunk_size=4, fifo_capacity=12)

        weight_tensor = np.random.randint(0, 256, size=(64, 64), dtype='int32')
        input_tensor = np.random.randint(0, 256, size=(64, 64), dtype='int32')

        input_tensor[np.random.choice(2, size=input_tensor.shape, p=[1 - input_sparsity, input_sparsity]).astype('bool')] = 0
        weight_tensor[np.random.choice(2, size=weight_tensor.shape, p=[1 - weight_sparsity, weight_sparsity]).astype('bool')] = 0

        sa_cycle = run_systolic_array_only_cycles(weight_tensor, input_tensor, config=sa_config, tile_index=t, verbose=True)
        ca_cycle = run_compressed_accelerator(weight_tensor, input_tensor, config=ca_config, tile_index=t, verbose=True)

        print(f"Case #{t+1}  "
              f"weight sparsity: {(1 - np.count_nonzero(weight_tensor) / np.size(weight_tensor))*100:.2f}%  "
              f"input sparsity: {(1 - np.count_nonzero(input_tensor) / np.size(input_tensor))*100:.2f}%  "
              f"compressed accelerator: {ca_cycle}  "
              f"systolic array: {sa_cycle}  "
              f"performance gain: {sa_cycle / ca_cycle:.6f}")


# if __name__ == '__main__':
#     weight_sparsity = 0.5
#     input_sparsity = 0.0
#     testcase = 5
#
#     for t in range(testcase):
#         sa_config = SystolicArrayWSConfig(sa_shape=(8, 8))
#         ca_config = CompressedAcceleratorConfig(engine_num=4, pe_num=16, mult_num=1, chunk_size=4, fifo_capacity=8)
#
#         weight_tensor = np.random.randint(0, 256, size=(256, 2304), dtype='int32')
#         input_tensor = np.random.randint(0, 256, size=(2304, 169), dtype='int32')
#
#         sa_cycle = run_systolic_array_only_cycles(weight_tensor, input_tensor, config=sa_config, tile_index=t, verbose=True)
#
#         print(f"{sa_cycle}")