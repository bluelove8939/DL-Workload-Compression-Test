import os
import numpy as np

from simulation.accelerators.accelerator_configs import CompressedAcceleratorConfig, SystolicArrayWSConfig
from simulation.accelerators.run_cycle_simulation import run_compressed_accelerator, run_systolic_array_only_cycles


def generate_matrix(shape: tuple[int, int], sparsity: float=0.0):
    mat = np.random.randint(0, 256, size=(64, 64), dtype='int32')
    mat[np.random.choice(2, size=mat.shape, p=[1 - sparsity, sparsity]).astype('bool')] = 0

    return mat


if __name__ == '__main__':
    log_dirname = os.path.join(os.curdir, 'logs')
    log_filename = f"{os.path.split(__file__)[1].split('.')[0]}.csv"

    testcase = 5
    sparsity = [(0.0, 0.0), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]

    wgt_shape = (64, 64)
    act_shape = (64, 64)

    ca_configs = [
        # number of multipliers: 1  capacity of the FIFO: 12
        CompressedAcceleratorConfig(engine_num=1, pe_num=64, mult_num=1, chunk_size=4, fifo_capacity=12),
        CompressedAcceleratorConfig(engine_num=2, pe_num=32, mult_num=1, chunk_size=4, fifo_capacity=12),
        CompressedAcceleratorConfig(engine_num=4, pe_num=16, mult_num=1, chunk_size=4, fifo_capacity=12),

        # number of multipliers: 2  capacity of the FIFO: 12
        CompressedAcceleratorConfig(engine_num=1, pe_num=32, mult_num=2, chunk_size=4, fifo_capacity=12),
        CompressedAcceleratorConfig(engine_num=2, pe_num=16, mult_num=2, chunk_size=4, fifo_capacity=12),

        # number of multipliers: 1  capacity of the FIFO: 8
        CompressedAcceleratorConfig(engine_num=1, pe_num=64, mult_num=1, chunk_size=4, fifo_capacity=8),
        CompressedAcceleratorConfig(engine_num=2, pe_num=32, mult_num=1, chunk_size=4, fifo_capacity=8),
        CompressedAcceleratorConfig(engine_num=4, pe_num=16, mult_num=1, chunk_size=4, fifo_capacity=8),

        # number of multipliers: 1  capacity of the FIFO: 8
        CompressedAcceleratorConfig(engine_num=1, pe_num=32, mult_num=2, chunk_size=4, fifo_capacity=8),
        CompressedAcceleratorConfig(engine_num=2, pe_num=16, mult_num=2, chunk_size=4, fifo_capacity=8),
    ]

    sa_config = SystolicArrayWSConfig(sa_shape=(8, 8))

    content = ["engine num,pe num,mult num,chunk size,fifo capacity,weight sparsity,input sparsity,ca cycle,sa cycle,gain"]

    for ca_idx, ca_config in enumerate(ca_configs):
        for wgt_sparsity, act_sparsity in sparsity:
            sa_cycle = 0
            ca_cycle = 0

            for t in range(testcase):
                wgt_mat = generate_matrix(shape=wgt_shape, sparsity=wgt_sparsity)
                act_mat = generate_matrix(shape=act_shape, sparsity=act_sparsity)

                sa_cycle += run_systolic_array_only_cycles(wgt_mat, act_mat, config=sa_config, tile_index=t, verbose=True)
                ca_cycle += run_compressed_accelerator(wgt_mat, act_mat, config=ca_config, tile_index=t, verbose=True)

            sa_cycle //= testcase
            ca_cycle //= testcase

            content.append(f"{ca_config.engine_num},{ca_config.pe_num},{ca_config.mult_num},{ca_config.chunk_size},"
                           f"{ca_config.fifo_capacity},{wgt_sparsity:.1f},{act_sparsity:.1f},"
                           f"{ca_cycle},{sa_cycle},{sa_cycle / ca_cycle:.6f}")

            print(f"[config {ca_idx}]  wgt sparsity: {wgt_sparsity:.1f}  act sparsity: {act_sparsity:.1f}  "
                  f"act ca: {ca_cycle}  sa: {sa_cycle}  gain: {sa_cycle / ca_cycle:.6f}")

    with open(os.path.join(log_dirname, log_filename), 'wt') as file:
        file.write('\n'.join(content))
