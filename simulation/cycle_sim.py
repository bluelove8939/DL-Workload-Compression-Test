import os
import sys
import numpy as np

sys.path.append(os.path.join(os.curdir, '..', '..', 'SystemPy'))

from compressed_accelerator import CompressedAccelerator, auto_config_matrices, restore_activation_mat


# Testbench for CompressedAccelerator
if __name__ == '__main__':
    # Verbose options
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # Simulation setup
    act_shape = (32, 32)  # shape of activation matrix
    wgt_shape = (32, 32)  # shape of weight matrix
    out_shape = (wgt_shape[0], act_shape[1])  # shape of output matrix

    ve_num = 16        # number of VEs
    chunk_size = 4     # size of a chunk
    fifo_capacity = 8  # capacity of FIFO inside the VE
    sparsity = 0.5     # sparsity of the input vectors (0 to 1)

    testcase = 5
    verbose = False

    # Instantiation of CompressedAccelerator
    ca_unit = CompressedAccelerator(
        ve_num=ve_num, chunk_size=chunk_size, fifo_capacity=fifo_capacity).compile(verbose=verbose)
    ca_unit.run(reset_n=1)

    for t in range(testcase):
        # Generate matrices and queues
        activation_matrix = (np.random.rand(*act_shape) * 10).astype(dtype=np.dtype('int32'))
        weight_matrix = (np.random.rand(*wgt_shape) * 10).astype(dtype=np.dtype('int32'))

        activation_matrix[np.random.choice(2, size=act_shape, p=[1 - sparsity, sparsity]).astype('bool')] = 0
        weight_matrix[np.random.choice(2, size=wgt_shape, p=[1 - sparsity, sparsity]).astype('bool')] = 0

        weight_queue, weight_masks, weight_controls, activation_queue_arr, activation_masks_arr = auto_config_matrices(
            weight=weight_matrix, activation=activation_matrix, ve_num=ve_num, chunk_size=chunk_size,
        )

        ca_unit.run(reset_n=0)
        ca_unit.run(reset_n=1)

        w_m_valid, a_m_valid = 1, [1] * ve_num
        w_d_valid, a_d_valid = 1, [1] * ve_num

        cycle_cnt = 0
        output_act = [[] for _ in range(ve_num)]

        while True:
            ca_unit.run(
                clk=0,
                control=0 if len(weight_controls) == 0 else weight_controls[0],
                w_d_valid=w_d_valid,
                w_m_valid=w_m_valid,
                a_d_valid_arr=a_d_valid,
                a_m_valid_arr=a_m_valid,
                w_d_in=np.zeros(shape=chunk_size) if len(weight_queue) == 0 else weight_queue[0],
                w_m_in=np.zeros(shape=chunk_size) if len(weight_masks) == 0 else weight_masks[0],
                a_d_in_arr=[0 if len(aq) == 0 else aq[0] for aq in activation_queue_arr],
                a_m_in_arr=[0 if len(am) == 0 else am[0] for am in activation_masks_arr],
            )
            ca_unit.run(clk=1)

            for ps_idx, (ps_port, ps_valid) in enumerate(zip(ca_unit.ps_out_arr, ca_unit.ps_valid_arr)):
                if ps_valid.get() == 1:
                    output_act[ps_idx].append(int(ps_port.get_raw()))

            if verbose:
                # sys.stdout.write(f"Iter{cycle_cnt}\n")
                # for vidx, ve_unit in enumerate(ca_unit.ve_array):
                #     sys.stdout.write(
                #         f"- VE{vidx} -> {ve_unit.summary('w_d_in', 'w_m_in', 'a_d_in', 'a_m_in', 'control', 'ps_out')}\n")
                sys.stdout.write(
                    f"cycle: {cycle_cnt}  "
                    f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  "
                    f"valid: {np.array([va.get_raw() for va in ca_unit.ps_valid_arr])}\n")
            else:
                sys.stdout.write(
                    f"\rcycle: {cycle_cnt}  "
                    f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  "
                    f"valid: {np.array([va.get_raw() for va in ca_unit.ps_valid_arr])}")

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

            for vidx in range(ve_num):
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

            if ca_unit.is_idle() and len(weight_masks) == 0:
                break

            cycle_cnt += 1

        orig_result = np.matmul(weight_matrix, activation_matrix)
        test_result = restore_activation_mat(np.array(output_act, dtype='int32').T, out_shape=out_shape)

        if verbose:
            print("\nweight matrix")
            print(weight_matrix)

            print("\nactivation matrix")
            print(activation_matrix)

            print("\noriginal result:")
            print(orig_result)
            print("\nprocessed result")
            print(test_result)
            print(f"\ntest {'passed' if np.array_equal(orig_result, test_result) else 'failed'}")
        else:
            print(f"\tCase #{t + 1}:\ttest {'passed' if np.array_equal(orig_result, test_result) else 'failed'}")