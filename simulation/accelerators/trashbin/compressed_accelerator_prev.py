import os
import sys
import math
import numpy as np

from systempy.elements import InputPort, ClockPort, OutputPort, Register, Wire
from systempy.module import Module
from systempy.keywords import bind


def slicing_array_with_chunks(arr: np.ndarray, chunk_size: int):
    parr = []

    for i in range(0, len(arr), chunk_size):
        if i+chunk_size < len(arr):
            parr.append(arr[i:i+chunk_size])
        else:
            parr.append(np.array(list(arr[i:]) + [0] * (chunk_size - len(arr) + i), dtype=arr.dtype))

    return parr


def restore_activation_mat(parr: np.ndarray, out_shape: tuple):
    ph, pw = parr.shape
    ah, aw = out_shape

    arr = np.zeros(out_shape, dtype=parr.dtype)

    for r_offset, c_offset in zip(range(0, ph, ah), range(0, aw, pw)):
        arr[:, c_offset:c_offset+pw] = parr[r_offset:r_offset+ah, :]

    return arr


def compress_vector(vec, chunk_size):
    """
    This function compresses a vector into nonzero array and bitmask. (ZVC algorithm)
    Compression involves reshaping process which matches the row size with 'chunk size'.
    Therefore, compressed result can directly be assigned to the ProcessingElement.

    For example, if the size of the vector is 32 and chunk size is 8,
    the shape of compressed bitmap will be (4, 8).
    Furthermore, if there are 10 nonzero elements inside the vector,
    the shape of compressed nonzero array will be (2, 8).

    :param vec: vector for compression
    :param chunk_size: size of a chunk
    :return: nonzero array and bitmask
    """

    matrix_flatten = vec[np.nonzero(vec)].flatten()
    mask = (vec != 0).astype('int32')

    vec_compressed = []
    mask_reshaped = []


    for i in range(0, len(matrix_flatten), chunk_size):
        if i + chunk_size <= len(matrix_flatten):
            vec_compressed.append(list(matrix_flatten[i:i + chunk_size]))
        else:
            vec_compressed.append(list(matrix_flatten[i:]) + [0] * (chunk_size - len(matrix_flatten) + i))

    for i in range(0, len(mask), chunk_size):
        if i + chunk_size <= len(mask):
            mask_reshaped.append(list(mask[i:i+chunk_size]))
        else:
            mask_reshaped.append(list(mask[i:]) + [0] * (chunk_size - len(mask) + i))

    # return np.array(vec_compressed, dtype=vec.dtype), mask_reshaped
    return vec_compressed, mask_reshaped


def auto_config_matrices(weight, activation, pe_num, chunk_size, transpose_activation: bool=True):
    """
    This function generates weight and activation queue containing the weight and activation values,
    that will to be sequentially assigned to the accelerator. (CompressedAccelerator)
    Compression involves reshaping process which matches the row size with 'chunk size'.

    :param weight: weight matrix
    :param activation: activation matrix
    :param pe_num: number of VEs
    :param chunk_size: size of a chunk
    :param transpose_activation: indicates whether to transpose the activation
    :return: weight queue, weight mask, weight control, activation queue array, activation mask array
    """
    # Shape of weight and activation
    wh, ww = weight.shape
    ah, aw = activation.shape

    # Transpose activation map if required
    if transpose_activation:
        activation = activation.T

    # Weight mapping
    weight_queue = np.array(list(weight[np.nonzero(weight)].flatten()) * (aw // pe_num), dtype=weight.dtype)
    weight_queue = slicing_array_with_chunks(weight_queue, chunk_size=chunk_size)

    weight_masks = np.array(list((weight != 0).astype('int32').flatten()) * (aw // pe_num), dtype=weight.dtype)
    weight_masks = slicing_array_with_chunks(weight_masks, chunk_size=chunk_size)

    weight_controls = ([0] * (ww // chunk_size - 1) + [1]) * (wh * (aw // pe_num))

    # Activation Mapping
    activation_queue_arr = [[] for _ in range(pe_num)]
    activation_masks_arr = [[] for _ in range(pe_num)]

    for ridx, row in enumerate(activation):
        nz_arr = list(row[np.nonzero(row)].flatten()) * wh
        mask = list((row != 0).astype('int32')) * wh

        activation_queue_arr[ridx % pe_num] += nz_arr
        activation_masks_arr[ridx % pe_num] += mask

    for i in range(pe_num):
        activation_queue_arr[i] = slicing_array_with_chunks(np.array(activation_queue_arr[i]), chunk_size=chunk_size)
        activation_masks_arr[i] = slicing_array_with_chunks(np.array(activation_masks_arr[i]), chunk_size=chunk_size)

    return weight_queue, weight_masks, weight_controls, activation_queue_arr, activation_masks_arr


class ProcessingElement(Module):
    def __init__(self, mult_num: int, chunk_size: int, fifo_capacity: int):
        super(ProcessingElement, self).__init__()

        # Configuration
        self.mult_num = mult_num  # number of multipliers
        self.chunk_size = chunk_size  # size of the vector
        self.fifo_capacity = fifo_capacity  # capacity of the FIFO

        # FIFO
        self.w_fifo   = []  # weight data FIFO
        self.w_m_fifo = []  # weight mask FIFO
        self.a_fifo   = []  # activation data FIFO
        self.a_m_fifo = []  # activation mask FIFO
        self.con_fifo = []  # control queue

        self.w_d_packet = []  # weight data packet FIFO (just for simulation)
        self.w_m_packet = []  # weight mask packet FIFO (just for simulation)
        self.w_c_packet = []  # weight control packet FIFO (just for simulation)

        # Variables
        self.c_mask = None    # variable that stores compared mask
        self.w_prefix = None  # weight prefix sum
        self.a_prefix = None  # activation prefix sum

        # Clock and reset ports
        self.clk = ClockPort()      # clock signal
        self.reset_n = InputPort()  # reset signal

        self.control = InputPort()       # control signal (reset partial sum if 1)
        self.control_out = OutputPort()  # control output signal (toward adjacent VE)

        self.control_out_reg = Register()
        self.control_out.assign(self.control_out_reg)

        # Weight and activation data input ports
        self.w_d_valid = InputPort()  # validity of the weight data
        self.w_m_valid = InputPort()  # validity of the weight mask
        self.a_d_valid = InputPort()  # validity of the activation data
        self.a_m_valid = InputPort()  # validity of the activation mask

        self.w_d_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # weight data input
        self.w_m_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # weight mask input

        self.a_d_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # activation data input
        self.a_m_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # activation mask input

        # Partial sum ports
        self.ps_out = OutputPort()  # partial sum output
        self.ps_valid = OutputPort()

        self.ps_out_reg = Register()
        self.ps_valid_reg = Register()
        self.ps_tmp_reg = Register()

        self.ps_out.assign(self.ps_out_reg)
        self.ps_valid.assign(self.ps_valid_reg)

        # Activation output ports (to SRAM or FIFO buffer)
        self.a_d_in_required = OutputPort()  # indicates if the activation data FIFO has enough space to hold new weight data
        self.a_m_in_required = OutputPort()  # indicates if the activation mask FIFO has enough space to hold new weight mask

        self.a_d_in_required_reg = Register()
        self.a_m_in_required_reg = Register()
        self.a_d_in_required.assign(self.a_d_in_required_reg)
        self.a_m_in_required.assign(self.a_m_in_required_reg)

        # Weight output ports (to/from adjacent VE)
        self.w_d_out = OutputPort(shape=(self.chunk_size, ), dtype='int32')  # weight output port
        self.w_m_out = OutputPort(shape=(self.chunk_size, ), dtype='int32')  # weight metadata output port

        self.w_d_out_required = InputPort()  # indicates if the adjacent VE requires new weight data
        self.w_m_out_required = InputPort()  # indicates if the adjacent VE requires new weight mask

        self.w_d_in_required = OutputPort()  # indicates if the weight data FIFO has enough space to hold new weight data
        self.w_m_in_required = OutputPort()  # indicates if the weight mask FIFO has enough space to hold new weight mask

        self.w_d_out_valid = OutputPort()  # validity of the output weight data
        self.w_m_out_valid = OutputPort()  # validity of the output weight mask packet output

        self.w_d_out_reg = Register(shape=(self.chunk_size, ), dtype='int32')
        self.w_m_out_reg = Register(shape=(self.chunk_size, ), dtype='int32')
        self.w_d_out.assign(self.w_d_out_reg)
        self.w_m_out.assign(self.w_m_out_reg)

        self.w_d_in_required_reg = Register()
        self.w_m_in_required_reg = Register()
        self.w_d_in_required.assign(self.w_d_in_required_reg)
        self.w_m_in_required.assign(self.w_m_in_required_reg)

        self.w_d_out_valid_reg = Register()
        self.w_m_out_valid_reg = Register()
        self.w_d_out_valid.assign(self.w_d_out_valid_reg)
        self.w_m_out_valid.assign(self.w_m_out_valid_reg)


    @Module.always('posedge clk', 'negedge reset_n')
    def main(self):
        if self.reset_n == 0:
            self.w_fifo   = []
            self.w_m_fifo = []
            self.a_fifo   = []
            self.a_m_fifo = []
            self.con_fifo   = []

            self.w_d_packet = []
            self.w_m_packet = []
            self.w_c_packet = []

            self.c_mask = None  # variable that stores compared mask
            self.w_prefix = None  # weight prefix sum
            self.a_prefix = None  # activation prefix sum

            self.control_out_reg <<= 0

            self.ps_out_reg   <<= 0
            self.ps_valid_reg <<= 0
            self.ps_tmp_reg   <<= 0

            self.a_d_in_required_reg <<= 1 if self.is_a_d_in_available() else 0
            self.a_m_in_required_reg <<= 1 if self.is_a_m_in_available() else 0

            self.w_d_out_reg <<= np.zeros(shape=self.chunk_size, dtype='int32')
            self.w_m_out_reg <<= np.zeros(shape=self.chunk_size, dtype='int32')
            self.w_d_in_required_reg <<= 1 if self.is_w_d_in_available() else 0
            self.w_m_in_required_reg <<= 1 if self.is_w_m_in_available() else 0
            self.w_d_out_valid_reg <<= 0
            self.w_m_out_valid_reg <<= 0
        else:
            # Store input weight and activation data
            if self.w_d_valid == 1:
                self.w_fifo += list(self.w_d_in.get_raw())
                self.w_d_packet.append(self.w_d_in.get_raw())
            if self.w_m_valid == 1:
                self.w_m_fifo.append(self.w_m_in.get_raw())
                self.w_m_packet.append(self.w_m_in.get_raw())
                self.w_c_packet.append(self.control.get_raw())
                self.con_fifo.append(self.control.get_raw())
            if self.a_d_valid == 1:
                self.a_fifo += list(self.a_d_in.get_raw())
            if self.a_m_valid == 1:
                self.a_m_fifo.append(self.a_m_in.get_raw())

            # Activation data transfer toward VE
            self.a_d_in_required_reg <<= 1 if self.is_a_d_in_available() else 0
            self.a_m_in_required_reg <<= 1 if self.is_a_m_in_available() else 0

            # Weight data transfer between adjacent VEs
            self.w_m_in_required_reg <<= 1 if self.is_w_m_in_available() else 0
            self.w_d_in_required_reg <<= 1 if self.is_w_d_in_available() else 0

            if self.w_m_out_required == 1 and len(self.w_m_packet) > 0:
                self.w_m_out_reg <<= self.w_m_packet[0]
                self.control_out_reg <<= self.w_c_packet[0]
                self.w_m_out_valid_reg <<= 1
                del self.w_m_packet[0]
                del self.w_c_packet[0]
            else:
                self.w_m_out_valid_reg <<= 0

            if self.w_d_out_required == 1 and len(self.w_d_packet) > 0:
                self.w_d_out_reg <<= self.w_d_packet[0]
                self.w_d_out_valid_reg <<= 1
                del self.w_d_packet[0]
            else:
                self.w_d_out_valid_reg <<= 0

            # If previous output was valid, then next will be initially be invalid
            if self.ps_valid_reg == 1:
                self.ps_valid_reg <<= 0

            # If there are data available and sufficient amount of elements inside the data FIFO
            if not (len(self.w_m_fifo) == 0 or len(self.a_m_fifo) == 0):
                # Generate compared mask and prefix sum of the mask
                w_mask = self.w_m_fifo[0]
                a_mask = self.a_m_fifo[0]

                if len(self.w_fifo) >= np.sum(w_mask) and len(self.a_fifo) >= np.sum(a_mask):
                    if self.c_mask is None or np.count_nonzero(self.c_mask) == 0:
                        self.c_mask = np.logical_and(w_mask, a_mask)
                        self.w_prefix = np.array([np.sum(w_mask[:i]) for i in range(len(w_mask))])
                        self.a_prefix = np.array([np.sum(a_mask[:i]) for i in range(len(a_mask))])

                    # Mapping MAC operation
                    mac_result = self.ps_tmp_reg.get_raw()
                    for _ in range(self.mult_num):  # mapping operands to the multipliers
                        if np.count_nonzero(self.c_mask) != 0:
                            cidx = np.where(self.c_mask == True)[0][0]
                            widx, aidx = self.w_prefix[cidx], self.a_prefix[cidx]

                            wv, av = self.w_fifo[widx], self.a_fifo[aidx]
                            self.c_mask[cidx] = False
                            mac_result += (wv * av)

                    # Define temporary buffer
                    if np.count_nonzero(self.c_mask) == 0 and self.con_fifo[0] == 1:
                        self.ps_tmp_reg <<= 0
                    else:
                        self.ps_tmp_reg <<= mac_result

                    # Remove used operands from the FIFO
                    if np.count_nonzero(self.c_mask) == 0:
                        for _ in range(np.sum(w_mask)):
                            del self.w_fifo[0]
                        del self.w_m_fifo[0]

                        for _ in range(np.sum(a_mask)):
                            del self.a_fifo[0]
                        del self.a_m_fifo[0]

                        # If control signal is 1, then the MAC operation result must be valid
                        if self.con_fifo[0] == 1:
                            self.ps_valid_reg <<= 1
                            self.ps_out_reg <<= mac_result

                        del self.con_fifo[0]
                
    def is_w_d_in_available(self):
        return len(self.w_fifo) <= (self.fifo_capacity - self.chunk_size) # and len(self.w_d_packet) < math.floor(self.fifo_capacity / self.chunk_size)

    def is_a_d_in_available(self):
        return len(self.a_fifo) <= (self.fifo_capacity - self.chunk_size)

    def is_w_m_in_available(self):
        return len(self.w_m_fifo) < math.floor(self.fifo_capacity / self.chunk_size) # and len(self.w_m_packet) < math.floor(self.fifo_capacity / self.chunk_size)

    def is_a_m_in_available(self):
        return len(self.a_m_fifo) < math.floor(self.fifo_capacity / self.chunk_size)

    def is_idle(self):
        return len(self.w_m_fifo) == 0 and len(self.a_m_fifo) == 0


class CompressedAccelerator(Module):
    def __init__(self, pe_num: int, mult_num: int, chunk_size: int, fifo_capacity: int):
        super(CompressedAccelerator, self).__init__()

        # Configuration
        self.pe_num        = pe_num         # number of VEs
        self.chunk_size    = chunk_size     # size of a chunk
        self.fifo_capacity = fifo_capacity  # FIFO capacity

        # Ports
        self.clk = ClockPort()
        self.reset_n = InputPort()
        self.control = InputPort()

        self.w_d_valid = InputPort()  # validity of the weight data (for leftmost VE)
        self.w_m_valid = InputPort()  # validity of the weight mask (for leftmost VE)

        self.a_d_valid_arr = [InputPort() for _ in range(self.pe_num)]
        self.a_m_valid_arr = [InputPort() for _ in range(self.pe_num)]

        self.w_d_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # weight data input (for leftmost VE)
        self.w_m_in = InputPort(shape=(self.chunk_size, ), dtype='int32')  # weight mask input (for leftmost VE)

        self.a_d_in_arr = [InputPort(shape=(self.chunk_size, ), dtype='int32') for _ in range(self.pe_num)]
        self.a_m_in_arr = [InputPort(shape=(self.chunk_size, ), dtype='int32') for _ in range(self.pe_num)]

        self.ps_out_arr = [OutputPort() for _ in range(self.pe_num)]
        self.ps_valid_arr = [OutputPort() for _ in range(self.pe_num)]

        self.a_d_in_required_arr = [OutputPort() for _ in range(self.pe_num)]
        self.a_m_in_required_arr = [OutputPort() for _ in range(self.pe_num)]

        self.w_d_out = OutputPort()
        self.w_m_out = OutputPort()

        self.w_d_out_required = InputPort()
        self.w_m_out_required = InputPort()

        self.w_d_in_required = OutputPort()
        self.w_m_in_required = OutputPort()

        self.w_d_out_valid = OutputPort()
        self.w_m_out_valid = OutputPort()

        # VE array
        self.pe_array = [ProcessingElement(
            mult_num=mult_num, chunk_size=chunk_size, fifo_capacity=fifo_capacity
        ) for _ in range(self.pe_num)]

        for vidx, ve in enumerate(self.pe_array):
            ve.assign(
                clk=self.clk,
                reset_n=self.reset_n,

                control=self.pe_array[vidx-1].control_out if vidx > 0 else self.control,

                w_d_valid=self.pe_array[vidx-1].w_d_out_valid if vidx > 0 else self.w_d_valid,
                w_m_valid=self.pe_array[vidx-1].w_m_out_valid if vidx > 0 else self.w_m_valid,
                a_d_valid=self.a_d_valid_arr[vidx],
                a_m_valid=self.a_m_valid_arr[vidx],

                w_d_in=self.pe_array[vidx-1].w_d_out if vidx > 0 else self.w_d_in,
                w_m_in=self.pe_array[vidx-1].w_m_out if vidx > 0 else self.w_m_in,

                a_d_in=self.a_d_in_arr[vidx],
                a_m_in=self.a_m_in_arr[vidx],

                ps_out=self.ps_out_arr[vidx],
                ps_valid=self.ps_valid_arr[vidx],

                a_d_in_required=self.a_d_in_required_arr[vidx],
                a_m_in_required=self.a_m_in_required_arr[vidx],

                w_d_out=self.pe_array[vidx+1].w_d_in if vidx < self.pe_num-1 else self.w_d_out,
                w_m_out=self.pe_array[vidx+1].w_m_in if vidx < self.pe_num-1 else self.w_m_out,

                w_d_out_required=self.pe_array[vidx+1].w_d_in_required if vidx < self.pe_num-1 else self.w_d_out_required,
                w_m_out_required=self.pe_array[vidx+1].w_m_in_required if vidx < self.pe_num-1 else self.w_m_out_required,

                w_d_in_required=self.pe_array[vidx-1].w_d_out_required if vidx > 0 else self.w_d_in_required,
                w_m_in_required=self.pe_array[vidx-1].w_m_out_required if vidx > 0 else self.w_m_in_required,

                w_d_out_valid=self.pe_array[vidx+1].w_d_valid if vidx < self.pe_num-1 else self.w_d_out_valid,
                w_m_out_valid=self.pe_array[vidx+1].w_m_valid if vidx < self.pe_num-1 else self.w_m_out_valid,
            )

    def is_idle(self):
        return np.count_nonzero(np.logical_not(np.array([len(ve.w_m_fifo) == 0 and len(ve.a_m_fifo) == 0 for ve in self.pe_array]))) == 0


# Testbench for CompressedAccelerator
if __name__ == '__main__':
    from simulation.accelerators.systolic_array_only_cycles import systolic_array_cycles_ws

    # Verbose options
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    # Simulation setup
    act_shape = (32, 64)  # shape of activation matrix
    wgt_shape = (64, 32)  # shape of weight matrix
    out_shape = (wgt_shape[0], act_shape[1])  # shape of output matrix

    # Systolic array cycles
    print(f"systolic array cycles: {systolic_array_cycles_ws(arr_shape=(8, 8), wgt_shape=wgt_shape, act_shape=act_shape)}")

    pe_num = 32        # number of VEs
    mult_num = 2       # number of multipliers per PE
    chunk_size = 4     # size of a chunk
    fifo_capacity = 8  # capacity of FIFO inside the PE
    sparsity = 0.7     # sparsity of the input vectors (0 to 1)

    testcase = 5
    verbose = False

    # Instantiation of CompressedAccelerator
    ca_unit = CompressedAccelerator(
        pe_num=pe_num, mult_num=mult_num, chunk_size=chunk_size, fifo_capacity=fifo_capacity).compile(verbose=verbose)
    ca_unit.run(reset_n=1)

    for t in range(testcase):
        # Generate matrices and queues
        activation_matrix = (np.random.rand(*act_shape) * 10).astype(dtype=np.dtype('int32'))
        weight_matrix = (np.random.rand(*wgt_shape) * 10).astype(dtype=np.dtype('int32'))

        activation_matrix[np.random.choice(2, size=act_shape, p=[1 - sparsity, sparsity]).astype('bool')] = 0
        weight_matrix[np.random.choice(2, size=wgt_shape, p=[1 - sparsity, sparsity]).astype('bool')] = 0

        weight_queue, weight_masks, weight_controls, activation_queue_arr, activation_masks_arr = auto_config_matrices(
            weight=weight_matrix, activation=activation_matrix, pe_num=pe_num, chunk_size=chunk_size,
        )

        ca_unit.run(reset_n=0)
        ca_unit.run(reset_n=1)

        w_m_valid, a_m_valid = 1, [1] * pe_num
        w_d_valid, a_d_valid = 1, [1] * pe_num

        cycle_cnt = 0
        output_act = [[] for _ in range(pe_num)]

        while True:
            ca_unit.run(
                clk=0,
                control=0 if len(weight_controls) == 0 else weight_controls[0],
                w_d_valid=w_d_valid,
                w_m_valid=w_m_valid,
                a_d_valid_arr=a_d_valid,
                a_m_valid_arr=a_m_valid,
                w_d_in=np.zeros(shape=chunk_size) if len(weight_queue) == 0 else weight_queue[0],
                w_m_in=0 if len(weight_masks) == 0 else weight_masks[0],
                a_d_in_arr=[np.zeros(shape=chunk_size) if len(aq) == 0 else aq[0] for aq in activation_queue_arr],
                a_m_in_arr=[0 if len(am) == 0 else am[0] for am in activation_masks_arr],
            )
            ca_unit.run(clk=1)

            for ps_idx, (ps_port, ps_valid) in enumerate(zip(ca_unit.ps_out_arr, ca_unit.ps_valid_arr)):
                if ps_valid.get() == 1:
                    output_act[ps_idx].append(int(ps_port.get_raw()))

            if verbose:
                # sys.stdout.write(f"Iter{cycle_cnt}\n")
                # for vidx, ve_unit in enumerate(ca_unit.pe_array):
                #     sys.stdout.write(
                #         f"- VE{vidx} -> {ve_unit.summary('w_d_in', 'w_m_in', 'a_d_in', 'a_m_in', 'control', 'ps_out')}\n")
                sys.stdout.write(
                    f"cycle: {cycle_cnt}  "
                    f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  "
                    f"valid: {np.array([va.get_raw() for va in ca_unit.ps_valid_arr])}\n")
            else:
                # sys.stdout.write(
                #     f"\rcycle: {cycle_cnt}  "
                #     f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}  "
                #     f"valid: {np.array([va.get_raw() for va in ca_unit.ps_valid_arr])}")
                sys.stdout.write(
                    f"\rcycle: {cycle_cnt}  "
                    f"ps_out: {np.array([ps.get_raw() for ps in ca_unit.ps_out_arr])}")

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

            for vidx in range(pe_num):
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


# # Testbench for ProcessingElement
# if __name__ == '__main__':
#     act_shape = 32  # shape of activation vector
#     wgt_shape = 32  # shape of weight vector
#     chunk_size = 4  # size of a chunk
#     sparsity = 0.2  # sparsity of the input vectors (0 to 1)
#
#     testcase = 1
#     verbose = True
#
#     ve_unit = ProcessingElement(chunk_size=chunk_size, fifo_capacity=8).compile(verbose=verbose)
#     ve_unit.run(reset_n=1)
#
#     for t in range(testcase):
#         ve_unit.run(reset_n=0)
#         ve_unit.run(reset_n=1)
#
#         activation_vector = (np.random.rand(act_shape) * 10).astype(dtype=np.dtype('int32'))
#         weight_vector = (np.random.rand(wgt_shape) * 10).astype(dtype=np.dtype('int32'))
#
#         if sparsity != 0:
#             activation_vector[np.random.choice(act_shape, int(sparsity * act_shape), replace=False)] = 0
#             weight_vector[np.random.choice(wgt_shape, int(sparsity * wgt_shape), replace=False)] = 0
#
#         a_compr, a_masks = compress_vector(activation_vector, chunk_size=chunk_size)
#         w_compr, w_masks = compress_vector(weight_vector, chunk_size=chunk_size)
#
#         # control = 0
#         w_m_valid, a_m_valid = 1, 1
#         w_d_valid, a_d_valid = 1, 1
#
#         while True:
#             ve_unit.run(
#                 clk=0,
#                 control=1 if len(w_compr) == 1 else 0,
#                 w_m_valid=w_m_valid,
#                 a_m_valid=a_m_valid,
#                 w_d_valid=w_d_valid,
#                 a_d_valid=a_d_valid,
#                 w_d_in=np.zeros(shape=chunk_size) if len(w_compr) == 0 else w_compr[0],
#                 w_m_in=np.zeros(shape=chunk_size) if len(w_masks) == 0 else w_masks[0],
#                 a_d_in=np.zeros(shape=chunk_size) if len(a_compr) == 0 else a_compr[0],
#                 a_m_in=np.zeros(shape=chunk_size) if len(a_masks) == 0 else a_masks[0],
#             )
#             ve_unit.run(clk=1)
#
#             control = 0
#
#             if verbose:
#                 ve_unit.print_summary('w_d_in', 'w_m_in', 'a_d_in', 'a_m_in', 'control', 'ps_out')
#                 # print(len(ve_unit.w_m_fifo), len(ve_unit.a_m_fifo))
#
#             if ve_unit.w_d_in_required == 1 and len(w_compr):
#                 del w_compr[0]
#                 w_d_valid = 1
#             else:
#                 w_d_valid = 0
#
#             if ve_unit.w_m_in_required == 1 and len(w_masks):
#                 del w_masks[0]
#                 w_m_valid = 1
#             else:
#                 w_m_valid = 0
#
#             if ve_unit.a_d_in_required == 1 and len(a_compr):
#                 del a_compr[0]
#                 a_d_valid = 1
#             else:
#                 a_d_valid = 0
#
#             if ve_unit.a_m_in_required == 1 and len(a_masks):
#                 del a_masks[0]
#                 a_m_valid = 1
#             else:
#                 a_m_valid = 0
#
#             # if ve_unit.is_idle() and len(w_masks) == 0 and len(a_masks) == 0:
#             #     break
#             if ve_unit.ps_valid == 1:
#                 break
#
#         orig_result = np.sum(activation_vector * weight_vector)
#         test_result = ve_unit.ps_out.get_raw()
#
#         if verbose:
#             print(f"\nactivation: {activation_vector}")
#             print(f"weight:     {weight_vector}")
#             print(f"\noriginal result: {orig_result}")
#             print(f"test result:     {test_result}")
#             print(f"test {'passed' if test_result == orig_result else 'failed'}\n")
#         else:
#             print(f"Case #{t+1}:\ttest {'passed' if test_result == orig_result else 'failed'}")