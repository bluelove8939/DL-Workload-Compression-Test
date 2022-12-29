import numpy as np

from systempy.module   import Module
from systempy.elements import InputPort, OutputPort, ClockPort, Register
from systempy.keywords import bind


class ProcessingElement(Module):
    def __init__(self):
        super(ProcessingElement, self).__init__()

        self.clk = ClockPort()
        self.reset_n = InputPort()

        self.control = InputPort()

        self.a_in = InputPort()
        self.d_in = InputPort()

        self.a_out = OutputPort()
        self.d_out = OutputPort()

        self.a_out_reg = Register()
        self.psum_stored = Register()
        self.weight_stored = Register()

        self.a_out.assign(self.a_out_reg)

    @Module.always('control', 'weight_stored', 'psum_stored')
    def output_assign(self):
        if self.control.get() == 1:
            self.d_out << self.weight_stored
        else:
            self.d_out << self.psum_stored

    @Module.always('posedge clk', 'negedge reset_n')
    def weight_psum_calc(self):
        if self.reset_n.get() == 0:
            self.a_out_reg     <<= 0
            self.psum_stored   <<= 0
            self.weight_stored <<= 0
        else:
            if self.control.get() == 0:  # IDLE
                self.a_out_reg     <<= 0
                self.psum_stored   <<= 0
                self.weight_stored <<= 0

            elif self.control.get() == 1:  # WEIGHT INPUT
                self.weight_stored <<= self.d_in

            elif self.control.get() == 2:  # COMPUTATION
                self.a_out_reg   <<= self.a_in
                self.psum_stored <<= (self.weight_stored * self.a_in + self.d_in)

            else:
                self.a_out_reg     <<= 0
                self.psum_stored   <<= 0
                self.weight_stored <<= 0


class SystolicArray(Module):
    def __init__(self, width=8, height=8):
        super(SystolicArray, self).__init__()

        # array config
        self._width = width
        self._height = height

        # port declaration
        self.clk = ClockPort()
        self.reset_n = InputPort()

        self.control = InputPort()

        self.w_in_vec = InputPort(shape=(self._width,), dtype='int32')
        self.a_in_vec = InputPort(shape=(self._width,), dtype='int32')

        self.ps_out_vec = OutputPort(shape=self._width, dtype='int32')

        # array generation
        self.array: list[list[ProcessingElement]] = [[ProcessingElement() for _ in range(self._width)] for _ in range(self._height)]

        for ro_idx in range(self._height):
            for co_idx in range(self._width):
                self.array[ro_idx][co_idx].clk.assign(self.clk)
                self.array[ro_idx][co_idx].reset_n.assign(self.reset_n)
                self.array[ro_idx][co_idx].control.assign(self.control)

        for ro_idx in range(self._height-1):
            for co_idx in range(self._width):
                self.array[ro_idx+1][co_idx].d_in.assign(self.array[ro_idx][co_idx].d_out)

        for co_idx in range(self._width-1):
            for ro_idx in range(self._height):
                self.array[ro_idx][co_idx+1].a_in.assign(self.array[ro_idx][co_idx].a_out)

    @Module.always('w_in_vec', 'a_in_vec', 'control')
    def array_input_set(self):
        for co_idx in range(self._width):
            if self.control.get() == 1:
                self.array[0][co_idx].d_in << self.w_in_vec[co_idx]
            else:
                self.array[0][co_idx].d_in << 0

        for ro_idx in range(self._height):
            self.array[ro_idx][0].a_in << self.a_in_vec[ro_idx]

    @Module.always('array[m][*].d_out')
    def ps_out_set(self):
        self.ps_out_vec << bind(*[self.array[self._height-1][co_idx].d_out for co_idx in range(self._width)], dtype='int32')

    def current_output(self):
        return list(map(lambda x: x.get(), self.ps_out_vec.get()))


def process_weight_mat(arr: np.ndarray):
    return np.array(list(reversed(arr)))

def process_activation_mat(arr: np.ndarray):
    height, width = arr.shape
    processed_arr = np.zeros(shape=(height+width-1, width), dtype=arr.dtype)

    for ridx in range(height):
        for cidx in range(width):
            processed_arr[ridx+cidx][cidx] = arr[ridx][cidx]

    return processed_arr

def restore_activation_mat(processed_arr: np.ndarray):
    pheight, pwidth = processed_arr.shape
    height, width = pheight-pwidth+1, pwidth
    arr = np.zeros(shape=(height, width), dtype=processed_arr.dtype)

    for ridx in range(height):
        for cidx in range(width):
            arr[ridx][cidx] = processed_arr[ridx+cidx][cidx]

    return arr


# Testbench for systolic array
if __name__ == '__main__':
    arr_width, arr_height  = 4, 4      # shape of systolic array
    act_shape = (4, arr_width)          # shape of activation matrix
    wgt_shape = (arr_height, arr_width)  # shape of weight matrix

    sa_unit = SystolicArray(width=arr_width, height=arr_height).compile(verbose=True)

    activation_matrix = (np.random.rand(*act_shape) * 3).astype(dtype=np.dtype('int32'))
    weight_matrix = (np.random.rand(*wgt_shape) * 3).astype(dtype=np.dtype('int32'))

    sa_unit.run(reset_n=0)
    sa_unit.run(reset_n=1)

    for w_in_vec in process_weight_mat(weight_matrix):
        sa_unit.run(clk=0, control=1, w_in_vec=w_in_vec)
        sa_unit.run(clk=1)
        sa_unit.print_summary('w_in_vec', 'a_in_vec', 'ps_out_vec')

    iter_cnt = 0
    output_act = []

    for a_in_vec in process_activation_mat(activation_matrix):
        sa_unit.run(clk=0, control=2, a_in_vec=a_in_vec)
        sa_unit.run(clk=1)
        sa_unit.print_summary('w_in_vec', 'a_in_vec', 'ps_out_vec')
        if iter_cnt >= arr_width-1:
            output_act.append(sa_unit.current_output())
        iter_cnt += 1

    for _ in range(act_shape[1]-1):
        sa_unit.run(clk=0)
        sa_unit.run(clk=1)
        sa_unit.print_summary('w_in_vec', 'a_in_vec', 'ps_out_vec')

        output_act.append(sa_unit.current_output())

    orig_result = np.matmul(activation_matrix, weight_matrix)
    test_result = restore_activation_mat(np.array(output_act))

    print("\noriginal result:")
    print(orig_result)
    print("\nprocessed result")
    print(test_result)
    print(f"\ntest {'passed' if np.array_equal(orig_result, test_result) else 'failed'}")