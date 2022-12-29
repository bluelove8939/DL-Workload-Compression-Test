import numpy as np

from systempy.elements import InputPort, OutputPort, ClockPort, Register, Wire
from systempy.module import Module


class AsyncFIFO(Module):
    def __init__(self, capacity: int=10):
        super(AsyncFIFO, self).__init__()

        self.capacity = capacity

        # Handshake interface
        self.req = OutputPort()
        self.ack = InputPort()
        self.req_reg = Register()

        self.req.assign(self.req_reg)

        # Data output port
        self.d_out = OutputPort()
        self.d_out_reg = Register()

        self.d_out.assign(self.d_out_reg)

        # Container implementation
        self.cursor = Register()
        self.cursor_nxt = Register()
        self.container = Register(shape=self.capacity, dtype='int32')

    @Module.always('ack')
    def master_main(self):
        if self.ack == 0:
            self.d_out_reg << self.container[self.cursor]
            self.req_reg << 1
            self.cursor_nxt << self.cursor + 1
        else:
            self.req_reg << 0
            self.cursor << self.cursor_nxt

    def initialize(self, arr: list):
        for idx, val in enumerate(arr):
            self.container[idx].value = val


class PrefetchBuffer(Module):
    def __init__(self, capacity: int=16, threshold: int=8):
        super(PrefetchBuffer, self).__init__()

        self.capacity = capacity
        self.threshold = threshold

        if self.threshold > self.capacity:
            raise Exception(f"Threshold value ({self.threshold}) needs to be smaller than the capacity ({self.capacity})")

        self.container = Array(node_type=Register, initial_value=0, shape=self.capacity)

        # Clock and reset signal
        self.clk = ClockPort()
        self.reset_n = InputPort()

        # Data I/O channels
        self.w_d_in = InputPort()
        self.r_d_out = OutputPort()
        self.p_d_out = OutputPort()

        self.r_d_out_reg = Register()
        self.p_d_out_reg = Register()

        self.r_d_out.assign(self.r_d_out_reg)
        self.p_d_out.assign(self.p_d_out_reg)

        # Prefetch interface (as master)
        self.req_prefetch = OutputPort()
        self.ack_prefetch = InputPort()
        self.req_prefetch_reg = Register()

        self.req_prefetch.assign(self.req_prefetch_reg)

        # Read interface (as master)
        self.read_addr = InputPort()
        # self.read_abs_addr = Register(initial_value=0)  # abs_addr = addr + offset(pointer)

        self.req_read = OutputPort()
        self.ack_read = InputPort()
        self.req_read_reg = Register()

        self.req_read.assign(self.req_read_reg)

        # Write interface (as slave)
        self.req_write = InputPort()
        self.ack_write = OutputPort()
        self.ack_write_reg = Register()

        self.ack_write.assign(self.ack_write_reg)

        # Memory interface
        self.full = OutputPort()    # indicates whether the queue is full
        self.empty = OutputPort()   # indicates whether the queue is empty
        self.plenty = OutputPort()  # indicates whether there are plenty number of elements to process a sparce vector

        self.head_ptr           = Register()  # write pointer
        self.head_ptr_nxt       = Register()
        self.head_ptr_phase     = Register()
        self.head_ptr_phase_nxt = Register()

        self.tail_ptr           = Register()  # read pointer
        self.tail_ptr_nxt       = Register()
        self.tail_ptr_phase     = Register()
        self.tail_ptr_phase_nxt = Register()
        self.tail_redefine_trig = InputPort()  # trigger signal to redefine tail pointer
        self.current_vec_siz    = InputPort()  # current vector size obtained from PrefixSumUnit

        self.pref_ptr           = Register()  # prefetch pointer
        self.pref_ptr_nxt       = Register()
        self.pref_ptr_phase     = Register()
        self.pref_ptr_phase_nxt = Register()

    def initialize(self, arr: list):
        for idx, val in enumerate(arr):
            self.container[idx].value = val
            self.head_ptr << self.head_ptr_nxt
            self.head_ptr_phase <<= self.head_ptr_phase_nxt

    @Module.always('head_ptr', 'tail_ptr', 'pref_ptr', 'head_ptr_phase', 'tail_ptr_phase', 'pref_ptr_phase')
    def full_empty_decision(self):
        h, t, p = self.head_ptr.get(), self.tail_ptr.get(), self.pref_ptr.get()
        hp, tp, pp = self.head_ptr_phase.get(), self.tail_ptr_phase.get(), self.pref_ptr_phase.get()
        head_tail_distance = (h - t) if (hp == tp) else (self.capacity + (h - t))
        head_pref_distance = (h - p) if (hp == pp) else (self.capacity + (h - p))

        if hp != tp and h == t:
            self.plenty << 1
        elif hp == tp and h == t:
            self.plenty << 0
        else:
            self.plenty << (1 if (self.threshold <= head_tail_distance) else 0)

        if head_tail_distance > head_pref_distance:
            if hp != tp and h == t:
                self.full   << 1
                self.empty  << 0
            elif hp == tp and h == t:
                self.full   << 0
                self.empty  << 1
            else:
                self.full   << 0
                self.empty  << 0
        else:
            if hp != pp and h == p:
                self.full   << 1
                self.empty  << 0
            elif hp == pp and h == p:
                self.full   << 0
                self.empty  << 1
            else:
                self.full   << 0
                self.empty  << 0

    @Module.always('full', 'req_write')
    def write_request_gen(self):
        if self.full == 0 and self.req_write == 0:
            self.ack_write << 0

    @Module.always('head_ptr', 'head_ptr_phase')
    def head_ptr_decision(self):
        if self.head_ptr == self.capacity-1:
            self.head_ptr_nxt       << 0
            self.head_ptr_phase_nxt << (0 if (self.head_ptr_phase_nxt == 1) else 1)
        else:
            self.head_ptr_nxt       << self.head_ptr + 1
            self.head_ptr_phase_nxt << self.head_ptr_phase

    @Module.always('tail_ptr', 'tail_ptr_phase', 'current_vec_siz')
    def tail_ptr_decision(self):
        if self.tail_ptr + self.current_vec_siz > self.capacity-1:
            self.tail_ptr_nxt       << self.tail_ptr + self.current_vec_siz - self.capacity
            self.tail_ptr_phase_nxt << (0 if (self.tail_ptr_phase_nxt == 1) else 1)
        else:
            self.tail_ptr_nxt       << self.tail_ptr + self.current_vec_siz
            self.tail_ptr_phase_nxt << self.tail_ptr_phase

    @Module.always('pref_ptr', 'pref_ptr_phase')
    def pref_ptr_decision(self):
        if self.pref_ptr == self.capacity - 1:
            self.pref_ptr_nxt       << 0
            self.pref_ptr_phase_nxt << (0 if (self.pref_ptr_phase_nxt == 1) else 1)
        else:
            self.pref_ptr_nxt       << self.pref_ptr + 1
            self.pref_ptr_phase_nxt << self.pref_ptr_phase

    @Module.always('posedge clk', 'negedge reset_n')
    def write_main(self):
        if self.reset_n == 0:
            for i in range(self.capacity):
                self.container.value[i] <<= 0

            self.head_ptr <<= 0
            self.head_ptr_phase <<= 0
        else:
            if (self.full == 0) and (self.ack_write == 0 and self.req_write == 1):
                self.container[self.head_ptr].cc_set(self.w_d_in)
                self.head_ptr       <<= self.head_ptr_nxt
                self.head_ptr_phase <<= self.head_ptr_phase_nxt
                self.ack_write_reg  <<= 1

    @Module.always('posedge clk', 'negedge reset_n')
    def read_main(self):
        if self.reset_n == 0:
            self.tail_ptr <<= 0
            self.tail_ptr_phase <<= 0
        else:
            if (self.empty == 0) and (self.ack_read == 0):
                addr = self.tail_ptr.value + self.read_addr.value
                if addr > self.capacity-1:
                    addr -= self.capacity

                self.r_d_out_reg <<= self.container[addr]
                if self.tail_redefine_trig == 1:
                    self.tail_ptr       <<= self.tail_ptr_nxt
                    self.tail_ptr_phase <<= self.tail_ptr_phase_nxt
                self.req_read_reg <<= 1

    @Module.always('posedge clk', 'negedge reset_n')
    def prefetch_main(self):
        if self.reset_n == 0:
            self.pref_ptr       <<= 0
            self.pref_ptr_phase <<= 0
        else:
            if (self.empty == 0) and (self.ack_prefetch == 0):
                self.p_d_out_reg      <<= self.container[self.pref_ptr]
                self.pref_ptr         <<= self.pref_ptr_nxt
                self.pref_ptr_phase   <<= self.pref_ptr_phase_nxt
                self.req_prefetch_reg <<= 1

    def size(self):
        h, t, p = self.head_ptr.get(), self.tail_ptr.get(), self.pref_ptr.get()
        hp, tp, pp = self.head_ptr_phase.get(), self.tail_ptr_phase.get(), self.pref_ptr_phase.get()
        head_tail_distance = (h - t) if (hp == tp) else (self.capacity + (h - t))
        head_pref_distance = (h - p) if (hp == pp) else (self.capacity + (h - p))

        return max(head_tail_distance, head_pref_distance)


class MetadataPrefetchBuffer(Module):
    def __init__(self, capacity: int=4, bitwidth: int=8):
        super(MetadataPrefetchBuffer, self).__init__()

        self.capacity = capacity
        self.bitwidth = bitwidth

        self.initial_metadata = np.zeros(shape=self.bitwidth, dtype='int32')

        self.container = Array(
            node_type=Register, initial_value=self.initial_metadata, shape=self.capacity)

        # Clock and reset signal
        self.clk = ClockPort()
        self.reset_n = InputPort()

        # Data I/O channels
        self.w_d_in  = InputPort(shape=self.bitwidth, dtype='int32')
        self.r_d_out = OutputPort(shape=self.bitwidth, dtype='int32')
        self.p_d_out = OutputPort(shape=self.bitwidth, dtype='int32')

        self.r_d_out_reg = Register(shape=self.bitwidth, dtype='int32')
        self.p_d_out_reg = Register(shape=self.bitwidth, dtype='int32')

        self.r_d_out.assign(self.r_d_out_reg)
        self.p_d_out.assign(self.p_d_out_reg)

        # Prefetch interface (as master)
        self.req_prefetch = OutputPort()
        self.ack_prefetch = InputPort()
        self.req_prefetch_reg = Register()

        self.req_prefetch.assign(self.req_prefetch_reg)

        # Write interface (as slave)
        self.req_write = InputPort()
        self.ack_write = OutputPort()
        self.ack_write_reg = Register()

        self.ack_write.assign(self.ack_write_reg)

        # Memory interface
        self.full = OutputPort()    # indicates whether the queue is full
        self.empty = OutputPort()   # indicates whether the queue is empty

        self.head_ptr           = Register()  # write pointer
        self.head_ptr_nxt       = Register()
        self.head_ptr_phase     = Register()
        self.head_ptr_phase_nxt = Register()

        self.tail_ptr           = Register()  # read pointer
        self.tail_ptr_nxt       = Register()
        self.tail_ptr_phase     = Register()
        self.tail_ptr_phase_nxt = Register()
        self.enable_read        = InputPort()  # enable signal of read request (moves tail pointer)

        self.pref_ptr           = Register()  # prefetch pointer
        self.pref_ptr_nxt       = Register()
        self.pref_ptr_phase     = Register()
        self.pref_ptr_phase_nxt = Register()

    def initialize(self, arr: list):
        for idx, val in enumerate(arr):
            self.container[idx].value = val
            self.head_ptr << self.head_ptr_nxt
            self.head_ptr_phase <<= self.head_ptr_phase_nxt

    @Module.always('head_ptr', 'tail_ptr', 'pref_ptr', 'head_ptr_phase', 'tail_ptr_phase', 'pref_ptr_phase')
    def full_empty_decision(self):
        h, t, p = self.head_ptr.get(), self.tail_ptr.get(), self.pref_ptr.get()
        hp, tp, pp = self.head_ptr_phase.get(), self.tail_ptr_phase.get(), self.pref_ptr_phase.get()
        head_tail_distance = (h - t) if (hp == tp) else (self.capacity + (h - t))
        head_pref_distance = (h - p) if (hp == pp) else (self.capacity + (h - p))

        if head_tail_distance > head_pref_distance:
            if hp != tp and h == t:
                self.full   << 1
                self.empty  << 0
            elif hp == tp and h == t:
                self.full   << 0
                self.empty  << 1
            else:
                self.full   << 0
                self.empty  << 0
        else:
            if hp != pp and h == p:
                self.full   << 1
                self.empty  << 0
            elif hp == pp and h == p:
                self.full   << 0
                self.empty  << 1
            else:
                self.full   << 0
                self.empty  << 0

    @Module.always('full', 'req_write')
    def write_request_gen(self):
        if self.full == 0 and self.req_write == 0:
            self.ack_write << 0

    @Module.always('head_ptr', 'head_ptr_phase')
    def head_ptr_decision(self):
        if self.head_ptr == self.capacity-1:
            self.head_ptr_nxt       << 0
            self.head_ptr_phase_nxt << (0 if (self.head_ptr_phase_nxt == 1) else 1)
        else:
            self.head_ptr_nxt       << self.head_ptr + 1
            self.head_ptr_phase_nxt << self.head_ptr_phase

    @Module.always('tail_ptr', 'tail_ptr_phase')
    def tail_ptr_decision(self):
        if self.tail_ptr == self.capacity-1:
            self.tail_ptr_nxt       << 0
            self.tail_ptr_phase_nxt << (0 if (self.tail_ptr_phase_nxt == 1) else 1)
        else:
            self.tail_ptr_nxt       << self.tail_ptr + 1
            self.tail_ptr_phase_nxt << self.tail_ptr_phase


    @Module.always('pref_ptr', 'pref_ptr_phase')
    def pref_ptr_decision(self):
        if self.pref_ptr == self.capacity - 1:
            self.pref_ptr_nxt       << 0
            self.pref_ptr_phase_nxt << (0 if (self.pref_ptr_phase_nxt == 1) else 1)
        else:
            self.pref_ptr_nxt       << self.pref_ptr + 1
            self.pref_ptr_phase_nxt << self.pref_ptr_phase

    @Module.always('posedge clk', 'negedge reset_n')
    def write_main(self):
        if self.reset_n == 0:
            for i in range(self.capacity):
                self.container.value[i] <<= self.initial_metadata

            self.head_ptr       <<= 0
            self.head_ptr_phase <<= 0
            self.ack_write_reg  <<= 1
        else:
            if (self.full == 0) and (self.ack_write == 0 and self.req_write == 1):
                self.container[self.head_ptr].cc_set(self.w_d_in)
                self.head_ptr       <<= self.head_ptr_nxt
                self.head_ptr_phase <<= self.head_ptr_phase_nxt
                self.ack_write_reg  <<= 1

    @Module.always('posedge clk', 'negedge reset_n')
    def read_main(self):
        if self.reset_n == 0:
            self.r_d_out_reg    <<= self.initial_metadata
            self.tail_ptr       <<= 0
            self.tail_ptr_phase <<= 0
        else:
            if (self.empty == 0) and (self.enable_read == 1):
                self.r_d_out_reg <<= self.container[self.tail_ptr]
                self.tail_ptr       <<= self.tail_ptr_nxt
                self.tail_ptr_phase <<= self.tail_ptr_phase_nxt

    @Module.always('posedge clk', 'negedge reset_n')
    def prefetch_main(self):
        if self.reset_n == 0:
            self.p_d_out_reg      <<= self.initial_metadata
            self.pref_ptr         <<= 0
            self.pref_ptr_phase   <<= 0
            self.req_prefetch_reg <<= 0
        else:
            if (self.empty == 0) and (self.ack_prefetch == 0):
                self.p_d_out_reg      <<= self.container[self.pref_ptr]
                self.pref_ptr         <<= self.pref_ptr_nxt
                self.pref_ptr_phase   <<= self.pref_ptr_phase_nxt
                self.req_prefetch_reg <<= 1

    def size(self):
        h, t, p = self.head_ptr.get(), self.tail_ptr.get(), self.pref_ptr.get()
        hp, tp, pp = self.head_ptr_phase.get(), self.tail_ptr_phase.get(), self.pref_ptr_phase.get()
        head_tail_distance = (h - t) if (hp == tp) else (self.capacity + (h - t))
        head_pref_distance = (h - p) if (hp == pp) else (self.capacity + (h - p))

        return max(head_tail_distance, head_pref_distance)


# Testbench for AsyncFIFO
if __name__ == '__main__':
    sb_unit = AsyncFIFO().compile(verbose=True)
    sb_unit.print_module_info()
    sb_unit.initialize([1, 2, 3, 4, 5, 6, 7, 8, 9])

    lim = 4
    collected = []

    while len(collected) < lim and sb_unit.req.get() == 0:
        sb_unit.run(ack=0)
        while sb_unit.req.get() == 0:
            continue
        collected.append(sb_unit.d_out.get())
        sb_unit.print_summary('ack', 'req', 'd_out')
        sb_unit.run(ack=1)

    print(collected)


# # Testbench for PrefetchBuffer
# if __name__ == '__main__':
#     clk_count = 0
#
#     pb_unit = PrefetchBuffer(capacity=16, threshold=8).compile(verbose=True)
#     pb_unit.print_module_info()
#
#     pb_unit.run(reset_n=1)
#     pb_unit.run(reset_n=0)
#     pb_unit.run(reset_n=1)
#
#     # Write test
#     print("- Write test")
#     for i in range(9):
#         while pb_unit.ack_write.get() == 1:
#             pb_unit.run(clk=0)
#             pb_unit.run(clk=1)
#             clk_count += 1
#
#         pb_unit.run(clk=0, req_write=1, w_d_in=i)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         while pb_unit.ack_write.get() == 0:
#             continue
#
#         pb_unit.run(req_write=0)
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'w_d_in', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty', 'plenty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Prefetch test
#     print("\n- Prefetch test")
#     for i in range(5):
#         pb_unit.run(clk=0, ack_prefetch=0)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         while pb_unit.req_prefetch.get() == 0:
#             continue
#
#         pb_unit.run(ack_prefetch=1)
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'p_d_out', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty', 'plenty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Read test
#     print("\n- Read test")
#     for i in range(6):
#         pb_unit.run(clk=0, ack_read=0, read_addr=i)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         while pb_unit.req_read.get() == 0:
#             continue
#
#         pb_unit.run(ack_read=1)
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'r_d_out', 'head_ptr', 'tail_ptr', 'pref_ptr', 'read_addr', 'full', 'empty', 'plenty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Read offset movement test
#     print("\n- Read offset movement test")
#     pb_unit.run(clk=0, ack_read=0, read_addr=5, tail_redefine_trig=1, current_vec_siz=6)
#     pb_unit.run(clk=1)
#     clk_count += 1
#
#     while pb_unit.req_read.get() == 0:
#         continue
#
#     pb_unit.run(ack_read=1)
#     print(f"clk: {clk_count:2d}   "
#           f"{pb_unit.summary('container', 'r_d_out', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty', 'plenty')}   "
#           f"size: {pb_unit.size()}")
#
#     # Additional cycles
#     print("\n- Additional cycles")
#     for _ in range(5):
#         pb_unit.run(clk=0)
#         pb_unit.run(clk=1)
#         clk_count += 1
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty', 'plenty')}   "
#               f"size: {pb_unit.size()}")


# # Testbench for MetadataPrefetchBuffer
# if __name__ == '__main__':
#     clk_count = 0
#
#     pb_unit = MetadataPrefetchBuffer(capacity=4, bitwidth=2).compile(verbose=True)
#     pb_unit.print_module_info()
#
#     pb_unit.run(reset_n=1)
#     pb_unit.run(reset_n=0)
#     pb_unit.run(reset_n=1)
#
#     metadata = [
#         np.array([1, 0]),
#         np.array([0, 0]),
#         np.array([1, 1]),
#         np.array([0, 1]),
#     ]
#
#     # Write test
#     print("- Write test")
#     for i in metadata:
#         while pb_unit.ack_write.get() == 1:
#             pb_unit.run(clk=0)
#             pb_unit.run(clk=1)
#             clk_count += 1
#
#         pb_unit.run(clk=0, req_write=1, w_d_in=i)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         while pb_unit.ack_write.get() == 0:
#             continue
#
#         pb_unit.run(req_write=0)
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'w_d_in', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Prefetch test
#     print("\n- Prefetch test")
#     for i in range(2):
#         pb_unit.run(clk=0, ack_prefetch=0)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         while pb_unit.req_prefetch.get() == 0:
#             continue
#
#         pb_unit.run(ack_prefetch=1)
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'p_d_out', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Read test
#     print("\n- Read test")
#     for i in range(3):
#         pb_unit.run(clk=0, enable_read=1)
#         pb_unit.run(clk=1)
#         clk_count += 1
#
#         pb_unit.run(enable_read=0)
#
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'r_d_out', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty')}   "
#               f"size: {pb_unit.size()}")
#
#     # Additional cycles
#     print("\n- Additional cycles")
#     for _ in range(5):
#         pb_unit.run(clk=0)
#         pb_unit.run(clk=1)
#         clk_count += 1
#         print(f"clk: {clk_count:2d}   "
#               f"{pb_unit.summary('container', 'head_ptr', 'tail_ptr', 'pref_ptr', 'full', 'empty', 'plenty')}   "
#               f"size: {pb_unit.size()}")