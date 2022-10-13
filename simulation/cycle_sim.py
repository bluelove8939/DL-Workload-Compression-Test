import abc
import math

import torch
import numpy as np
from typing import Callable, List

from models.tools.lowering import ifm_lowering, weight_lowering, ConvLayerInfo
from models.tools.progressbar import progressbar


class AcceleratorConfig(object):
    def __init__(self, ve_num: int=128, vector_size: int=64, fifo_capacity: int=128,
                 fetch_cycle: int=1, index_cycle: int=1, mac_cycle: int=1):
        # Parameters of Accelerator
        self.ve_num = ve_num  # number of vector engines

        # Parameters of Data Stremer
        self.vector_size = vector_size      # number of elements within vector tile
        self.fifo_capacity = fifo_capacity  # capacity of input and weight FIFO in terms of element number

        self.fetch_cycle = fetch_cycle  # cycles spent on fetching data from buffers
        self.index_cycle = index_cycle  # cycles spent on generating nonzero indices

        # Parameters of MAC Unit
        self.mac_cycle   = mac_cycle  # cycles spent on mac operation


class _SimModule(metaclass=abc.ABCMeta):
    def __init__(self):
        super(_SimModule, self).__init__()

    @abc.abstractmethod
    def trigger(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class DataStreamer(_SimModule):
    IDLE_STATE  = 0
    INDEX_STATE = 2
    PUSH_STATE  = 3

    def __init__(self, config: AcceleratorConfig):
        super(DataStreamer, self).__init__()

        self.vector_size = config.vector_size
        self.fifo_capacity = config.fifo_capacity

        self.fetch_cycle = config.fetch_cycle
        self.index_cycle = config.index_cycle

        self.op_fifo = 0  # number of elements inside the operand FIFO

        self.valid_op_num = None  # number of valid operands pairs

        self.input_buffer: List[torch.tensor] or None  = None  # input buffer containing row vectors
        self.weight_buffer: List[torch.tensor] or None = None  # weight buffer containing row vectors
        self.input_idx  = 0  # index of current input vector
        self.weight_idx = 0  # index of current weight vector
        self.input_cursor  = 0  # current cursor of the input buffer
        self.weight_cursor = 0  # currsne cursor of the weight buffer

        self.paused = 0  # paused cycles
        self.state = DataStreamer.IDLE_STATE  # state of the machine

    def trigger(self):
        if self.state == DataStreamer.IDLE_STATE:
            if self.paused == 0:
                if self.input_buffer is None or self.weight_buffer is None:
                    return
                if self.done():
                    return

                input_vector = self.input_buffer[self.input_idx]
                weight_vector = self.weight_buffer[self.weight_idx]

                ivec = input_vector[self.input_cursor:min(self.input_cursor+self.vector_size, input_vector.shape[0])]
                wvec = weight_vector[self.weight_cursor:min(self.weight_cursor+self.vector_size, weight_vector.shape[0])]

                self.input_cursor += ivec.shape[0]
                self.weight_cursor += wvec.shape[0]

                imask = (ivec != 0).cpu()
                wmask = (wvec != 0).cpu()

                self.valid_op_num = torch.count_nonzero(torch.logical_and(imask, wmask))

            self.paused += 1

            if self.paused >= self.fetch_cycle:
                if self.valid_op_num == 0:
                    if self.input_cursor >= self.input_buffer[self.input_idx].shape[0] or \
                            self.weight_cursor >= self.weight_buffer[self.weight_idx].shape[0]:

                        self.weight_idx += 1
                        if self.weight_idx == len(self.weight_buffer):
                            self.input_idx += 1
                            if self.input_idx < len(self.input_buffer):
                                self.weight_idx = 0

                        self.input_cursor = 0
                        self.weight_cursor = 0

                    self.state = DataStreamer.IDLE_STATE
                else:
                    self.state = DataStreamer.INDEX_STATE
                self.paused = 0

        elif self.state == DataStreamer.INDEX_STATE:
            self.paused += 1

            if self.paused >= self.index_cycle:
                self.state = DataStreamer.PUSH_STATE
                self.paused = 0

        elif self.state == DataStreamer.PUSH_STATE:
            if self.valid_op_num > 0:
                self.op_fifo += 1
                self.valid_op_num -= 1

            if self.valid_op_num <= 0:
                if self.input_cursor >= self.input_buffer[self.input_idx].shape[0] or \
                        self.weight_cursor >= self.weight_buffer[self.weight_idx].shape[0]:

                    self.weight_idx += 1
                    if self.weight_idx == len(self.weight_buffer):
                        self.input_idx += 1
                        if self.input_idx < len(self.input_buffer):
                            self.weight_idx = 0

                    self.input_cursor = 0
                    self.weight_cursor = 0

                self.state = DataStreamer.IDLE_STATE

    def reset(self):
        self.valid_op_num = None

        self.input_buffer  = None
        self.weight_buffer = None
        self.input_idx     = 0
        self.weight_idx    = 0
        self.input_cursor  = 0
        self.weight_cursor = 0

        self.paused = 0
        self.state = DataStreamer.IDLE_STATE

    def buffer_empty(self) -> bool:
        return self.input_buffer is None and self.weight_buffer is None

    def fifo_empty(self) -> bool:
        return self.op_fifo <= 0

    def done(self):
        return self.input_buffer is None or (self.input_idx >= len(self.input_buffer) and self.weight_idx >= len(self.weight_buffer))

    def set_buffers(self, input_buffer, weight_buffer):
        self.input_buffer = input_buffer
        self.weight_buffer = weight_buffer
        self.input_idx = 0
        self.weight_idx = 0
        self.input_cursor = 0
        self.weight_cursor = 0


class MACUnit(_SimModule):
    IDLE_STATE = 0
    CALC_STATE = 1

    def __init__(self, config: AcceleratorConfig, data_streamer: DataStreamer or None=None):
        super(MACUnit, self).__init__()

        self.mac_cycle: int = config.mac_cycle
        self.data_streamer: DataStreamer = data_streamer

        self.paused = 0
        self.state = MACUnit.IDLE_STATE

    def register_data_streamer(self, data_streamer: DataStreamer or None=None):
        self.data_streamer = data_streamer

    def trigger(self):
        if self.state == MACUnit.IDLE_STATE:
            if not self.data_streamer.fifo_empty():
                self.state = MACUnit.CALC_STATE

        elif self.state == MACUnit.CALC_STATE:
            self.paused += 1
            self.data_streamer.op_fifo -= 1

            if self.paused >= self.mac_cycle:
                if self.data_streamer.fifo_empty():
                    self.state = MACUnit.IDLE_STATE
                else:
                    self.state = MACUnit.CALC_STATE

                self.paused = 0

    def reset(self):
        self.paused = 0
        self.state = MACUnit.IDLE_STATE


class AcceleratorCycleSim(_SimModule):
    def __init__(self, config: AcceleratorConfig, quant : bool=True, device='cpu',
                 verbose: bool=True, verbose_step: int=1000):

        super(AcceleratorCycleSim, self).__init__()

        self.config = config
        self.quant  = quant
        self.device = device

        self.verbose = verbose
        self.verbose_step = verbose_step

        self.data_streamers = []
        self.mac_units = []

        # self.input_mapping = []
        # self.weight_mapping = []
        self.done_mapping = []
        self.cycle = 0

        self.results = {}

        for _ in range(self.config.ve_num):
            ds = DataStreamer(config=self.config)
            mac = MACUnit(config=self.config, data_streamer=ds)

            self.data_streamers.append(ds)
            self.mac_units.append(mac)

            # self.input_mapping.append(0)
            # self.weight_mapping.append(0)
            self.done_mapping.append(False)

        self.input_mat: torch.tensor or None = None
        self.weight_mat: torch.tensor or None = None

    def trigger(self):
        for vidx, (ds, mac) in enumerate(zip(self.data_streamers, self.mac_units)):
            if not self.done_mapping[vidx]:
                ds.trigger()
                mac.trigger()

                if ds.done():
                    self.done_mapping[vidx] = True

        self.cycle += 1

    def reset(self):
        for ds, mac in zip(self.data_streamers, self.mac_units):
            ds.reset()
            mac.reset()

        self.done_mapping = [False] * self.config.ve_num
        self.cycle = 0

        self.input_mat = None
        self.weight_mat = None

    def finished(self) -> bool:
        return self.done_mapping == ([True] * self.config.ve_num)

    def run_test(self, input_mat: torch.tensor, weight_mat: torch.tensor):
        self.input_mat = input_mat
        self.weight_mat = weight_mat

        viter = 0
        ih, iw = input_mat.shape
        wh, ww = weight_mat.shape
        total_mapping = ih*wh

        weight_buffer = [wvec for wvec in self.weight_mat]

        for didx, ds in enumerate(self.data_streamers):
            input_buffer = []
            for iidx, ivec in enumerate(self.input_mat):
                if iidx % self.config.ve_num == didx:
                    input_buffer.append(ivec)

            if len(input_buffer) == 0:
                ds.set_buffers(input_buffer=None, weight_buffer=None)
            else:
                ds.set_buffers(input_buffer=input_buffer, weight_buffer=weight_buffer)

        while not self.finished():
            if self.verbose:
                if viter % self.verbose_step == 0:
                    current_mapping = 0
                    for ds in self.data_streamers:
                        current_mapping += min(ds.input_idx, ih) * wh + min(ds.weight_idx, wh)
                    print(f"\r{progressbar(status=current_mapping, total=total_mapping, scale=50)}"
                          f"{math.ceil(current_mapping / total_mapping * 100):3d}%  "
                          f"cycle: {self.cycle}", end='')
                viter += 1

            self.trigger()

    def register_model(self, model: torch.nn.Module, model_name: str='default',
                       layer_filter: Callable=lambda model_name, layer_name: True):

        model_name = type(model).__name__ if model_name == 'default' else model_name
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device=self.device)

        def generate_hook(model_name: str, layer_name : str) -> Callable:
            def accelerator_cycle_sim_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                # Generation of lowered input feature map and weight data
                print(f"Generating lowered data with layer: {layer_name}", end='\n' if self.verbose else '')

                result_key = (model_name, layer_name)

                ifm = input_tensor[0]
                weight = model.state_dict()[layer_name + '.weight']

                if self.quant:
                    ifm = ifm.int_repr()
                    weight = weight.int_repr()

                lowered_ifm = ifm_lowering(ifm=ifm, layer_info=layer_info[layer_name]).detach()
                lowered_weight = weight_lowering(weight=weight, layer_info=layer_info[layer_name]).detach()

                ih, iw = lowered_ifm.shape
                wh, ww = lowered_weight.shape
                total = math.ceil(ih / self.config.ve_num) * wh * iw

                # Run cycle test
                print(f"\rRunning cycle-accurate test with:   {layer_name:30s}  "
                      f"total: {total:7d}  "
                      f"input: {lowered_ifm.shape}\t"
                      f"weight: {lowered_weight.shape}  ", end='\n' if self.verbose else '')

                self.reset()
                self.run_test(input_mat=lowered_ifm, weight_mat=lowered_weight)
                self.results[result_key] = (self.cycle, total)

                print(f"\rSimulation finished with layer:     {layer_name:30s}  "
                      f"cycle: {self.cycle:7d}  "
                      f"total: {total:7d}  "
                      f"input: {lowered_ifm.shape}\t"
                      f"weight: {lowered_weight.shape}  ", end='\n\n' if self.verbose else '\n')
            return accelerator_cycle_sim_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                if layer_filter(model_name, layer_name):
                    sublayer.register_forward_hook(generate_hook(model_name, layer_name))

    def get_performance(self):
        return self.results
