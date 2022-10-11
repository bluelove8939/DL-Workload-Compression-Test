import abc
import torch
from typing import Callable

from models.tools.lowering import ifm_lowering, weight_lowering, ConvLayerInfo


class AcceleratorConfig(object):
    def __init__(self, ve_num: int=128, vector_size: int=64, fifo_capacity: int=128,
                 fetch_cycle: int=1, index_cycle: int=1, mac_cycle: int=1):
        # Parameters of Accelerator
        self.ve_num = ve_num  # number of vector engines

        # Parameters of Data Stremer
        self.vector_size = vector_size  # number of elements within vector tile
        self.fifo_capacity = fifo_capacity  # capacity of input and weight FIFO in terms of element number

        self.fetch_cycle = fetch_cycle  # cycles spent on fetching data from buffers
        self.index_cycle = index_cycle  # cycles spent on generating nonzero indices

        # Parameters of MAC Unit
        self.mac_cycle   = mac_cycle    # cycles spent on mac operation


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
    FETCH_STATE = 1
    INDEX_STATE = 2
    PUSH_STATE  = 3

    def __init__(self, config: AcceleratorConfig):
        super(DataStreamer, self).__init__()

        self.vector_size = config.vector_size      # number of elements within vector tile
        self.fifo_capacity = config.fifo_capacity  # capacity of input and weight FIFO in terms of element number

        self.fetch_cycle = config.fetch_cycle  # cycles spent on fetching data from buffers
        self.index_cycle = config.index_cycle  # cycles spent on generating nonzero indices

        self.op_fifo = 0  # number of elements inside the operand FIFO

        self.valid_op_num = None  # number of valid operands pairs

        self.input_buffer: torch.tensor or None = None   # input buffer containing row vector
        self.weight_buffer: torch.tensor or None = None  # weight buffer containing row vector
        self.input_cursor = 0   # current cursor of the input buffer
        self.weight_cursor = 0  # currsne cursor of the weight buffer

        self.paused = 0  # paused cycles
        self.state = DataStreamer.IDLE_STATE  # state of the machine

    def trigger(self):
        if self.state == DataStreamer.IDLE_STATE:
            if self.input_buffer is not None and self.weight_buffer is not None:
                if self.input_cursor < self.input_buffer.shape[0] and self.weight_buffer < self.weight_buffer.shape[0]:
                    self.state = DataStreamer.FETCH_STATE

        elif self.state == DataStreamer.FETCH_STATE:
            if self.paused == 0:
                if self.input_buffer is None or self.weight_buffer is None:
                    assert Exception('Input or weight buffer is empty')
                if self.input_cursor >= self.input_buffer.shape[0] or self.weight_buffer >= self.weight_buffer.shape[0]:
                    self.state = DataStreamer.IDLE_STATE
                    return

                ivec = self.input_buffer[self.input_cursor:min(self.input_cursor+self.vector_size, self.input_buffer.shape[0])]
                wvec = self.weight_buffer[self.weight_cursor:min(self.weight_cursor+self.vector_size, self.weight_buffer.shape[0])]

                self.input_cursor += ivec.shape[0]
                self.weight_cursor += wvec.shape[0]

                imask = ivec != 0
                wmask = wvec != 0

                self.valid_op_num = torch.count_nonzero(torch.logical_and(imask, wmask))

            self.paused += 1

            if self.paused >= self.fetch_cycle:
                if self.valid_op_num == 0:
                    self.state = DataStreamer.FETCH_STATE
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
                self.state = DataStreamer.FETCH_STATE

    def reset(self):
        self.valid_op_num = None

        self.input_buffer: torch.tensor or None = None
        self.weight_buffer: torch.tensor or None = None
        self.input_cursor = 0
        self.weight_cursor = 0

        self.paused = 0
        self.state = DataStreamer.IDLE_STATE


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
            if self.data_streamer.op_fifo > 0:
                self.state = MACUnit.CALC_STATE

        elif self.state == MACUnit.CALC_STATE:
            self.paused += 1

            if self.paused >= self.mac_cycle:
                if self.data_streamer.op_fifo <= 0:
                    self.state = MACUnit.IDLE_STATE
                else:
                    self.state = MACUnit.CALC_STATE

                self.paused = 0

    def reset(self):
        self.paused = 0
        self.state = MACUnit.IDLE_STATE


class AcceleratorCycleSim(_SimModule):
    def __init__(self, config: AcceleratorConfig, quant : bool=True, device='cpu'):
        super(AcceleratorCycleSim, self).__init__()

        self.config = config
        self.quant  = quant
        self.device = device

        self.data_streamers = []
        self.mac_units = []

        self.input_mapping = []
        self.weight_mapping = []
        self.done = []
        self.cycle = 0

        self.results = {}

        for _ in range(self.config.ve_num):
            ds = DataStreamer(config=self.config)
            mac = MACUnit(config=self.config, data_streamer=ds)

            self.data_streamers.append(ds)
            self.mac_units.append(mac)

            self.input_mapping.append(0)
            self.weight_mapping.append(0)
            self.done.append(False)

        self.input_mat: torch.tensor or None = None
        self.weight_mat: torch.tensor or None = None

    def trigger(self):
        for vidx, (ds, mac) in enumerate(zip(self.data_streamers, self.mac_units)):
            if not self.done[vidx]:
                if ds.state == DataStreamer.IDLE_STATE:
                    if self.weight_mapping[vidx] < self.weight_mat.shape[0] - 1:
                        self.weight_mapping[vidx] += 1
                        ds.weight_buffer = self.weight_mat[self.weight_mapping[vidx]]
                    elif self.input_mapping[vidx] * self.config.ve_num + vidx < self.input_mat.shape[0] - 1:
                        self.input_mapping[vidx] += 1
                        ds.input_buffer = self.input_mat[self.input_mapping[vidx] * self.config.ve_num + vidx]
                    else:
                        self.done[vidx] = True

                ds.trigger()
                mac.trigger()

        self.cycle += 1

    def reset(self):
        for ds, mac in zip(self.data_streamers, self.mac_units):
            ds.reset()
            mac.reset()

        self.input_mapping = [0] * self.config.ve_num
        self.weight_mapping = [0] * self.config.ve_num
        self.done = [False] * self.config.ve_num
        self.cycle = 0

        self.input_mat = None
        self.weight_mat = None

    def finished(self) -> bool:
        return self.done == ([True] * self.config.ve_num)

    def run_test(self, input_mat: torch.tensor, weight_mat: torch.tensor):
        self.input_mat = input_mat
        self.weight_mat = weight_mat

        while not self.finished():
            self.trigger()

        print(f'total computation cycles: {self.cycle}')

    def register_model(self, model: torch.nn.Module, model_name: str='default'):
        model_name = type(model).__name__ if model_name == 'default' else model_name
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device=self.device)

        def generate_hook(model_name: str, layer_name : str) -> Callable:
            def accelerator_cycle_sim_hook(layer : torch.nn.Module, input_tensor : torch.Tensor, output_tensor : torch.Tensor):
                # Generation of lowered input feature map and weight data
                print(f"Generating lowered data with layer: {layer_name}", end='')

                result_key = (model_name, layer_name)

                ifm = input_tensor[0]
                weight = model.state_dict()[layer_name + '.weight']

                if self.quant:
                    ifm = ifm.int_repr()
                    weight = weight.int_repr()

                lowered_ifm = ifm_lowering(ifm=ifm, layer_info=layer_info[layer_name]).detach()
                lowered_weight = weight_lowering(weight=weight, layer_info=layer_info[layer_name]).detach()

                self.reset()
                self.run_test(input_mat=lowered_ifm, weight_mat=lowered_weight)

                self.results[result_key] = self.cycle

                print(f"\rSimulation finished with layer: {layer_name}", end='\n')
            return accelerator_cycle_sim_hook

        for layer_name, sublayer in model.named_modules():
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
                sublayer.register_forward_hook(generate_hook(model_name, layer_name))

    def get_performance(self):
        return self.results


# if __name__ == '__main__':
#     from models.model_presets import imagenet_pretrained
#
#     model_config = imagenet_pretrained['AlexNet']
#     model = model_config.generate()
#
#     accelerator_config = AcceleratorConfig(ve_num=128, vector_size=64, fifo_capacity=128,
#                                            fetch_cycle=1, index_cycle=1, mac_cycle=1)
#     sim = AcceleratorCycleSim(config=accelerator_config, quant=False, device='cuda')

