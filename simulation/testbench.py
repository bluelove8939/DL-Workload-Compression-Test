import torch

from modules import SimulatorModule


class Testbench(object):
    def __init__(self, model: torch.nn.Module, simod: Module):
        self.model = model
        self.simod = simod