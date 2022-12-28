import abc
import torch

from models.tools.layer_info import ConvLayerInfo, LinearLayerInfo, LayerInfo


class Sim(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_hook(self, model_name, submodule_name, submodule_info) -> callable:
        pass

    def register_model(self, model: torch.nn.Module, model_name: str,
                       testbench_filter: callable=lambda x, y: True) -> None:

        layer_info_dict = LayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if sm_name in layer_info_dict.keys() and testbench_filter(model_name, sm_name):
                sm.register_forward_hook(
                    self.generate_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))


class ConvSim(Sim, metaclass=abc.ABCMeta):
    def __init__(self):
        super(ConvSim, self).__init__()

    def register_model(self, model: torch.nn.Module, model_name: str,
                       testbench_filter: callable=lambda x, y: True) -> None:

        layer_info_dict = ConvLayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if sm_name in layer_info_dict.keys() and testbench_filter(model_name, sm_name):
                sm.register_forward_hook(
                    self.generate_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))


class LinearSim(Sim, metaclass=abc.ABCMeta):
    def __init__(self):
        super(LinearSim, self).__init__()

    def register_model(self, model: torch.nn.Module, model_name: str,
                       testbench_filter: callable=lambda x, y: True) -> None:

        layer_info_dict = LinearLayerInfo.generate_from_model(model=model, input_shape=(1, 3, 224, 224))

        for sm_name, sm in model.named_modules():
            if testbench_filter(model_name, sm_name):
                sm.register_forward_hook(
                    self.generate_hook(
                        model_name=model_name, submodule_name=sm_name, submodule_info=layer_info_dict[sm_name]))