import abc
import math
import torch
import numpy as np


def is_linear_layer(sublayer: torch.nn.Module):
    return 'linear' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight')

def is_conv_layer(sublayer: torch.nn.Module):
    return 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight')


class LayerInfo(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def is_linear(self):
        return isinstance(self, LinearLayerInfo)

    def is_conv(self):
        return isinstance(self, ConvLayerInfo)

    @classmethod
    def generate_from_model(cls, model: torch.nn.Module, input_shape=(1, 3, 226, 226), device='cpu'):
        dummy_image = torch.tensor(np.zeros(input_shape, dtype=np.dtype('float32'))).to(device)
        result = {}

        def linear_info_generate_hook(layer_name):
            def linear_info_hook(layer, input_tensor, output_tensor):
                N, _ = input_tensor[0].shape
                IF = layer.in_features
                OF = layer.out_features

                result[layer_name] = LinearLayerInfo(N=N, IF=IF, OF=OF)

            return linear_info_hook

        def conv_info_generate_hook(layer_name):
            def conv_info_hook(layer, input_tensor, output_tensor):
                N, Ci, W, H = input_tensor[0].shape
                Co = layer.out_channels
                FW, FH = layer.kernel_size
                S = layer.stride[0]
                P = layer.padding[0]
                OW = math.floor((W - FW + (2 * P)) / S) + 1
                OH = math.floor((H - FH + (2 * P)) / S) + 1
                result[layer_name] = ConvLayerInfo(N, Ci, W, H, Co, FW, FH, S, P, OW, OH)

            return conv_info_hook

        for name, sublayer in model.named_modules():
            if is_linear_layer(sublayer=sublayer):
                sublayer.register_forward_hook(linear_info_generate_hook(name))

            if is_conv_layer(sublayer=sublayer):
                sublayer.register_forward_hook(conv_info_generate_hook(name))

        model.eval()
        model(dummy_image)

        return result


class LinearLayerInfo(LayerInfo):
    def __init__(self,
                 N:  int or None=None, IF: int or None=None, OF: int or None=None):

        super(LinearLayerInfo, self).__init__()

        self.N  = N   # batch size
        self.IF = IF  # number of input features
        self.OF = OF  # number of output features

    @classmethod
    def generate_from_model(cls, model: torch.nn.Module, input_shape=(1, 3, 226, 226), device='cpu') -> dict:
        dummy_image = torch.tensor(np.zeros(input_shape, dtype=np.dtype('float32'))).to(device)
        result = {}

        def linear_info_generate_hook(layer_name):
            def linear_info_hook(layer, input_tensor, output_tensor):
                N, _ = input_tensor[0].shape
                IF = layer.in_features
                OF = layer.out_features

                result[layer_name] = LinearLayerInfo(N=N, IF=IF, OF=OF)

            return linear_info_hook

        for name, sublayer in model.named_modules():
            if is_linear_layer(sublayer=sublayer):
                sublayer.register_forward_hook(linear_info_generate_hook(name))

        model.eval()
        model(dummy_image)

        return result

    @classmethod
    def generate_from_layer(cls, layer: torch.nn.Module, N):
        if 'linear' not in type(layer).__name__.lower():
            raise Exception(f'Invalid layer type: {type(layer).__name__}')

        IF = layer.in_features
        OF = layer.out_features

        return LinearLayerInfo(N=N, IF=IF, OF=OF,)

    def ifm_shape(self) -> tuple:
        return self.N, self.IF

    def weight_shape(self) -> tuple:
        return self.IF, self.OF

    def __str__(self) -> str:
        return '    '.join([f"{attr_name:2s}: {attr_val:4d}" for attr_name, attr_val in self.__dict__.items()])


class ConvLayerInfo(LayerInfo):
    def __init__(self,
                 N  : int or None=None, Ci : int or None=None, W  : int or None=None, H : int or None=None,
                 Co : int or None=None, FW : int or None=None, FH : int or None=None, S : int or None=None,
                 P  : int or None=None, OW : int or None=None, OH : int or None=None):

        super(ConvLayerInfo, self).__init__()

        self.N  = N   # batch size
        self.Ci = Ci  # input channel
        self.W  = W   # image width
        self.H  = H   # image height
        self.Co = Co  # output channel
        self.FW = FW  # filter width
        self.FH = FH  # filter height
        self.S  = S   # stride
        self.P  = P   # padding

        self.OW = OW if OW is not None else math.floor((W - FW + (2 * P)) / S) + 1  # output width
        self.OH = OH if OH is not None else math.floor((H - FH + (2 * P)) / S) + 1  # output height

    @classmethod
    def generate_from_model(cls, model: torch.nn.Module, input_shape=(1, 3, 226, 226), device='cpu') -> dict:
        dummy_image = torch.tensor(np.zeros(input_shape, dtype=np.dtype('float32'))).to(device)
        result = {}

        def generate_hook(layer_name):
            def conv_info_hook(layer, input_tensor, output_tensor):
                N, Ci, W, H = input_tensor[0].shape
                Co = layer.out_channels
                FW, FH = layer.kernel_size
                S = layer.stride[0]
                P = layer.padding[0]
                OW = math.floor((W - FW + (2 * P)) / S) + 1
                OH = math.floor((H - FH + (2 * P)) / S) + 1
                result[layer_name] = ConvLayerInfo(N, Ci, W, H, Co, FW, FH, S, P, OW, OH)
            return conv_info_hook

        for name, sublayer in model.named_modules():
            if is_conv_layer(sublayer=sublayer):
                sublayer.register_forward_hook(generate_hook(name))

        model.eval()
        model(dummy_image)

        return result

    @classmethod
    def generate_from_layer(cls, layer: torch.nn.Module, N, W, H):
        if 'conv' not in type(layer).__name__.lower():
            raise Exception(f'Invalid layer type: {type(layer).__name__}')

        Ci = layer.in_channels
        Co = layer.out_channels
        FW, FH = layer.kernel_size
        S = layer.stride[0]
        P = layer.padding[0]

        return ConvLayerInfo(
            N=N, Ci=Ci, W=W, H=H,
            Co=Co, FW=FW, FH=FH,
            S=S, P=P,
            OW = math.floor((W - FW + (2 * P)) / S) + 1,
            OH = math.floor((H - FH + (2 * P)) / S) + 1,
        )

    def ifm_shape(self) -> tuple:
        return self.N, self.Ci, self.W, self.H

    def weight_shape(self) -> tuple:
        return self.Co, self.Ci, self.FW, self.FH

    def __str__(self) -> str:
        return '    '.join([f"{attr_name:2s}: {attr_val:4d}" for attr_name, attr_val in self.__dict__.items()])


if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate().to('cpu')
    layer_info = LayerInfo.generate_from_model(model=model, input_shape=(1, 3, 226, 226), device='cpu')

    for name, info in layer_info.items():
        print(f"{name:30s}: {info}")