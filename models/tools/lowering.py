import math
import torch
import numpy as np


class ConvLayerInfo(object):
    def __init__(self, N  : int or None=None, Ci : int or None=None, W  : int or None=None, H : int or None=None,
                       Co : int or None=None, FW : int or None=None, FH : int or None=None, S : int or None=None,
                       P  : int or None=None, OW : int or None=None, OH : int or None=None):

        self.N  = N         # batch size
        self.Ci = Ci        # input channel
        self.W  = W         # image width
        self.H  = H         # image height
        self.Co = Co        # output channel
        self.FW = FW        # filter width
        self.FH = FH        # filter height
        self.S  = S         # stride
        self.P  = P         # padding
        self.OW = OW        # output width
        self.OH = OH        # output height

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
            if 'conv' in type(sublayer).__name__.lower() and hasattr(sublayer, 'weight'):
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


def weight_lowering(weight: torch.Tensor, layer_info: ConvLayerInfo):
    Co, Ci, FW, FH = layer_info.weight_shape()
    return torch.reshape(weight.detach(), shape=(Co, Ci*FW*FH))

def ifm_lowering(ifm: torch.Tensor, layer_info: ConvLayerInfo):
    N, Ci, W, H = layer_info.ifm_shape()
    FW, FH, P, S = layer_info.FW, layer_info.FH, layer_info.P, layer_info.S

    ifm = torch.nn.functional.pad(ifm, (P, P, P, P), value=0)  # add zero padding manually

    lowered_ifm = []
    for n in range(N):
        for rp in range(0, H - FH + (2 * P) + 1, S):
            for cp in range(0, W - FW + (2 * P) + 1, S):
                lowered_ifm.append(list(ifm[n, :, rp:rp + FH, cp:cp + FW].flatten()))
    lowered_ifm = torch.tensor(lowered_ifm)

    return lowered_ifm


if __name__ == '__main__':
    from models.model_presets import imagenet_quant_pretrained

    config = imagenet_quant_pretrained['ResNet50']
    model = config.generate()

    info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226))

    target_name = 'conv1'
    print(info[target_name])

    N, Ci, W, H = info[target_name].ifm_shape()

    ifm = torch.tensor(np.arange(0, N*Ci*W*H, 1, dtype=np.dtype('int32')).reshape(info[target_name].ifm_shape()))
    lowered_ifm = ifm_lowering(ifm, layer_info=info[target_name])


    print(ifm.shape)
    print(lowered_ifm.shape)

    weight = model.state_dict()[target_name + '.weight']
    lowered_weight = weight_lowering(weight, layer_info=info[target_name])

    print(lowered_weight.shape)