import os
import torch

from models.model_presets import imagenet_pretrained
from models.tools.lowering import ConvLayerInfo, weight_lowering, ifm_lowering


def weight_lowered_shape(layer_config: ConvLayerInfo):
    Co, Ci, FW, FH = layer_config.weight_shape()
    return Co, Ci*FW*FH

def ifm_lowered_shape(layer_config: ConvLayerInfo):
    N, Ci, W, H = layer_config.N, layer_config.Ci, layer_config.W, layer_config.H
    FW, FH, P, S = layer_config.FW, layer_config.FH, layer_config.P, layer_config.S

    return W*H, Ci*FW*FH


if __name__ == '__main__':
    model_cfg_dirpath = os.path.join(os.curdir, 'model_cfg')
    os.makedirs(model_cfg_dirpath, exist_ok=True)

    for model_type, model_config in imagenet_pretrained.items():
        model_cfg_path = os.path.join(model_cfg_dirpath, model_type+'.csv')

        model = model_config.generate()
        layer_info = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226), device='cpu')

        with open(model_cfg_path, 'wt') as file:
            content = ['layer_name,M,N,K']
            for layer_name, layer_config in layer_info.items():
                WH, WW = weight_lowered_shape(layer_config)
                IH, IW = ifm_lowered_shape(layer_config)

                print(WH, WW, IH, IW)

                content.append(f"{layer_name},{WH},{IH},{WW}")

            file.write('\n'.join(content))