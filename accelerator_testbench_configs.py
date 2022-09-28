import os

from models.model_presets import imagenet_pretrained, imagenet_quant_pretrained
from models.tools.lowering import ConvLayerInfo


if __name__ == '__main__':
    save_dirname = os.path.join(os.curdir, 'logs')
    save_filename = 'accelerator_testbench_configs.txt'
    content = []

    for model_name, config in imagenet_pretrained.items():
        model = config.generate()
        tb_configs = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226))

        for layer_name, layer_info in tb_configs.items():
            content.append(f"{model_name:15s} {layer_name:30s} {layer_info}")

    with open(os.path.join(save_dirname, save_filename), 'wt') as file:
        file.write('\n'.join(content))

if __name__ == '__main__':
    save_dirname = os.path.join(os.curdir, 'logs')
    save_filename = 'accelerator_testbench_configs_quant.txt'
    content = []

    for model_name, config in imagenet_quant_pretrained.items():
        model = config.generate()
        tb_configs = ConvLayerInfo.generate_from_model(model, input_shape=(1, 3, 226, 226))

        for layer_name, layer_info in tb_configs.items():
            content.append(f"{model_name:15s} {layer_name:30s} {layer_info}")

    with open(os.path.join(save_dirname, save_filename), 'wt') as file:
        file.write('\n'.join(content))
