import os
from models.model_presets import generate_from_quant_chkpoint, imagenet_pretrained


if __name__ == '__main__':
    model = generate_from_quant_chkpoint(
        model_primitive=imagenet_pretrained['AlexNet'].generate(),
        chkpoint_path=os.path.join(os.curdir, 'model_output', 'AlexNet_quantized_tuned_citer_10.pth'))

    print(model)