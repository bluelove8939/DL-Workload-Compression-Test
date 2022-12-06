import os
from models.model_presets import generate_from_chkpoint, imagenet_pretrained


if __name__ == '__main__':
    model = generate_from_chkpoint(
        model_primitive=imagenet_pretrained['AlexNet'].generate(),
        chkpoint_path=os.path.join(os.curdir, 'model_output', 'AlexNet_pruned_tuned_pamt_0.5.pth'))

    print(model)