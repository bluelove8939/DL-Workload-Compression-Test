testbenches = {
    # VGG16
    ("VGG16", "features.2"):  'VC1',
    ("VGG16", "features.7"):  'VC2',
    ("VGG16", "features.12"): 'VC3',
    ("VGG16", "features.19"): 'VC4',

    # ResNet50
    ("ResNet50", "layer1.0.conv2"): 'RC1',
    ("ResNet50", "layer2.3.conv2"): 'RC2',
    ("ResNet50", "layer3.5.conv2"): 'RC3',
    ("ResNet50", "layer4.2.conv2"): 'RC4',

    # AlexNet
    ("AlexNet", "features.3"):  'AC1',
    ("AlexNet", "features.6"):  'AC2',
    ("AlexNet", "features.10"): 'AC3',
}


def testbench_filter(model_name, submodule_name):
    return (model_name, submodule_name) in testbenches.keys()