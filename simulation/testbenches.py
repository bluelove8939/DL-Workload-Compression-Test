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
    # ("lexNet", "features.8"):  'AC3',
    ("AlexNet", "features.10"): 'AC3',
}

quantized_testbenches = {
    # ResNet50
    ("ResNet50", "layer1.0.conv2"): 'QRC1',
    ("ResNet50", "layer2.3.conv2"): 'QRC2',
    ("ResNet50", "layer3.5.conv2"): 'QRC3',
    ("ResNet50", "layer4.2.conv2"): 'QRC4',

    # GoogLeNet
    ("GoogLeNet", "inception3a.branch2.1.conv"): 'QGC1',
    ("GoogLeNet", "inception3b.branch3.1.conv"): 'QGC2',
    ("GoogLeNet", "inception4a.branch2.1.conv"): 'QGC3',
    ("GoogLeNet", "inception4c.branch2.0.conv"): 'QGC4',
}

pruned_testbenches = {
    # ResNet50
    ("ResNet50", "layer1.0.conv2"): 'RC1',
    ("ResNet50", "layer2.3.conv2"): 'RC2',
    ("ResNet50", "layer3.5.conv2"): 'RC3',
    ("ResNet50", "layer4.2.conv2"): 'RC4',

    # AlexNet
    ("AlexNet", "features.3"): 'AC1',
    ("AlexNet", "features.6"): 'AC2',
    ("AlexNet", "features.10"): 'AC3',
}


def layer_filter(model_name, layer_name):
    return (model_name, layer_name) in testbenches.keys()

def quant_layer_filter(model_name, layer_name):
    return (model_name, layer_name) in quantized_testbenches.keys()

def pruned_layer_filter(model_name, layer_name):
    return (model_name, layer_name) in pruned_testbenches.keys()