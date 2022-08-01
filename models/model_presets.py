import torchvision


class ModelConfig(object):
    def __init__(self, model_type, weights=None):
        self.model_type = model_type
        self.weights = weights

    def generate(self):
        return self.model_type(weights=self.weights)

class QuantModelConfig(object):
    def __init__(self, model_type, weights=None):
        self.model_type = model_type
        self.weights = weights

    def generate(self):
        return self.model_type(weights=self.weights, quantize=True)


imagenet_pretrained = {
    'ResNet50': ModelConfig(torchvision.models.resnet50, weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1),
    'AlexNet': ModelConfig(torchvision.models.alexnet, weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1),
    'VGG16': ModelConfig(torchvision.models.vgg16, weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1),
    'SqueezeNet': ModelConfig(torchvision.models.squeezenet1_0, weights=torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1),
    'InceptionV3': ModelConfig(torchvision.models.inception_v3, weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1),
}

imagenet_quant_pretrained = {
    'ResNet50': QuantModelConfig(torchvision.models.quantization.resnet50, weights=torchvision.models.quantization.ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
    'GoogLeNet': QuantModelConfig(torchvision.models.quantization.googlenet, weights=torchvision.models.quantization.GoogLeNet_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
    'InceptionV3': QuantModelConfig(torchvision.models.quantization.inception_v3, weights=torchvision.models.quantization.Inception_V3_QuantizedWeights.IMAGENET1K_FBGEMM_V1),
}