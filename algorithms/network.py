import torch
import torch.nn as nn
import torchvision.models as models


def get_extractor_by_name(name, pretrained=True):
    """
    根据模型名称加载预训练模型。

    参数:
        name (str): 模型名称，例如 'resnet50', 'vgg16', 'densenet121' 等。
        pretrained (bool): 是否加载预训练权重。默认为 True。

    返回:
        model (nn.Module): 加载的模型。
    """
    # 定义一个字典，将模型名称映射到对应的模型加载函数
    model_functions = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "vgg11": models.vgg11,
        "vgg13": models.vgg13,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19,
        "densenet121": models.densenet121,
        "densenet169": models.densenet169,
        "densenet201": models.densenet201,
        "inception_v3": models.inception_v3,
        "mobilenet_v2": models.mobilenet_v2,
        "shufflenet_v2_x0_5": models.shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
        "shufflenet_v2_x1_5": models.shufflenet_v2_x1_5,
        "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0,
        "squeezenet1_0": models.squeezenet1_0,
        "squeezenet1_1": models.squeezenet1_1,
        "wide_resnet50_2": models.wide_resnet50_2,
        "wide_resnet101_2": models.wide_resnet101_2,
    }

    # 检查输入的模型名称是否有效
    if name not in model_functions:
        raise ValueError(f"Invalid model name: {name}. Supported models are: {list(model_functions.keys())}")

    # 加载模型
    if pretrained:
        # 在较新的 PyTorch 版本中，使用 weights 参数
        try:
            model = model_functions[name](weights=models.get_model_weights(name).DEFAULT)
        except AttributeError:
            # 在较旧的 PyTorch 版本中，使用 pretrained 参数
            model = model_functions[name](pretrained=True)
    else:
        model = model_functions[name](pretrained=False)
        # 获取特征维度
    if name.startswith("resnet"):
        feature_dim = model.fc.in_features
    elif name.startswith("vgg"):
        feature_dim = model.classifier[6].in_features
    elif name.startswith("densenet"):
        feature_dim = model.classifier.in_features
    elif name.startswith("inception"):
        feature_dim = model.fc.in_features
    elif name.startswith("mobilenet"):
        feature_dim = model.classifier[1].in_features
    elif name.startswith("shufflenet"):
        feature_dim = model.fc.in_features
    elif name.startswith("squeezenet"):
        feature_dim = model.classifier[1].in_channels
    elif name.startswith("wide_resnet"):
        feature_dim = model.fc.in_features
    else:
        raise ValueError(f"Unsupported model: {name}")

        # 删除全连接层，保留特征提取部分
    if name.startswith("vgg"):
        feature_extractor = nn.Sequential(*list(model.children())[:-2])
    elif name.startswith("squeezenet"):
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    else:
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

    return feature_extractor, feature_dim

class CustomClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FeatureExtractorWithClassifier(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(FeatureExtractorWithClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


if __name__ == '__main__':
    model_name = "resnet18"
    num_classes = 10

    # 获取特征提取器和特征维度
    feature_extractor, feature_dim = get_extractor_by_name(model_name, True)

    # 创建自定义分类器
    custom_classifier = CustomClassifier(feature_dim, num_classes)

    # 创建完整的模型
    model_with_classifier = FeatureExtractorWithClassifier(feature_extractor, custom_classifier)

    # 测试模型
    input_tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model_with_classifier(input_tensor)

    print(f"Loaded model: {model_name}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Output shape: {output.shape}")