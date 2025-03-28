from backbone.ResNet import ResNet10, ResNet12, ResNet18, ResNet20, ResNet34
from backbone.googlenet import GoogLeNet, Inception
from backbone.mobilnet_v2 import MobileNetV2
from backbone.efficientnet import EfficientNetB0
from backbone.shufflenet import ShuffleNetG2


def get_model_by_name(model_name,nclasses=10):
    try:
        # 获取当前模块中的全局变量字典
        globals_dict = globals()

        # 根据模型名从全局变量字典中获取模型
        model = globals_dict[model_name]

        # 实例化模型
        instance = model(num_classes=nclasses)

        return instance
    except KeyError:
        return f"Model '{model_name}' not found."
    except Exception as e:
        return f"Error: {e}"

def init_nets(n_parties,nets_name_list,nclasses=10):
    nets_list = [ None for net_i in range(n_parties)]
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        net = get_model_by_name(net_name, nclasses)
        print(net_name)
        nets_list[net_i] = net
    return nets_list
