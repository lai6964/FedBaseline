"""
FedRep《Exploiting Shared Representations for Personalized Federated Learning》 ICML2021
看文章是把模型分成 特征提取器+特征降维层+分类器 三部分。文中的head指的是分类器，$\phi$ 指的是特征提取器（这和常见的检测头检测层刚好相反，别弄错了），
然后全局更新时只平均特征提取器，本地更新时先固定分类器更新特征提取器，再固定特征提取器更新分类器。代码中直接忽略了特征降维层，没有看到SVD的痕迹
"""

from algorithms.base import *
from torchvision.models.resnet import ResNet, BasicBlock
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int = 10) -> None:
        super(ResNet_new, self).__init__(block, layers, num_classes)
        self.feature_extra = nn.Sequential(self.conv1,
                                           self.bn1,
                                           self.relu,
                                           self.maxpool,
                                           self.layer1,
                                           self.layer2,
                                           self.layer3,
                                           self.layer4,
                                           self.avgpool,
                                           nn.Flatten())
        self.classifier = self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extra(x)
        out = self.fc(feature)
        return out
def MYNET(num_classes: int):
    return ResNet_new(BasicBlock, [2, 2, 2, 2], num_classes)


class FedRep_Client(ClientBase):
    def ini(self, model_name=None):
        self.model = MYNET(self.args.N_Class)
        self.model.to(self.device)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        # ！！！！ 他们这变把optimizer分成optimizer_feat 和 optimizer_cla 两个分别更新，我直接写一起了

        # 先运行一次更新分类器
        for name, param in self.model.named_parameters():  # 冻结 特征提取层 的权重
            if 'feature_extra' in name:
                param.requires_grad = False
            if 'classifier' in name:
                param.requires_grad = True
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新特征提取器
        for name, param in self.model.named_parameters():# 冻结 分类器 的权重
            if 'feature_extra' in name:
                param.requires_grad = True
            if 'classifier' in name:
                param.requires_grad = False
        for epoch in range(self.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                trainloss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return None

    def receive_model(self, global_model):
        for name, new_param in global_model.named_parameters():
            if name.startswith('feature_extra'):
                old_param = self.model.get_parameter(name)
                if old_param is not None:
                    old_param.data = new_param.data.clone()
                else:
                    raise ValueError("未定义的模型参数{}".format(name))

class FedRep_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedRep'
        self.aggregate_mode = 'weights'

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedRep_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def eval_one(self, epoch, dataloader):
        acc_list = []
        clients = [self.clients[idx] for idx in self.clients_num_choice]
        for client in clients:
            net = client.model.to(self.device)
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct / total

                acc_list.append(acc)
        acc = sum(acc_list) / len(acc_list)
        with open("{}_result.txt".format(self.name), 'a+') as fp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.writelines("\n[{}]epoch_{}_acc:{:.3f}".format(timestamp, epoch, acc))
        return acc


if __name__ == '__main__':
    model = MYNET(10,64)
    model_dict = model.state_dict()
    for name in model_dict.keys():
        print(name)