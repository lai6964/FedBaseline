"""
《FedProto: Federated Prototype Learning across Heterogeneous Clients》AAAI2022
每个客户端算类别原型，计算类别的全局原型（这里原文和代码不同，原本有客户端上类别样本数量的权重，代码直接取了平均）
全局只传输了原型，没有传模型
"""
from algorithms.base import *
from torchvision.models.resnet import ResNet, BasicBlock
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int = 10) -> None:
        super(ResNet_new, self).__init__(block, layers, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        out = self.fc(feature)
        return feature, out
def MYNET(num_classes: int):
    return ResNet_new(BasicBlock, [2, 2, 2, 2], num_classes)

def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

class FedProto_Client(ClientBase):
    def __init__(self, args, id, train_loader):
        super().__init__(args, id, train_loader)
        self.local_protos = {}
        self.mu = 0.01

    def ini(self):
        self.model = MYNET(self.num_classes)
        self.model.to(self.device)


    def train(self, global_protos):
        self.model.to(self.device)
        # optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())

                if len(global_protos) == 0:
                    continue
                else:
                    f_new = copy.deepcopy(features.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            f_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss += self.mu * nn.MSELoss()(f_new, features)
                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.get_local_protos()
        return None

    def get_local_protos(self):
        agg_protos_label = {}
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs = self.model(images)
                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(features[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [features[i, :]]
        self.local_special_protos = agg_func(agg_protos_label)
        return None


class FedProto_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedProto'
        self.aggregate_mode = 'weights'
        self.global_protos = []

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedProto_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)


    def proto_aggregation(self):
        # 官方代码和原文不一致，原文应该有考虑每个client每个label的数量，但官方代码直接先客户端算平均标签特征，然后全局再均值
        # 这个函数是官方代码copy过来的
        agg_protos_label = dict()
        for idx in self.clients_num_choice:
            local_protos = self.clients[idx].local_protos
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label

    def proto_aggregation_ratio(self):
        # 这是按原文理解来写的
        agg_protos_label = dict()
        clients_labelnums = [self.clients_labelnums[i] for i in self.clients_num_choice]
        clients = [self.clients[i] for i in self.clients_num_choice]

        for label in range(len(clients_labelnums[0])):
            ratios = [clients_labelnums[i][label] for i in range(len(clients_labelnums))]
            ratios = ratios/sum(ratios)
            for ratio, client in zip(ratios, clients):
                if label in agg_protos_label:
                    agg_protos_label[label] += client.local_protos[label].data * ratio
                else:
                    agg_protos_label[label] = client.local_protos[label].data * ratio
                agg_protos_label[label] = [agg_protos_label[label]]
        return agg_protos_label

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].train(self.global_protos)
        return None

    def global_update(self):
        self.aggregate_nets()
        self.global_protos = self.proto_aggregation()
        return None

    def eval(self, epoch, dataloader):
        net = self.global_model.to(self.device)
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

        with open("{}_result.txt".format(self.name), 'a+') as fp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.writelines("\n[{}]epoch_{}_acc:{:.3f}".format(timestamp, epoch, acc))
        return acc

if __name__ == '__main__':
    model = MYNET(10)
    random_input = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        feature, output = model(random_input)

    # 查看输出结果
    print("Output shape:", output.shape)
    print("Output tensor:", output)