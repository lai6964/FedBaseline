"""
DFL《 Disentangled Federated Learning for Tackling Attributes Skew via Invariant Aggregation and Diversity Transferring》ICLM2022
客户端上有两个特征提取器和一个分类器，分别提取全局不变特征和域独有特征，这两特征放入分类器中都能正确分类。本地更新时这两个特征提取器交替更新
全局更新阶段平均特征提取器（不变特征的那一个），平均全局原型
"""

"""
没有官方源代码，也没人复现过，我复现的感觉有些问题。。。

"""

from algorithms.base import *
from algorithms.FedProto import agg_func
from backbone_f.ResNet import *
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        super(ResNet_new, self).__init__(block, num_blocks, num_classes, nf)
        self.feature_G = nn.Sequential(self.conv1,
                                           self.bn1,
                                           self.relu,
                                           self.maxpool,
                                           self.layer1,
                                           self.layer2,
                                           self.layer3,
                                           self.layer4,
                                           self.avgpool,
                                           nn.Flatten())
        self.feature_S = nn.Sequential(self.conv1,
                                           self.bn1,
                                           self.relu,
                                           self.maxpool,
                                           self.layer1,
                                           self.layer2,
                                           self.layer3,
                                           self.layer4,
                                           self.avgpool,
                                           nn.Flatten())
        self.head = self.fc

    def forward(self, x: torch.Tensor, train_G = True) -> torch.Tensor:
        feature_G = self.feature_G(x)
        feature_S = self.feature_S(x)
        if train_G:
            out = self.head(feature_G)
        else:
            out = self.head(feature_S)
        return feature_G, feature_S, out
    def classifier(self, features):
        out = self.head(features)
        return out

def MYNET(num_classes: int) -> ResNet:
    return ResNet_new(BasicBlock, [2, 2, 2, 2], num_classes)

def js_divergence(teacher_outputs, student_outputs):
    T = 1
    p = nn.functional.softmax(teacher_outputs / T, dim=1)
    q = nn.functional.softmax(student_outputs / T, dim=1)
    m = 0.5 * (p + q)
    kl_pm = nn.functional.kl_div(torch.log(p), m, reduction='batchmean')
    kl_qm = nn.functional.kl_div(torch.log(q), m, reduction='batchmean')
    return 0.5 * (kl_pm + kl_qm)

class DFL_Client(ClientBase):
    def __init__(self, args, id, train_loader):
        super().__init__(args, id, train_loader)
        self.local_special_protos = {}

    def train_freeze(self, special_protos_list, train_G=True, train_S=False):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        for name, param in self.model.named_parameters():
            if 'feature_G' in name:
                param.requires_grad = train_G
            if 'feature_S' in name:
                param.requires_grad = train_S
            if 'head' in name:
                param.requires_grad = True

        for epoch in range(self.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features_G, features_S, outputs = self.model(images, train_G)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                if train_G:# 最大化和全局的互信息
                    loss -= js_divergence(features_G, features_S.detach())
                else:# 最小化和本地的互信息
                    loss += js_divergence(features_S, features_G.detach())

                # # 这部分特征增强怪怪的？我先注释掉了
                # for special_protos in special_protos_list:
                #     f_new = copy.deepcopy(features_G.data)
                #     if len(special_protos) > 0:
                #         i = 0
                #         for label in labels:
                #             if label.item() in special_protos.keys():
                #                 f_new[i, :] += special_protos[label.item()][0].data / len(special_protos_list)
                #             i += 1
                # output_aug = self.model.classifier(f_new)
                # loss += nn.CrossEntropyLoss()(output_aug, labels.long())

                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def train(self, special_protos_list):
        self.train_freeze(special_protos_list, train_G=False, train_S=True)
        self.train_freeze(special_protos_list, train_G=True, train_S=False)
        self.get_feature_attribute()

    def receive_model(self, global_model):
        for name, new_param in global_model.named_parameters():
            if name.startswith('feature_G') or name.startswith('head'):
                old_param = self.model.get_parameter(name)
                if old_param is not None:
                    old_param.data = new_param.data.clone()
                else:
                    raise ValueError("未定义的模型参数{}".format(name))

    def get_feature_attribute(self):
        agg_protos_label = {}
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features_G, features_S, outputs = self.model(images)
                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(features_S[i, :])
                    else:
                        agg_protos_label[labels[i].item()] = [features_S[i, :]]
        self.local_special_protos = agg_func(agg_protos_label)


class DFL_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'DFL'
        self.special_protos_list = [{} for i in range(args.N_Participants)]

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(DFL_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def global_update(self):
        self.aggregate_nets()
        for idx in self.clients_num_choice:
            self.special_protos_list[idx] = self.clients[idx].local_special_protos

        return None

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].receive_model(self.global_model)
            self.clients[idx].train(self.special_protos_list)
        return None

    def eval_one(self, epoch, dataloader):
        net = self.global_model.to(self.device)
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, _, outputs = net(images, train_G=True)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

        with open("{}_result.txt".format(self.name), 'a+') as fp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.writelines("\n[{}]epoch_{}_acc:{:.3f}".format(timestamp, epoch, acc))
        return None