"""
FedRoD《On Bridging Generic and Personalized Federated Learning for Image Classification》 ICLR2022
看文章是把模型分成 特征提取器+两个分类器 。一个针对个性化进行经验风险最小化，一个和fedavg一样进行平衡风险最小化
聚合时个性化分类器没平均。
"""

from algorithms.base import *
from backbone_f.ResNet import *
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        super(ResNet_new, self).__init__(block, num_blocks, num_classes, nf)
        self.feature_extra = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4,
                                       nn.Flatten())
        self.G_head = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.P_head = nn.Linear(nf * 8 * block.expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extra(x)
        out_G = self.G_head(feature)
        out_P = self.P_head(feature.detach())
        output = out_G.detach() + out_P
        return feature, out_G, output
def MYNET(num_classes: int, nf: int = 64) -> ResNet:
    return ResNet_new(BasicBlock, [2, 2, 2, 2], num_classes, nf)


class FedRoD_Client(ClientBase):
    def __init__(self, args, id, train_loader, sample_per_class):
        super(FedRoD_Client, self).__init__(args, id, train_loader)
        self.sample_per_class = torch.tensor(sample_per_class)

    def ini(self, model_name=None):
        self.model = MYNET(self.args.N_Class)
        self.model.to(self.device)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        # optimizer_feat = optim.SGD(self.model.feature_extra.parameters(), lr=self.local_lr)
        # optimizer_head = torch.optim.SGD(self.model.classifier.parameters(), lr=self.local_lr)
        for epoch in range(self.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs_G, outputs = self.model(images)
                # 这里不知道为啥他们要分开写，我把detach写在model里面了，直接一起回传
                loss_bsm = self.balanced_softmax_loss(labels, features)
                loss_CE = nn.CrossEntropyLoss()(outputs, labels.long())
                loss = loss_CE + loss_bsm
                trainloss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return None

    def balanced_softmax_loss(self, labels, logits, reduction="mean"):
        """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          sample_per_class: A int tensor of size [no of classes].
          reduction: string. One of "none", "mean", "sum"
        Returns:
          loss: A float tensor. Balanced Softmax Loss.
        """
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
        return loss

    def receive_model(self, global_model):
        for name, new_param in global_model.named_parameters():
            if name.startswith('feature_extra') or name.startswith('G_head'):
                old_param = self.model.get_parameter(name)
                if old_param is not None:
                    old_param.data = new_param.data.clone()
                else:
                    raise ValueError("未定义的模型参数{}".format(name))

class FedRoD_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedRoD'
        self.aggregate_mode = 'weights'

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedRoD_Client(self.args, idx, client_data_loaders[idx], self.clients_labelnums[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def eval_one(self, epoch, dataloader, personalization=False):
        net = self.global_model.to(self.device)
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                _, outputs, _ = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc_G = 100 * correct / total

        if personalization:
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
                        _, outputs_G, outputs = net(images)
                        outputs = outputs-outputs_G
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    acc_list.append(acc)
            acc_P = sum(acc_list) / len(acc_list)
            with open("{}_result.txt".format(self.name), 'a+') as fp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fp.writelines("\n[{}]epoch_{}_accG:{:.3f}_accP:{:.3f}".format(timestamp, epoch, acc_G, acc_P))
            return acc_G, acc_P
        with open("{}_result.txt".format(self.name), 'a+') as fp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.writelines("\n[{}]epoch_{}_acc:{:.3f}".format(timestamp, epoch, acc_G))
        return acc_G

if __name__ == '__main__':
    model = MYNET(10)
    model_dict = model.state_dict()
    for name in model_dict.keys():
        print(name)