from algorithms.base import *  # Client, Server


def eval_one(net, dataloader, device):
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        losstotal = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            _, outputs = net(images)
            loss = torch.nn.CrossEntropyLoss(reduction='mean')(outputs, labels.long())
            losstotal += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        lossavg = losstotal/len(dataloader)
    return lossavg, acc

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


class FedProto_Client(Client):
    def __init__(self, args, id, train_loader):
        super(FedProto_Client, self).__init__(args, id, train_loader)
        self.local_protos = {}
        self.mu = 0.01
        self.train_loader = train_loader

    def ini(self, model_name=None):
        from backbone.init_fl_model import get_model_by_name
        self.model = get_model_by_name('ResNet18')
        self.model.to(self.args.device)

    def train(self, global_protos):
        self.model.to(self.args.device)
        # optimizer = optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=0.9, weight_decay=1e-5)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        self.model.train()
        agg_protos_label = {}
        for epoch in range(self.args.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                features, outputs = self.model(images)
                loss_CE = nn.CrossEntropyLoss()(outputs, labels.long())

                if len(global_protos) == 0:
                    lossProto = 0*loss_CE
                else:
                    f_new = copy.deepcopy(features.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            f_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    lossProto = nn.MSELoss()(f_new, features)
                loss = loss_CE + lossProto * self.mu
                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch == self.args.local_epoch-1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(features[i,:])
                        else:
                            agg_protos_label[labels[i].item()] = [features[i,:]]
        self.local_protos = agg_func(agg_protos_label)


class FedProto(Server):
    def __init__(self, args, pri_data_loader_list, clients_labelnums):
        super().__init__(args, pri_data_loader_list, clients_labelnums)
        self.name = 'FedProto'
        self.global_protos = []

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedProto_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.args.device)


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
        self.aggregate_nets(mode="weights")
        self.global_protos = self.proto_aggregation()
        for idx in self.clients_num_choice:
            for param, target_param in zip(self.clients[idx].model.parameters(), self.global_model.parameters()):
                param.data = target_param.data.clone()
        return None

    def run(self, test_loader=None):
        testacc, testloss = 0, 0
        for epoch in range(self.args.CommunicationEpoch):
            self.clients_num_choice = self.select_clients_by_ratio(self.args.clients_select_ratio)
            self.local_update()
            self.global_update()

            if 1:  # epoch>10:# and len(test_loader):
                testloss, testacc = eval_one(self.global_model, test_loader, self.args.device)
                with open("{}_result.txt".format(self.name), 'a+') as fp:
                    fp.writelines("\nepoch_{}_acc:{:.3f}_loss:{:.6f}".format(epoch, testacc, testloss))
            print("epoch_{}_acc:{:.3f}_loss:{:.6f}".format(epoch, testacc, testloss))
        return None