import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy
from datetime import datetime
from typing import List




class ClientBase(nn.Module):
    def __init__(self, args, id, train_loader):
        super(ClientBase, self).__init__()
        self.args = args
        self.id = id
        self.train_loader = train_loader
        self.device = args.device
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.num_classes = args.N_Class

    def ini(self, model_name=None):
        self.model = torchvision.models.resnet18(num_classes=self.args.N_Class)
        self.model.to(self.device)

    def train(self):
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr, weight_decay=1e-5)

        self.model.train()
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

    def receive_model(self, global_model):
        for new_param, old_param in zip(global_model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()


class ServerBase(nn.Module):

    def __init__(self, args):
        super(ServerBase, self).__init__()
        self.args = args
        self.clients = []
        self.clients_num_choice = []
        self.clients_labelnums = args.clients_labelnums
        self.aggregate_mode = 'avg'
        self.device = args.device

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(ClientBase(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def run(self, test_loader = None):
        for epoch in range(self.args.CommunicationEpoch):
            self.clients_num_choice = self.select_clients_by_ratio(self.args.clients_select_ratio)
            self.local_update()
            self.global_update()

            if test_loader is not None:
                if epoch%self.args.eval_epoch_gap==0:
                    self.eval(epoch, test_loader)
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
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

        with open("{}_result.txt".format(self.name), 'a+') as fp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.writelines("\n[{}]epoch_{}_acc:{:.3f}".format(timestamp, epoch, acc))
        return acc

    def global_update(self):
        self.aggregate_nets()
        return None

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].receive_model(self.global_model)
            self.clients[idx].train()
        return None

    def aggregate_nets(self):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)


        clients = [self.clients[idx] for idx in self.clients_num_choice]
        if self.aggregate_mode == 'avg':
            parti_num = len(self.clients_num_choice)
            ratios = [1 / parti_num for _ in range(parti_num)]
        elif self.aggregate_mode == 'weights':
            clients_datanums = [sum(self.clients_labelnums[idx]) for idx in self.clients_num_choice]
            ratios = [datanums/sum(clients_datanums) for datanums in clients_datanums]
        else:
            raise ValueError("未定义的模型聚合方式{}".format(self.aggregate_mode))

        for ratio, client in zip(ratios, clients):
            client_dict = client.model.state_dict()
            for key in client_dict.keys():
                global_dict[key] += client_dict[key] * ratio

        self.global_model.load_state_dict(global_dict)
        return None


    def select_clients_by_num(self, M):
        N = self.args.N_Participants
        if M > N:
            raise ValueError("M 不能大于 N")
        selected_clients = random.sample(list(range(N)), M)
        return selected_clients

    def select_clients_by_ratio(self, ratio):
        if not (0 <= ratio <= 1):
            raise ValueError("比例必须在0到1之间")
        return self.select_clients_by_num(max(1, int(self.args.N_Participants * ratio)))
