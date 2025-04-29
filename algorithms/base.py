import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import copy

def init_model_by_name(name):
    # rewrite soon
    return torchvision.models.resnet18()
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
            outputs = net(images)
            loss = torch.nn.CrossEntropyLoss(reduction='mean')(outputs, labels.long())
            losstotal += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        lossavg = losstotal/len(dataloader)
    return lossavg, acc

class Client(nn.Module):
    def __init__(self, args, id, train_loader):
        super(Client, self).__init__()
        self.args = args
        self.id = id
        self.train_loader = train_loader

    def ini(self, model_name=None):
        self.model = torchvision.models.resnet18()
        self.model.to(self.args.device)

    def train(self):
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.args.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class Server(nn.Module):
    def __init__(self, args, pri_data_loader_list, clients_labelnums):
        super(Server, self).__init__()
        self.args = args
        self.pri_data_loader_list = pri_data_loader_list
        self.clinets = []
        self.clients_num_choice = []
        self.clients_labelnums = clients_labelnums

    def ini(self):
        for idx in range(self.args.N_Participants):
            self.clinets.append(Client(self.args, idx))
            if len(self.args.Nets_Name_List)==1:
                self.clinets[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clinets[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clinets[0].model)
        self.global_model.to(self.args.device)


    def global_update(self):
        self.aggregate_nets()
        for idx in self.clients_num_choice:
            for param, target_param in zip(self.clinets[idx].model.parameters(), self.global_model.parameters()):
                param.data = target_param.data.clone()
        return None

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clinets[idx].train()
        return None

    def aggregate_nets(self, mode='avg'):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)

        if mode == 'avg':
            for idx in self.clients_num_choice:
                client_dict = self.clinets[idx].model.state_dict()
                for key in client_dict.keys():
                    global_dict[key] += client_dict[key] / len(self.clients_num_choice)
        elif mode == 'weights':
            datanum_using = sum([sum(self.clients_labelnums[idx]) for idx in self.clients_num_choice])
            for idx in self.clients_num_choice:
                client_dict = self.clinets[idx].model.state_dict()
                for key in client_dict.keys():
                    global_dict[key] += client_dict[key] * sum(self.clients_labelnums[idx]) / datanum_using
        else:
            raise ValueError("未定义的模型聚合方式{}".format(mode))

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
