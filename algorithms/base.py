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
    def __init__(self, args, id):
        super(Client, self).__init__()
        self.args = args
        self.idx = id

    def ini(self, model_name=None):
        self.model = init_model_by_name(model_name)
        self.model.to(self.args.device)

    def train(self, train_loader):
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.args.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                trainloss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class Server(nn.Module):
    def __init__(self, args):
        super(Server, self).__init__()
        self.args = args
        self.clinets = []

    def ini(self):
        self.global_model = init_model_by_name(self.args.Nets_Name_List[0])
        self.global_model.to(self.args.device)
        for idx in range(self.args.N_Participants):
            self.clinets.append(Client(self.args, idx))
            if len(self.args.Nets_Name_List)==1:
                self.clinets[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clinets[idx].ini(self.args.Nets_Name_List[idx])

    def global_update(self, clients_list):
        pass


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
