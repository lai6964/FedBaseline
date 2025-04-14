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
    return torchvision.models.resnet18()


class Client(nn.Module):
    def __init__(self, args, id, model_name=None):
        super(Client, self).__init__()
        self.args = args
        self.idx = id
        self.model = init_model_by_name(model_name)

    def local_update(self, train_loader):
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.args.local_epoch):
            trainloss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
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

    def ini(self):
        self.global_model = init_model_by_name(self.args.Nets_Name_List[0])
        self.clinets = [Client(self.args, idx, self.args.Nets_Name_List[idx]) for idx in self.args.N_Participants]

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
