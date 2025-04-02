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
    def __init__(self, id, model_name=None, train_loader=None, test_loader=None):
        super(Client, self).__init__()
        self.name = "client_{}_{}".format(id, model_name)
        self.model_name = model_name
        self.train_loader, self.test_loader = train_loader, test_loader

    def init_clinet(self):
        self.model = init_model_by_name(self.model_name)

    def _localupdate(self):
        self.model.to(self.args.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.local_lr, weight_decay=1e-5)

        self.model.train()
        iterator = tqdm(range(self.args.local_epoch))
        for epoch in iterator:
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
            trainloss = trainloss / len(self.train_loader)
            iterator.desc = "{} Epoch %d Local loss = %0.6f at lr = %0.7f for testloss = %0.4f testacc = %.3f" % (
                self.name, epoch, trainloss, optimizer.param_groups[0]['lr'])


    def clone_model(self, global_model):
        for param, target_param in zip(self.model.parameters(), global_model.parameters()):
            param.data = target_param.data.clone()

class Server(nn.Module):
    def __init__(self, args):
        super(Server, self).__init__()
        self.args = args
    def init_server(self):
        self.global_model = init_model_by_name(self.args.Nets_Name_List[0])


    def aggregation(self, clients_list):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

        for client_model in self.clients_list_choice:
            client_dict = client_model.state_dict()
            for key in client_dict.keys():
                global_dict[key] += client_dict[key] / len(self.clients_list_choice)
        self.global_model.load_state_dict(global_dict)

    def update_client(self):
        for client in self.clients_list_choice:
            client = copy.deepcopy(self.global_model)
            client._localupdate()

