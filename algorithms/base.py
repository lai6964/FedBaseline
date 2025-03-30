import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from network import *

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

class Client(nn.Module):
    def __init__(self, id, model_name, train_loader=None, test_loader=None):
        super(Client, self).__init__()
        self.name = "client_{}_{}".format(id, model_name)
        self.model_path = self.result_path+self.name+".pth"
        self.train_loader, self.test_loader = train_loader, test_loader

    def init_clinet(self):
        feature_extractor, feature_dim = get_extractor_by_name(model_name, True)
        custom_classifier = CustomClassifier(feature_dim, num_classes)
        self.model = FeatureExtractorWithClassifier(feature_extractor, custom_classifier)

    def train(self):
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

    def eval(self):
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        return acc

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def clone_model(self, global_model):
        for param, target_param in zip(self.model.parameters(), global_model.parameters()):
            param.data = target_param.data.clone()

class Server(nn.Module):
    def __init__(self, args):
        super(Server, self).__init__()
        self.name = "FedAvg"
        self.args = args
        self.result_path = 'results/' + self.name + '/'
        self.clients_list = []

    def init_server(self):
        create_if_not_exists(self.result_path)
        for idx in range(self.args.N_Participants):
            client = Client(idx)