"""
《Federated Optimization in Heterogeneous Networks》MLSys2020
本地更新增加正则项，惩罚模型参数与全局模型参数之间的偏差
"""
from algorithms.base import *

class FedProx_Client(ClientBase):

    def train(self, global_model):
        self.model.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        global_model.to(self.device)
        global_weight_collector = list(global_model.parameters())
        for _ in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((0.01 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += self.mu * fed_prox_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return None

class FedProx_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedProx'

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedProx_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].train(self.global_model)
        return None
