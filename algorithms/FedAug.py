from algorithms.base import *

class FedAug_Client(ClientBase):
    def ini(self, model_name=None):
        from backbone_f.init_fl_model import get_model_by_name
        self.model = get_model_by_name(model_name)
        self.model.to(self.device)

    def train(self, global_protos):
        self.model.to(self.args.device)
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
        return None


class FedAug_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "FedAug"


