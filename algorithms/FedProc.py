"""
<FedProc: Prototypical Contrastive Federated Learning on Non-IID data> FGCS2023
本地更新增加原型对比损失。减小本地类的原型与全局同一类原型的距离，增大与全局不同类原型之间的距离
"""
from algorithms.base import *
from algorithms.FedProto import FedProto_Server, agg_func

class FedProc_Client(ClientBase):
    def __init__(self, args, id, train_loader):
        super(FedProc_Client, self).__init__(args, id, train_loader)
        self.local_protos = {}

    def train(self, global_protos, alpha):

        net = self.model.to(self.device)
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)

        if len(global_protos) != 0:
            all_global_protos_keys = np.array(list(global_protos.keys()))
            all_f = []
            for protos_key in all_global_protos_keys:
                temp_f = global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(self.device)
                all_f.append(temp_f.cpu())

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                f = net.features(images)
                outputs = net.classifier(f)
                lossCE = criterion(outputs, labels)

                if len(global_protos) == 0:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i = 0
                    loss_InfoNCE = None
                    for label in labels:
                        if label.item() in global_protos.keys():

                            f_pos = np.array(all_f)[all_global_protos_keys==label.item()][0].to(self.device)

                            f_neg = torch.cat(list(np.array(all_f)[all_global_protos_keys != label.item()])).to(
                                self.device)

                            f_now = f[i].unsqueeze(0)

                            embedding_len = f_pos.shape
                            f_neg = f_neg.unsqueeze(1).view(-1, embedding_len[0])
                            f_pos = f_pos.view(-1, embedding_len[0])
                            f_proto = torch.cat((f_pos, f_neg), dim=0)
                            l = torch.cosine_similarity(f_now, f_proto, dim=1)
                            l = l

                            exp_l = torch.exp(l)
                            exp_l = exp_l.view(1, -1)
                            # l = torch.einsum('nc,ck->nk', [f_now, f_proto.T])
                            # l = l /self.T
                            # exp_l = torch.exp(l)
                            # exp_l = l
                            pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
                            pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
                            pos_mask = pos_mask.view(1, -1)
                            # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
                            pos_l = exp_l * pos_mask
                            sum_pos_l = pos_l.sum(1)
                            sum_exp_l = exp_l.sum(1)
                            loss_instance = -torch.log(sum_pos_l / sum_exp_l)
                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE +=loss_instance
                        i += 1
                    loss_InfoNCE = loss_InfoNCE / i
                loss_InfoNCE = loss_InfoNCE

                loss = alpha * loss_InfoNCE + (1-alpha) * lossCE
                loss.backward()
                optimizer.step()

                if iter == self.local_epoch-1:
                    for i in range(len(labels)):
                        if labels[i].item() in agg_protos_label:
                            agg_protos_label[labels[i].item()].append(f[i,:])
                        else:
                            agg_protos_label[labels[i].item()] = [f[i,:]]

        self.local_protos = agg_func(agg_protos_label)


class FedProc_Server(FedProto_Server):

    def ini(self, client_data_loaders):
        for idx in range(self.args.N_Participants):
            self.clients.append(FedProc_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)

    def run(self, test_loader = None):
        for epoch in range(self.args.CommunicationEpoch):
            self.clients_num_choice = self.select_clients_by_ratio(self.args.clients_select_ratio)
            self.local_update(epoch)
            self.global_update()

            if test_loader is not None:
                testloss, testacc = self.eval_one(test_loader)
                with open("{}_result.txt".format(self.name), 'a+') as fp:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fp.writelines("\n[{}]epoch_{}_acc:{:.3f}_loss:{:.6f}".format(timestamp, epoch, testacc, testloss))
        return None

    def local_update(self, epoch):
        alpha = 1 - epoch/(self.args.CommunicationEpoch-1)
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].receive_model(self.global_model)
            self.clients[idx].train(self.global_protos, alpha)
        return None