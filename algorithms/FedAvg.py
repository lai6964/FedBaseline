from algorithms.base import *


class FedAvg(Server):
    def __init__(self, args):
        super(FedAvg,self).__init__(args)
        self.name = "FedAvg"

    def global_update(self, clients_num_choice):
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)

        for idx in clients_num_choice:
            client_dict = self.clinets[idx].model.state_dict()
            for key in client_dict.keys():
                global_dict[key] += client_dict[key] / len(clients_num_choice)
        self.global_model.load_state_dict(global_dict)

        for idx in clients_num_choice:
            for param, target_param in zip(self.clinets[idx].model.parameters(), self.global_model.parameters()):
                param.data = target_param.data.clone()

    def local_update(self, pri_data_loader_list, clients_num_choice):
        for idx in tqdm(clients_num_choice):
            self.clinets[idx].train(pri_data_loader_list[idx])

    def run(self,pri_data_loader_list, test_loader = None):
        testacc, testloss = 0,0
        # iterator = tqdm(range(self.args.CommunicationEpoch))
        # for epoch in iterator:
        for epoch in range(self.args.CommunicationEpoch):
            clients_num_choice = self.select_clients_by_ratio(self.args.clients_select_ratio)
            self.global_update(clients_num_choice)
            self.local_update(pri_data_loader_list, clients_num_choice)

            if 1:#epoch>10:# and len(test_loader):
                testloss, testacc = eval_one(self.global_model, test_loader, self.args.device)
                with open("FedAvg_result3.txt", 'a+') as fp:
                    fp.writelines("\nepoch_{}_acc:{:.3f}_loss:{:.6f}".format(epoch, testacc, testloss))
            print("epoch_{}_acc:{:.3f}_loss:{:.6f}".format(epoch, testacc, testloss))


            # iterator.desc = "Epoch %d testloss = %0.6f testacc = %.3f" % (epoch, testloss, testacc)




