from algorithms.base import *

class FedAvg_Client(ClientBase):
    pass


class FedAvg_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "FedAvg"

    def ini(self, client_data_loaders):
        self.write_settings()
        for idx in range(self.args.N_Participants):
            self.clients.append(FedAvg_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)



