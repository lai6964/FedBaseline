from algorithms.base import Client, Server


class FedAvg(Server):
    def __init__(self, args):
        super(FedAvg,self).__init__(args)
        self.name = "FedAvg"


