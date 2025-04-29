from algorithms.base import *#Client, Server







class FedAug(Server):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedAug'

    def run(self):
        return None