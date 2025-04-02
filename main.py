import argparse
from utils import *
from data_utils.data_loader import MyDigits
from data_utils.split_dirichlet import dirichlet_split_noniid
from  algorithms.base import Client, Server

def args_parser():
    parser = argparse.ArgumentParser()
    '''    Default Setting    '''
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--Seed',type=int,default=42)
    parser.add_argument('--Dataset_Dir',type=str,default='../Dataset/')
    parser.add_argument('--Model_Path',type=str,default='Model_Storage')
    parser.add_argument('--model', type=str, default='SOLO', help='Model name.')
    parser.add_argument('--Scenario',type=str,default='mnist')


    '''    Traning Setting    '''
    parser.add_argument('--CommunicationEpoch',type=int,default=40)
    parser.add_argument('--features_dim',type=int,default=512)
    parser.add_argument('--public_epoch',type=int,default=1)
    parser.add_argument('--local_epoch',type=int,default=40)
    parser.add_argument('--public_lr',type=float,default=0.001)
    parser.add_argument('--local_lr',type=float,default=0.001)
    parser.add_argument('--local_batch_size', type=int, default=256)
    parser.add_argument('--public_batch_size', type=int, default=256)
    parser.add_argument('--ReLoad', type=str2bool, default=False)
    parser.add_argument('--lrschedule', type=str2bool, default=False)

    '''    Data Setting    '''
    parser.add_argument('--Public_Dataset_Name', type=str, default='cifar_100')
    parser.add_argument('--public_len', type=int, default=5000)
    parser.add_argument('--pub_aug', type=str, default='weak')
    parser.add_argument('--N_Class', type=int, default=10)
    parser.add_argument('--N_Participants', type=int, default=4)
    parser.add_argument('--Dirichlet_beta', type=float, default=0.5)
    parser.add_argument('--DataPart', type=str2bool, default=True)

    args = parser.parse_args()
    args.device =torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = args_parser()
    # 设置随机种子
    init_seed(args.Seed)

    train_dataset = MyDigits(root=args.Dataset_Dir, data_name = args.Scenario)
    train_labels = train_dataset.targets

    # 使用狄利克雷分布分割数据集
    client_datasets, client_data_loaders = dirichlet_split_noniid(train_dataset, args.Dirichlet_beta, args.N_Participants)
    # 打印每个客户端的数据分布情况
    for i, dataset in enumerate(client_datasets):
        labels = [train_dataset.targets[idx].item() for idx in dataset.indices]
        print(f"Client {i} has {len(labels)} samples with distribution: {torch.bincount(torch.tensor(labels))}")

    server = Server(args)
    server.init_server()
    clients_list = []
    for idx in range(args.N_Participants):
        client = Client(idx, args.Nets_Name_List[idx])
        client.init_clinet()
        clients_list.append(client)

    for epoch in range(args.CommunicationEpoch):
        

        clients_per_round = max(int(self.args.N_Participants * self.clients_sample_ratio), 1)
        self.clients_list_choice = np.random.choice(self.args.N_Participants, clients_per_round)



