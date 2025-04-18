import argparse
import torch, torchvision
from backbone.ResNet import ResNet18
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
    parser.add_argument('--Scenario',type=str,default='MNIST')


    '''    Traning Setting    '''
    parser.add_argument('--CommunicationEpoch',type=int,default=400)
    parser.add_argument('--features_dim',type=int,default=512)
    parser.add_argument('--public_epoch',type=int,default=1)
    parser.add_argument('--local_epoch',type=int,default=20)
    parser.add_argument('--public_lr',type=float,default=0.001)
    parser.add_argument('--local_lr',type=float,default=0.001)
    parser.add_argument('--local_batch_size', type=int, default=256)
    parser.add_argument('--public_batch_size', type=int, default=256)
    parser.add_argument('--ReLoad', type=str2bool, default=False)
    parser.add_argument('--lrschedule', type=str2bool, default=False)
    parser.add_argument('--clients_select_ratio',type=float,default=0.3)
    parser.add_argument('--Nets_Name_List',type=list,default=['ResNet18'])

    '''    Data Setting    '''
    parser.add_argument('--Public_Dataset_Name', type=str, default='cifar_100')
    parser.add_argument('--public_len', type=int, default=5000)
    parser.add_argument('--pub_aug', type=str, default='weak')
    parser.add_argument('--N_Class', type=int, default=10)
    parser.add_argument('--N_Participants', type=int, default=60)
    parser.add_argument('--Dirichlet_beta', type=float, default=0.5)
    parser.add_argument('--DataPart', type=str2bool, default=True)

    args = parser.parse_args()
    args.device =torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    return args


if __name__ == '__main__':
    args = args_parser()
    # 设置随机种子
    init_seed(args.Seed)

    import torchvision.transforms as transforms
    Singel_Channel_Nor_TRANSFORM = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))])
    train_dataset = torchvision.datasets.MNIST(root=args.Dataset_Dir, train=True, download=True, transform=Singel_Channel_Nor_TRANSFORM)
    test_dataset = torchvision.datasets.MNIST(root=args.Dataset_Dir, train=False, download=True, transform=Singel_Channel_Nor_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 使用狄利克雷分布分割数据集
    client_datasets, client_data_loaders = dirichlet_split_noniid(train_dataset, args.Dirichlet_beta, args.N_Participants)

    from algorithms.FedAvg import FedAvg
    server = FedAvg(args)
    server.ini()
    server.run(client_data_loaders,test_loader)



