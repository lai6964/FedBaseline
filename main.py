import argparse
import torch, torchvision
from utils import *
from data_utils.split_dirichlet import dirichlet_split_noniid


def args_parser():
    parser = argparse.ArgumentParser()
    '''    Default Setting    '''
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--Seed',type=int,default=42)
    parser.add_argument('--Dataset_Dir',type=str,default='../Dataset/')
    parser.add_argument('--Model_Path',type=str,default='Model_Storage')
    parser.add_argument('--model', type=str, default='FedAvg', help='Model name.')
    parser.add_argument('--Scenario',type=str,default='MNIST')


    '''    Traning Setting    '''
    parser.add_argument('--CommunicationEpoch',type=int,default=1000)
    parser.add_argument('--features_dim',type=int,default=512)
    parser.add_argument('--public_epoch',type=int,default=1)
    parser.add_argument('--local_epoch',type=int,default=10)
    parser.add_argument('--public_lr',type=float,default=0.01)
    parser.add_argument('--local_lr',type=float,default=0.01)
    parser.add_argument('--local_batch_size', type=int, default=256)
    parser.add_argument('--public_batch_size', type=int, default=64)
    parser.add_argument('--ReLoad', type=str2bool, default=False)
    parser.add_argument('--lrschedule', type=str2bool, default=False)
    parser.add_argument('--N_Participants', type=int, default=100)
    parser.add_argument('--clients_select_ratio',type=float,default=0.1)
    parser.add_argument('--Nets_Name_List',type=list,default=['ResNet18'])
    parser.add_argument('--eval_epoch_gap', type=int, default=1)
    parser.add_argument('--using_multi_thread', type=str2bool, default=False)

    '''    Data Setting    '''
    parser.add_argument('--Public_Dataset_Name', type=str, default='cifar_100')
    parser.add_argument('--public_len', type=int, default=5000)
    parser.add_argument('--pub_aug', type=str, default='weak')
    parser.add_argument('--N_Class', type=int, default=10)
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

    CON_TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         # transforms.RandomHorizontalFlip(),
         # transforms.RandomApply([
         #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
         # ], p=0.8),
         # transforms.RandomGrayscale(p=0.2),
         transforms.ToTensor(),
         transforms.Normalize((0.4802, 0.4480, 0.3975),
                              (0.2770, 0.2691, 0.2821))])

    # train_dataset = torchvision.datasets.MNIST(root=args.Dataset_Dir, train=True, download=True, transform=Singel_Channel_Nor_TRANSFORM)
    # test_dataset = torchvision.datasets.MNIST(root=args.Dataset_Dir, train=False, download=True, transform=Singel_Channel_Nor_TRANSFORM)
    train_dataset = torchvision.datasets.CIFAR10(root=args.Dataset_Dir, train=True, download=True, transform=CON_TRANSFORM)
    test_dataset = torchvision.datasets.CIFAR10(root=args.Dataset_Dir, train=False, download=True, transform=CON_TRANSFORM)
    # 使用狄利克雷分布分割数据集
    client_datasets = dirichlet_split_noniid(train_dataset, args.Dirichlet_beta, args.N_Participants)
    client_data_loaders = [torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True) for dataset in client_datasets]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.local_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)



    clients_labelnums = []
    for i in range(args.N_Participants):
        dataset = client_datasets[i]
        datasetloader = client_data_loaders[i]
        labels = [train_dataset.targets[idx] for idx in dataset.indices]
        print(f"Client {i} has {len(labels)} samples with distribution: {torch.bincount(torch.tensor(labels), minlength=args.N_Class)},"
              f" total {len(dataset)}=={len(datasetloader.dataset.indices)}")
        clients_labelnums.append(torch.bincount(torch.tensor(labels), minlength=args.N_Class).tolist())
    args.clients_labelnums = clients_labelnums

    # args.model = "FedRep"
    if args.model == "FedAvg":
        from algorithms.FedAvg import FedAvg_Server
        server = FedAvg_Server(args)
    elif args.model == "FedProx":
        from algorithms.FedProx import FedProx_Server
        server = FedProx_Server(args)
    elif args.model == "FedProto": # failed, 20% in Cifar10 - dirichlet_split(0.5) - NP100(0.1)
        from algorithms.FedProto import FedProto_Server
        server = FedProto_Server(args)
    elif args.model == "FedProc":
        from algorithms.FedProc import FedProc_Server
        server = FedProc_Server(args)
    elif args.model == "FPL":
        from algorithms.FPL import FPL_Server
        server = FPL_Server(args)
    elif args.model == "FedRep":
        from algorithms.FedRep import FedRep_Server
        server = FedRep_Server(args)
    elif args.model == "FedRoD":
        from algorithms.FedRoD import FedRoD_Server
        server = FedRoD_Server(args)
    elif args.model == "FedPAC":
        from algorithms.FedPAC import FedPAC_Server
        server = FedPAC_Server(args)

    server.ini(client_data_loaders)
    server.run(test_loader)


