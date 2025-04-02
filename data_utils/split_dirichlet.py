import torch
from torch.utils.data import DataLoader, Subset
from torch.distributions.dirichlet import Dirichlet


def dirichlet_split_noniid(train_dataset, alpha, n_clients):
    train_labels = train_dataset.targets
    n_classes = train_labels.max() + 1  # 获取类别总数
    label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))  # 生成狄利克雷分布
    class_idcs = [torch.nonzero(train_labels == y).flatten() for y in range(n_classes)]  # 获取每个类别的索引
    client_idcs = [[] for _ in range(n_clients)]  # 初始化客户端索引列表

    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()  # 根据分布计算每个客户端的样本数量
        splits[-1] = total_size - splits[:-1].sum()  # 修正最后一个客户端的样本数量
        idcs = torch.split(c, splits.tolist())  # 按照计算的样本数量分割索引
        for i, idx in enumerate(idcs):
            client_idcs[i] += idx.tolist()  # 将索引分配给对应的客户端


    # 创建客户端数据加载器
    client_datasets = [Subset(train_dataset, idcs) for idcs in client_idcs]
    client_data_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in client_datasets]
    return client_datasets, client_data_loaders


if __name__ == '__main__':
    from data_loader import *
    # 加载 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(root='../../Dataset', train=True, download=True, transform=transform)

    # 设置参数
    n_clients = 10  # 客户端数量
    alpha = 0.5  # 狄利克雷分布的参数，控制数据分布的非独立同分布程度

    # 使用狄利克雷分布分割数据集
    client_datasets, client_data_loaders = dirichlet_split_noniid(train_dataset, alpha, n_clients)


    # 打印每个客户端的数据分布情况
    for i, dataset in enumerate(client_datasets):
        labels = [train_dataset.targets[idx].item() for idx in dataset.indices]
        print(f"Client {i} has {len(labels)} samples with distribution: {torch.bincount(torch.tensor(labels))}")