"""
同个特征提取器，经过隐藏层分出两个特征，分别代表不变特征和特定特征，都能正确分类，并且还能回退回原图

"""

from algorithms.base import *
from torchvision.models.resnet import ResNet, BasicBlock
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int = 10, hidden_dim=128, latent_dim=128) -> None:
        super(ResNet_new, self).__init__(block, layers, num_classes)
        self.encoder = nn.Sequential(self.conv1,
                                           self.bn1,
                                           self.relu,
                                           self.maxpool,
                                           self.layer1,
                                           self.layer2,
                                           self.layer3,
                                           self.layer4,
                                           self.avgpool,
                                           nn.Flatten())
        self.decoder = nn.Sequential(
            nn.Linear(512+latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3072),  # 图像大小为 32x32x3
            nn.Sigmoid()  # 将输出限制在 [0, 1] 范围内
        )
        # 定义隐藏层，将特征解耦为两个独立的特征向量
        self.hidden_layer = nn.Sequential(
            nn.Linear(512, hidden_dim * 2),
            nn.ReLU()
        )

        self.classifier1 = nn.Linear(hidden_dim, num_classes)
        self.classifier2 = nn.Linear(hidden_dim, num_classes)
        # 定义域信息，一个可学习的向量
        self.domain_vector = nn.Parameter(torch.randn(latent_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        domain_vector = self.domain_vector.unsqueeze(0).expand(encoded.size(0), -1)
        reconstructed = self.decoder(torch.cat((encoded, domain_vector), dim=1))

        disentangled_features = self.hidden_layer(encoded)
        feature1, feature2 = disentangled_features.chunk(2, dim=1)

        classified1 = self.classifier1(feature1)
        classified2 = self.classifier2(feature2)
        return encoded, reconstructed, classified1, classified2

class FedAug_Client(ClientBase):
    pass

class FedAug_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = "FedAug"


