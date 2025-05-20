"""
FedPAC《Personalized Federated Learning with Feature Alignment and Classifier Collaboration》 ICLR2023
分成特征提取器和分类器两部分，全局平均特征提取器，分类器用二次规划重新计算，个性化分类器
"""
import cvxpy as cvx
from algorithms.base import *
from algorithms.FedProto import local_proto_aggregation, global_proto_aggregation
from torchvision.models.resnet import ResNet, BasicBlock
class ResNet_new(ResNet):
    def __init__(self, block: BasicBlock, layers: List[int], num_classes: int = 10) -> None:
        super(ResNet_new, self).__init__(block, layers, num_classes)
        self.feature_extra = nn.Sequential(self.conv1,
                                           self.bn1,
                                           self.relu,
                                           self.maxpool,
                                           self.layer1,
                                           self.layer2,
                                           self.layer3,
                                           self.layer4,
                                           self.avgpool,
                                           nn.Flatten())
        self.head = self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_extra(x)
        out = self.head(feature)
        return feature, out
def MYNET(num_classes: int):
    return ResNet_new(BasicBlock, [2, 2, 2, 2], num_classes)


class FedPAC_Client(ClientBase):
    def __init__(self, args, id, train_loader):
        super(FedPAC_Client, self).__init__(args, id, train_loader)
        self.train_samples = sum(args.clients_labelnums[id])
        self.V, self.h = self.statistics_extraction()

    def ini(self, model_name=None):
        self.model = MYNET(self.num_classes)
        self.model.to(self.device)

    def train(self, global_protos):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        for name, param in self.model.named_parameters():  # 冻结 特征提取层 的权重
            if 'feature_extra' in name:
                param.requires_grad = False
            if 'head' in name:
                param.requires_grad = True
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            features, outputs = self.model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for name, param in self.model.named_parameters():# 冻结 分类器 的权重
            if 'feature_extra' in name:
                param.requires_grad = True
            if 'head' in name:
                param.requires_grad = False
        for epoch in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                features, outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                if len(global_protos) != 0:
                    f_new = copy.deepcopy(features.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_protos.keys():
                            f_new[i, :] = global_protos[label.item()][0].data
                        i += 1
                    loss += nn.MSELoss()(f_new, features) * self.mu
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.local_protos = local_proto_aggregation(self.model, self.train_loader, self.device)
        self.V, self.h = self.statistics_extraction()
        return None

    def statistics_extraction(self):
        model = self.model
        for x, y in self.train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = model.base(x).detach()
            break
        d = rep.shape[1]
        feature_dict = {}
        with torch.no_grad():
            for x, y in self.train_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                features = model.base(x)
                feat_batch = features.clone().detach()
                for i in range(len(y)):
                    yi = y[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = torch.zeros(self.num_classes)
        for x, y in self.train_loader:
            for yy in y:
                py[yy.item()] += 1
        py = py / torch.sum(py)
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, d), device=self.device)
        for k in range(self.num_classes):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / self.train_samples

        return v, h_ref

    def receive_model(self, global_model):
        for name, new_param in global_model.named_parameters():
            if name.startswith('feature_extra'):
                old_param = self.model.get_parameter(name)
                if old_param is not None:
                    old_param.data = new_param.data.clone()
                else:
                    raise ValueError("未定义的模型参数{}".format(name))

    def receive_head(self, new_head):
        for new_param, old_param in zip(new_head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()


class FedPAC_Server(ServerBase):
    def __init__(self, args):
        super().__init__(args)
        self.name = 'FedPAC'
        self.aggregate_mode = 'weights'
        self.global_protos = []
        self.Vars = []
        self.Hs = []

    def ini(self, client_data_loaders):
        self.write_settings()
        for idx in range(self.args.N_Participants):
            self.clients.append(FedPAC_Client(self.args, idx, client_data_loaders[idx]))
            if len(self.args.Nets_Name_List)==1:
                self.clients[idx].ini(self.args.Nets_Name_List[0])
            else:
                self.clients[idx].ini(self.args.Nets_Name_List[idx])
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.global_model.to(self.device)


    def global_update(self):
        self.aggregate_nets()

        local_protos_list = []
        for idx in tqdm(self.clients_num_choice):
            local_protos_list.append(self.clients[idx].local_protos)
        self.global_protos = global_proto_aggregation(local_protos_list)

        Vars, Hs, uploaded_heads = [], [], []
        for idx in self.clients_num_choice:
            Vars.append(self.clients[idx].V)
            Hs.append(self.clients[idx].h)
            uploaded_heads.append(self.clients[idx].model.head)
        head_weights = solve_quadratic(len(self.clients_num_choice), Vars, Hs)
        for idx in tqdm(self.clients_num_choice):
            new_head = copy.deepcopy(uploaded_heads[idx])
            if head_weights[idx] is not None:
                for param in new_head.parameters():
                    param.data.zero_()
                for w, head in zip(head_weights, uploaded_heads):
                    for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                        server_param.data += client_param.data.clone() * w
            self.clients[idx].receive_head(new_head)
        return None

    def local_update(self):
        for idx in tqdm(self.clients_num_choice):
            self.clients[idx].receive_model(self.global_model)
            self.clients[idx].train(self.global_protos)
        return None


# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0]  # number of classes
    d = Hs[0].shape[1]  # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k] - h_j1[k]).reshape(d, 1), (h_ref[k] - h_j2[k]).reshape(1, d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))

        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii].real >= 0.01:
                p_matrix_new += evals[ii].real * torch.mm(evecs[:, ii].reshape(num_users, 1),
                                                          evecs[:, ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix

        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i) * (i > eps) for i in alpha]  # zero-out small weights (<eps)
        else:
            alpha = None  # if no solution for the optimization problem, use local classifier only

        avg_weight.append(alpha)

    return avg_weight

# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

