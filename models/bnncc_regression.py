import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 设置torch的默认浮点数
torch.set_default_dtype(torch.float32)

# 设置随机种子
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, skip_layers=2):
        super(ResNetBlock, self).__init__()
        self.skip_layers = skip_layers

        self.fc_layer = nn.ModuleList()
        for i in range(skip_layers):
            self.fc_layer.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))

        self.elu = nn.ELU(inplace=False)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """ 前向传播
        :param x: 初始数据
        """

        residual = x

        for layer in self.fc_layer:
            x = layer(x)
            x = self.elu(x)

        x += self.fc_out(residual)

        return self.elu(x)


class BnnccRegressor(nn.Module):
    def __init__(self, structure_params):
        """ 类的实例化方法
       :param structure_params: 关于因果分组的分支网络结构信息
       """
        super(BnnccRegressor, self).__init__()

        # representation (backbone)
        self.input_layers = nn.ParameterDict()
        self.n_output_nodes = 0  # 因果分组节点数量的初始值

        # 因果分组的每个组分别对应一个分支网络
        for key, value in structure_params.items():
            self.input_layers[key] = nn.ModuleList()
            self.input_layers[key].append(
                nn.Linear(value["input_size"], value["hidden_size"])
            )

            # 确定隐藏层的层数, 第一层的隐藏层已经确定了, 所以需要减1
            for i in range(value["hidden_layers"] - 1):
                self.input_layers[key].append(
                    nn.Linear(value["hidden_size"], value["hidden_size"])
                )

            # This is prediction layer for groups
            self.input_layers[key].append(
                nn.Linear(value["hidden_size"], value["output_size"])
            )

            # 确定因果分组后输出的节点数量
            self.n_output_nodes += value["output_size"]

        # dropout丢弃神经元, 防止过拟合, 必须定义为实例化self的属性, 不然model.eval()无法控制它

        # self.dropout = nn.Dropout(p=0.2)  # IHDP
        # self.dropout = nn.Dropout(p=0.1)  # IHDP

        # self.dropout = nn.Dropout(p=0.2)  # simulation dataset

        # # 网格搜索
        self.dropout = nn.Dropout(p=0.1)  # 0.1, 0.2, 0.3, 0.4

        # 激活函数: inplace=False表示不会在输入张量上原地修改, inplace=True可能导致梯度计算出错, 尤其在反向传播需要保存中间状态时.
        self.elu = nn.ELU(inplace=False)

        # 因果分组后的隐藏层连接
        self.fusion_layers = nn.Linear(self.n_output_nodes, self.n_output_nodes)

        # 残差模块
        self.skip_layers = 2
        self.num_blocks = value["hidden_layers"] // (self.skip_layers + 1)  # //整除, /浮点数除法, %取余数
        self.blocks = nn.ModuleList(
            [ResNetBlock(value["hidden_size"], value["hidden_size"], skip_layers=self.skip_layers) for _ in
             range(self.num_blocks)]
        )

        # 自注意力模块
        self.num_heads = 1  # 注意力头数量
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.n_output_nodes, num_heads=self.num_heads, batch_first=True
        )

        self.n_targets = 1
        self.merge_layer = nn.Linear(self.n_output_nodes, self.n_targets)

    # Define the forward pass
    def forward(self, Xs):
        """ forward propagation/pass
        :param Xs: a dict with keys and values for each input group
        """

        def evaluate_input_layer(input_layer, x):
            """ 将因果分组的批量数据输入到分支网络进行计算
            :param input_layer: ModuleList()
            :param x: 因果分组中一组中的数据
            """

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            # 通过每个残差块
            for block in self.blocks:
                x = block(x)
                x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]
        # neck, 拼接后隐藏层的残差连接

        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        # 注意力前向计算
        x, attention_weights = self.self_attention(x, x, x)
        x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out


def bnncc_reg_loss(predict, target):
    """ 分类任务下损失函数
    :param predict: 模型预测值/估计值, 可以是每个样本的连续值
    :param target: 目标值/真实值
    """

    loss = nn.MSELoss()
    output = loss(predict, target)  # 计算实例化MSELoss类的forward方法，输入是forward()实例方法的输入

    return output


class BnnnccRegressor(BnnccRegressor):
    """无因果约束模块的BNNCC变体，网络保持不变，只要输入不是因果分组，随机分组就能满足无因果约束模块的消融实验"""
    def __init__(self, structure_params):
        super(BnnnccRegressor, self).__init__(structure_params)

    # Define the forward pass
    def forward(self, Xs):
        """ forward propagation/pass
        :param Xs: a dict with keys and values for each input group
        """

        def evaluate_input_layer(input_layer, x):
            """ 将因果分组的批量数据输入到分支网络进行计算
            :param input_layer: ModuleList()
            :param x: 因果分组中一组中的数据
            """

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            # 通过每个残差块
            for block in self.blocks:
                x = block(x)
                x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]
        # neck, 拼接后隐藏层的残差连接

        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        # 注意力前向计算
        x, attention_weights = self.self_attention(x, x, x)
        x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out

class BnnccnrRegressor(BnnccRegressor):
    """无残差模块的BNNCC变体"""
    def __init__(self, structure_params):
        super(BnnccnrRegressor, self).__init__(structure_params)

    # Define the forward pass
    def forward(self, Xs):
        """ forward propagation/pass
        :param Xs: a dict with keys and values for each input group
        """

        def evaluate_input_layer(input_layer, x):
            """ 将因果分组的批量数据输入到分支网络进行计算
            :param input_layer: ModuleList()
            :param x: 因果分组中一组中的数据
            """

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            # # 通过每个残差块
            # for block in self.blocks:
            #     x = block(x)
            #     x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]
        # neck, 拼接后隐藏层的残差连接

        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        # 注意力前向计算
        x, attention_weights = self.self_attention(x, x, x)
        x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out

class BnnccnaRegressor(BnnccRegressor):
    """无注意力机制模块的BNNCC变体"""
    def __init__(self, structure_params):
        super(BnnccnaRegressor, self).__init__(structure_params)

    # Define the forward pass
    def forward(self, Xs):
        """ forward propagation/pass
        :param Xs: a dict with keys and values for each input group
        """

        def evaluate_input_layer(input_layer, x):
            """ 将因果分组的批量数据输入到分支网络进行计算
            :param input_layer: ModuleList()
            :param x: 因果分组中一组中的数据
            """

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            # 通过每个残差块
            for block in self.blocks:
                x = block(x)
                x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]
        # neck, 拼接后隐藏层的残差连接

        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        # # 注意力前向计算
        # x, attention_weights = self.self_attention(x, x, x)
        # x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out


class NncgcRegressor(BnnccRegressor):
    """NN-CGC变体, 应用于IHDP数据集"""
    def __init__(self, structure_params):
        super(NncgcRegressor, self).__init__(structure_params)
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_size = 200
        self.hidden_layers = 3

        # 自注意力模块
        self.num_heads = 1  # 注意力头数量
        self.self_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True)
        # self.self_attention = nn.MultiheadAttention(embed_dim=self.n_output_nodes, num_heads=self.num_heads, batch_first=True)

        self.n_targets = 1
        # self.merge_layer = nn.Linear(self.hidden_size, self.n_targets)
        self.merge_layer = nn.Linear(self.n_output_nodes, self.n_targets)


    # Define the forward pass
    def forward(self, Xs):

        def evaluate_input_layer(input_layer, x):

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            for i in range(1, len(input_layer) - 1):
                x = input_layer[i](x)
                x = self.elu(x)
                if i % 2 == 0:  # 跳跃两层进行神经元丢弃
                    x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]

        # neck, 拼接后隐藏层的残差连接
        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        # x = nn.Linear(self.n_output_nodes, self.hidden_size)(x)
        # x = self.elu(x)
        # x = self.dropout(x)
        #
        # for _ in range(self.hidden_layers-1):
        #     x = nn.Linear(self.hidden_size, self.hidden_size)(x)
        #     x = self.elu(x)
        #     x = self.dropout(x)

        # # 注意力前向计算
        # x, attention_weights = self.self_attention(x, x, x)
        # x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out

class NncgcRegressor2(BnnccRegressor):
    """NN-CGC变体, 应用于合成数据集"""
    def __init__(self, structure_params):
        super(NncgcRegressor2, self).__init__(structure_params)
        self.dropout = nn.Dropout(p=0.2)
        self.hidden_size = 200

        # # nn-cgc: n200_p30_sigma0.01
        # 拼接后没有隐藏层

        # # nn-cgc: n200_p30_sigma0.01
        self.hidden_layers = 3


        self.n_targets = 1
        self.merge_layer = nn.Linear(self.hidden_size, self.n_targets)
        # self.merge_layer = nn.Linear(self.n_output_nodes, self.n_targets)


    # Define the forward pass
    def forward(self, Xs):

        def evaluate_input_layer(input_layer, x):

            # 第一个隐藏层
            x = input_layer[0](x)
            x = self.elu(x)

            for i in range(1, len(input_layer) - 1):
                x = input_layer[i](x)
                x = self.elu(x)
                if i % 2 == 0:  # 跳跃两层进行神经元丢弃
                    x = self.dropout(x)

            # 最后一个隐藏层
            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]

        # neck, 拼接后隐藏层的残差连接
        x = torch.cat(output_of_input_layers, dim=1)  # 按列拼接

        x = nn.Linear(self.n_output_nodes, self.hidden_size)(x)
        x = self.elu(x)
        x = self.dropout(x)

        for _ in range(self.hidden_layers-1):
            x = nn.Linear(self.hidden_size, self.hidden_size)(x)
            x = self.elu(x)
            x = self.dropout(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out