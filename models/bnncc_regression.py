import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

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
        residual = x

        for layer in self.fc_layer:
            x = layer(x)
            x = self.elu(x)

        x += self.fc_out(residual)

        return self.elu(x)


class BnnccRegressor(nn.Module):
    def __init__(self, structure_params):
        super(BnnccRegressor, self).__init__()

        # representation (backbone)
        self.input_layers = nn.ParameterDict()
        self.n_output_nodes = 0
        for key, value in structure_params.items():
            self.input_layers[key] = nn.ModuleList()
            self.input_layers[key].append(
                nn.Linear(value["input_size"], value["hidden_size"])
            )

            for i in range(value["hidden_layers"] - 1):
                self.input_layers[key].append(
                    nn.Linear(value["hidden_size"], value["hidden_size"])
                )

            self.input_layers[key].append(
                nn.Linear(value["hidden_size"], value["output_size"])
            )
            self.n_output_nodes += value["output_size"]

        self.dropout = nn.Dropout(p=0.1)  # 0.1, 0.2, 0.3, 0.4
        self.elu = nn.ELU(inplace=False)
        self.fusion_layers = nn.Linear(self.n_output_nodes, self.n_output_nodes)
        self.skip_layers = 2
        self.num_blocks = value["hidden_layers"] // (self.skip_layers + 1)
        self.blocks = nn.ModuleList(
            [ResNetBlock(value["hidden_size"], value["hidden_size"], skip_layers=self.skip_layers) for _ in
             range(self.num_blocks)]
        )
        self.num_heads = 1
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
            x = input_layer[0](x)
            x = self.elu(x)

            for block in self.blocks:
                x = block(x)
                x = self.dropout(x)

            x = input_layer[-1](x)
            x = self.elu(x)

            return x

        # backbone
        output_of_input_layers = [evaluate_input_layer(self.input_layers[k], Xs[k]) for k in Xs.keys()]

        # neck
        x = torch.cat(output_of_input_layers, dim=1)
        x, attention_weights = self.self_attention(x, x, x)
        x = self.elu(x)

        # head
        x = self.merge_layer(x)
        out = x

        return out


def bnncc_reg_loss(predict, target):
    loss = nn.MSELoss()
    output = loss(predict, target)

    return output