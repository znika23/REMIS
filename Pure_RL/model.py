import torch
import torch.nn.init as init

class QNet(torch.nn.Module):
    """
    This class defines the Q network.
    """
    def __init__(self, parameters, dims, dima) -> None:
        super().__init__()
        self.dims, self.dima = dims, dima
        self.n1_feature, self.n2_feature = parameters['n1_feature'], parameters['n2_feature']
        self.linear1 = torch.nn.Linear(self.dims, self.n1_feature)
        self.linear2 = torch.nn.Linear(self.n1_feature, self.n2_feature)
        self.linear3 = torch.nn.Linear(self.n2_feature, self.dima)
        self.activation = torch.nn.LeakyReLU()

        # random initialization
        init.kaiming_uniform_(self.linear1.weight, nonlinearity='leaky_relu', a=0.01)
        init.kaiming_uniform_(self.linear2.weight, nonlinearity='leaky_relu', a=0.01)
        init.kaiming_uniform_(self.linear3.weight, nonlinearity='leaky_relu', a=0.01)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()
        self.linear3.bias.data.zero_()

    def forward(self, s):
        y = self.linear1(s)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y