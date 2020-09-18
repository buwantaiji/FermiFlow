import torch
torch.set_default_dtype(torch.float64)

class MLP(torch.nn.Module):
    """
        A MLP with single hidden layer, whose output is set to be a scalar. The 
    gradient with respect to the input is handcoded for further convenience.
    """
    def __init__(self, D_in, D_hidden):
        """
            D_in: input dimension;  D_hidden: hidden layer dimension.
        """
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, D_hidden)
        self.fc2 = torch.nn.Linear(D_hidden, 1, bias=False)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.fc2(self.activation(self.fc1(x)))
        return output

    def d_sigmoid(self, output):
        return output * (1. - output)

    def grad(self, x):
        """
            Note that this implementation of grad works for the general case
        where x has ANY batch dimension, i.e., x has shape (..., D_in).
        """
        fc1_activation = self.activation(self.fc1(x))
        grad_fc1 = self.fc2.weight * self.d_sigmoid(fc1_activation)
        grad_x = grad_fc1.matmul(self.fc1.weight)
        return grad_x
