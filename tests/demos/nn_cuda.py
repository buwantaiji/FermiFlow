import torch
torch.set_default_dtype(torch.float64)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        h_tanh = self.activation(self.linear1(x))
        y_pred = self.linear2(h_tanh)
        y_pred += torch.randn(D_out, device=x.device)
        return y_pred

class Net(torch.nn.Module):
    def __init__(self, base):
        super(Net, self).__init__()
        self.base = base

    def forward(self, x):
        return (self.base(x)**2).sum(dim=-1)

if __name__ == "__main__":
    device = torch.device("cuda:1")
    D_in, H, D_out = 1000, 100, 10
    base = TwoLayerNet(D_in, H, D_out)
    net = Net(base)
    net.to(device=device)

    batch = 64
    x = torch.randn(batch, D_in, device=device)
    print("x.shape:", x.shape, "x.device:", x.device)
    y = base(x)
    print("y.shape:", y.shape, "y.device:", y.device)
    z = net(x)
    print("z.shape:", z.shape, "z.device:", z.device)
