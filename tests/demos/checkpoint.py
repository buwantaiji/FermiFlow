import torch
torch.set_default_dtype(torch.float64)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.Tanh()
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        h_tanh = self.activation(self.linear1(x))
        y_pred = self.linear2(h_tanh)
        self.loss = self.loss_fn(y_pred, y)
        return self.loss

def print_infos(nn_state_dict, optimizer_state_dict):
    """
        Print some information about state_dicts of the model and optimizer. 
    """
    print("---- nn_state_dict ----")
    print(nn_state_dict.keys())
    for key, value in nn_state_dict.items():
        print(key, value)

    print("---- optimizer_state_dict ----")
    state = optimizer_state_dict["state"]
    param_group = optimizer_state_dict["param_groups"][0]
    print("state:", state.keys())
    print("(the unique) param_group:", param_group)
    for param, state in state.items():
        print(param, state)

if __name__ == "__main__":
    import os

    batch, D_in, H, D_out = 64, 1000, 100, 10

    torch.manual_seed(42)
    x = torch.randn(batch, D_in)
    y = torch.randn(batch, D_out)

    model = TwoLayerNet(D_in, H, D_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint = "test.chkp"
    if os.path.exists(checkpoint):
        print("Load checkpoint file: %s" % checkpoint)
        states = torch.load(checkpoint)
        model.load_state_dict(states["nn_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
    else:
        print("Start from scratch...")

    niter = 100
    for i in range(niter):
        loss = model(x, y)
        print("iter %03d:" % (i+1), loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i == niter - 1):
            nn_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            #print_infos(nn_state_dict, optimizer_state_dict)
            states = {"nn_state_dict": nn_state_dict, 
                    "optimizer_state_dict": optimizer_state_dict}
            torch.save(states, checkpoint)
