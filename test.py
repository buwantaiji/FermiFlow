import torch
torch.set_default_dtype(torch.float64)

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        #ctx.needs_input_grad = (True, False)
        ctx.save_for_backward(x, y)
        return x**2 + y**3

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        gradx = grad_output * 2 * x if ctx.needs_input_grad[0] else None
        grady = grad_output * 3 * y**2 if ctx.needs_input_grad[1] else None
        return gradx, grady

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.fc1 = torch.nn.Linear(20, 30)
        self.fc2 = torch.nn.Linear(30, 5)
        self.activation = torch.nn.Tanh()

    def forward(self, x):
        parameters = tuple(self.parameters())
        print(len(parameters))
        for parameter in parameters:
            print(parameter)
        return self.fc2(self.activation(self.fc1(x)))

def add(nonsense, *nums, power=1):
    return sum(num ** power for num in nums)

if __name__ == "__main__":
    x = torch.randn(1, requires_grad=True)
    y = torch.randn(1, requires_grad=True)
    loss = Square.apply(x, y)
    print(x, y)
    print(loss)

    #gradx = torch.autograd.grad(loss, x)
    #print("gradx:", gradx, 2 * x)
    #gradx, grady = torch.autograd.grad(loss, (x, y))
    #print("gradx, grady:", gradx, grady, 2 * x, 3 * y**2)

    """
    model = Test()
    x = torch.randn(100, 20)
    print(model(x).shape)
    """

    """
    print(add(None, 1, 2, 3, 4, power=4))
    """
