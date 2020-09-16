import torch 
import torch.nn as nn 
import numpy as np 

class Drift(nn.Module):
    '''
    covariant drift 
    pi@v(x) = v(pi@x)
    '''
    def __init__(self, depth, hidden1, hidden2, n, dim):
        super(Drift, self).__init__()

        self.depth = depth
        self.h1size = hidden1 
        self.h2size = hidden2
        self.fsize = 2*hidden1 + hidden2

        self.n = n
        self.dim = dim

        self.fc1 = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.fc1.append(nn.Linear(2*(dim+1)+(dim+1)+1, self.h1size))
            else: 
                self.fc1.append(nn.Linear(self.fsize, self.h1size))

        self.fc2 = nn.ModuleList()
        for d in range(depth-1):
            if d == 0:
                self.fc2.append(nn.Linear((dim+1)+1, self.h2size))
            else:
                self.fc2.append(nn.Linear(self.h2size, self.h2size))
        self.final = nn.Linear(self.h1size, dim)

        self.activation = nn.Tanh()
    
        for m in self.fc1.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

        for m in self.fc2.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

        for m in self.final.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def set_time(self, t):
        self.t = t 

    def forward(self, x):
        batchsize = x.shape[0]
   
        h1 = self._h1(x)
        h2 = self._h2(x)
    
        for d in range(self.depth-1):
            f = self._combine(h1, h2)
            f = f.view(-1, f.shape[-1])

            if d>0:
                h1_update = self.activation(self.fc1[d](f)).view(batchsize, self.n, -1)
                h2_update = self.activation(self.fc2[d](h2.view(-1, h2.shape[-1]))).view(batchsize, self.n, self.n, -1)

                h1 = h1_update + h1
                h2 = h2_update + h2 
            else:
                ft = torch.cat([f, torch.ones(f.shape[0], 1, dtype=f.dtype, device=f.device)*self.t], dim=1)
                h2t = h2.view(-1, h2.shape[-1])
                h2t = torch.cat([h2t, torch.ones(h2t.shape[0], 1, dtype=h2t.dtype, device=h2t.device)*self.t], dim=1)

                h1_update = self.activation(self.fc1[d](ft)).view(batchsize, self.n, -1)
                h2_update = self.activation(self.fc2[d](h2t)).view(batchsize, self.n, self.n, -1)

                h1 = h1_update 
                h2 = h2_update

        f = self._combine(h1, h2)
        f = f.view(-1, f.shape[-1])
        h1 = self.activation(self.fc1[-1](f)).view(batchsize, self.n, -1) + h1
        h1 = h1.view(-1, h1.shape[-1])

        return self.final(h1).view(batchsize, self.n, self.dim)
    
    def _h1(self, x):
        '''
        [r_i, |r_i|]  
        shape: (b, n, dim+1)
        '''
        h1 = torch.cat([x, x.norm(dim=-1, keepdim=True)], dim=-1)
        return h1 
    
    def _h2(self, x):
        '''
        [r_ij, |r_ij|]
        shape: (b, n, n, dim+1)
        '''
        
        batchsize = x.shape[0]

        rij = x[:, :, None] - x[:, None, :]

        offset = torch.eye(self.n, dtype=x.dtype, device=x.device).view(1, self.n, self.n, 1).expand(batchsize, -1, -1, self.dim)
        rij = rij + offset # constant offset to avoid norm(0) 

        h2 = torch.cat([rij, rij.norm(dim=-1, keepdim=True)], dim=-1)

        return h2 

    def _combine(self, h1, h2):
        '''
        shape: (b, n, 2*h1 + h2)
        '''
        
        f = torch.cat([
        h1, 
        h1.mean(dim=1, keepdim=True).expand(-1, self.n, -1), 
        h2.mean(dim=2)
        ], dim = -1
        )

        return f 

if __name__=='__main__':
    torch.manual_seed(42)
    depth = 4
    hidden1 = 16
    hidden2 = 8
    n = 4 
    dim = 2
    net = Drift(depth, hidden1, hidden2, n, dim).double()
    net.set_time(0.0)
    
    batchsize = 10 
    x = torch.randn(batchsize, n, dim, dtype=torch.float64)
    v = net(x)
    
    perm = torch.randperm(n) 
    print (perm)
    x_perm = x[:, perm, :]
    v_perm = net(x_perm)
    print ((v[:, perm, :] - v_perm).abs().max().item())

