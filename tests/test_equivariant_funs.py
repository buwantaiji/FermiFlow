import torch
torch.set_default_dtype(torch.float64)

def test_FermiNet():
    from equivariant_funs import FermiNet
    batch, n, dim = 10, 5, 3
    L = 6
    spsize, tpsize = 7, 15

    print("\n---- FermiNet test---")
    print("n = %d, L = %d, dim = %d, spsize = %d, tpsize = %d" % (n, L, dim, spsize, tpsize))

    net = FermiNet(n, dim, L, spsize, tpsize)

    spsize0 = tpsize0 = dim + 1
    for i in range(L):
        if (i == 0):
            fsize = 2 * spsize0 + tpsize0
            print("layer %d: " % (i+1), net.spnet[i].weight.shape, " ", net.tpnet[i].weight.shape)
            assert net.spnet[i].weight.shape == (spsize, fsize)
            assert net.tpnet[i].weight.shape == (tpsize, tpsize0)
        elif (i != L-1):
            fsize = 2 * spsize + tpsize
            print("layer %d: " % (i+1), net.spnet[i].weight.shape, " ", net.tpnet[i].weight.shape)
            assert net.spnet[i].weight.shape == (spsize, fsize)
            assert net.tpnet[i].weight.shape == (tpsize, tpsize)
        else:
            fsize = 2 * spsize + tpsize
            print("layer %d: " % (i+1), net.spnet[i].weight.shape)
            assert net.spnet[i].weight.shape == (spsize, fsize)
    print("final: ", net.final.weight.shape)
    assert net.final.weight.shape == (dim, spsize)

    x = torch.randn(batch, n, dim)
    output = net(x)
    P = torch.randperm(n)
    Px = x[:, P, :] 
    Poutput = net(Px)
    assert torch.allclose(Poutput, output[:, P, :])
