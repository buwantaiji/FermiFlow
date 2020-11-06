import torch
torch.set_default_dtype(torch.float64)

def plot_iterations(Fs, Fs_std, Es, Es_std):
    import numpy as np
    import matplotlib.pyplot as plt

    print("Fs:", Fs)
    print("Es:", Es)
    #print("Fs_std:", Fs_std)
    #print("Es_std:", Es_std)
    assert Fs.shape == Es.shape
    iters, = Es.shape
    print("Number of iterations:", iters)

    iters = np.arange(1, iters + 1)
    Fs_numpy = Fs.to(device=torch.device("cpu")).numpy()
    Es_numpy = Es.to(device=torch.device("cpu")).numpy()
    plt.plot(iters, Fs_numpy, label="$F$")
    plt.plot(iters, Es_numpy, label="$E$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iters")
    plt.ylabel("Observable")
    plt.legend()
    #plt.savefig(checkpoint_dir + "observable.pdf")
    plt.show()

def plot_backflow_potential(eta, mu, device, r_max=20.0):
    import numpy as np
    import matplotlib.pyplot as plt
    r = np.linspace(0., r_max, num=int(r_max * 100))
    eta_r = eta( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
    plt.plot(r, eta_r, label="$\eta(r)$")
    if mu is not None:
        mu_r = mu( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
        plt.plot(r, mu_r, label="$\mu(r)$")
    plt.xlabel("$r$")
    plt.ylabel("Backflow potential")
    plt.title("$\\xi^{e-e}_i = \\sum_{j \\neq i} \\eta(|r_i - r_j|) (r_i - r_j)$" + 
              ("\t\t$\\xi^{e-n}_i = \\mu(|r_i|) r_i$" if mu is not None else ""))
    plt.grid(True)
    plt.legend()
    #plt.savefig(checkpoint_dir + "backflow.pdf")
    plt.show()

if __name__ == "__main__":
    from orbitals import HO2D
    from base_dist import FreeFermion

    from MLP import MLP
    from equivariant_funs import Backflow
    from flow import CNF

    from potentials import HO, CoulombPairPotential
    from VMC import BetaVMC

    beta = 6.0
    nup, ndown = 6, 0
    device = torch.device("cuda:1")

    orbitals = HO2D()
    basedist = FreeFermion(device=device)

    D_hidden_eta = D_hidden_mu = 50
    eta = MLP(1, D_hidden_eta)
    eta.init_zeros()
    mu = MLP(1, D_hidden_mu)
    mu.init_zeros()
    #mu = None
    v = Backflow(eta, mu=mu)

    t_span = (0., 1.)

    cnf = CNF(v, t_span)

    sp_potential = HO()
    Z = 0.5
    pair_potential = CoulombPairPotential(Z)

    model = BetaVMC(beta, nup, ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=sp_potential)
    model.to(device=device)

    """
    z, x = model.sample((8000,))
    for idx, times in model.state_indices_collection.items():
        print("%d:%d" % (idx, times), end=" ")
    print("\nTotal number of samples:", sum(times for times in model.state_indices_collection.values()))
    print(torch.tensor( list(model.state_indices_collection.elements()) ))
    print("z.shape:", z.shape, "x.shape:", x.shape)
    logp = model.logp(x)
    print("logp.shape:", logp.shape)
    exit(111)
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch = 8000
    base_iter = 1000

    checkpoint_dir = "datas/BetaFermionHO2D/init_zeros/" + \
            "beta_%.1f_" % beta + \
            "nup_%d_ndown_%d_" % (nup, ndown) + \
           ("cuda_%d_" % device.index if device.type == "cuda" else "cpu_") + \
            "Deta_%d_" % D_hidden_eta + \
            "Dmu_%s_" % (D_hidden_mu if mu is not None else None) + \
            "T0_%.1f_T1_%.1f_" % t_span + \
            "batch_%d_" % batch + \
            "Z_%.1f/" % Z
            
    checkpoint = checkpoint_dir + "iters_%04d.chkp" % base_iter 

    # ==============================================================================
    # Load the model and optimizer states from a checkpoint file, if any.
    import os
    if os.path.exists(checkpoint):
        print("Load checkpoint file: %s" % checkpoint)
        states = torch.load(checkpoint)
        model.load_state_dict(states["nn_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        Fs = states["Fs"]
        Fs_std = states["Fs_std"]
        Es = states["Es"]
        Es_std = states["Es_std"]
    else:
        print("Start from scratch...")
        Fs = torch.empty(0, device=device)
        Fs_std = torch.empty(0, device=device)
        Es = torch.empty(0, device=device)
        Es_std = torch.empty(0, device=device)

    plot_iterations(Fs, Fs_std, Es, Es_std)
    
    eta, mu = model.cnf.backflow_potential()
    plot_backflow_potential(eta, mu, device)
    exit(0)
    # ==============================================================================

    print("batch =", batch)
    iter_num = 1000
    print("iter_num:", iter_num)

    new_Fs = torch.empty(iter_num, device=device)
    new_Fs_std = torch.empty(iter_num, device=device)
    new_Es = torch.empty(iter_num, device=device)
    new_Es_std = torch.empty(iter_num, device=device)
    Fs = torch.cat((Fs, new_Fs))
    Fs_std = torch.cat((Fs_std, new_Fs_std))
    Es = torch.cat((Es, new_Es))
    Es_std = torch.cat((Es_std, new_Es_std))

    import time
    for i in range(base_iter + 1, base_iter + iter_num + 1):
        start = time.time()

        gradF_phi, gradF_theta = model(batch)
        optimizer.zero_grad()
        gradF_phi.backward()
        gradF_theta.backward()
        optimizer.step()

        speed = (time.time() - start) * 100 / 3600
        print("iter: %03d" % i, "F:", model.F, "F_std:", model.F_std, 
                                "E:", model.E, "E_std:", model.E_std, 
                "Instant speed (hours per 100 iters):", speed)

        Fs[i - 1] = model.F
        Fs_std[i - 1] = model.F_std
        Es[i - 1] = model.E
        Es_std[i - 1] = model.E_std

        nn_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        states = {"nn_state_dict": nn_state_dict, 
                "optimizer_state_dict": optimizer_state_dict, 
                "Fs": Fs[:i], 
                "Fs_std": Fs_std[:i],
                "Es": Es[:i], 
                "Es_std": Es_std[:i],
                }
        checkpoint = checkpoint_dir + "iters_%04d.chkp" % i 
        torch.save(states, checkpoint)
        #print("States saved to the checkpoint file: %s" % checkpoint)
