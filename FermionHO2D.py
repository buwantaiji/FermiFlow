import torch
torch.set_default_dtype(torch.float64)

from flow import CNF

def plot_iterations(Es, Es_std):
    import numpy as np
    import matplotlib.pyplot as plt

    print("Es:", Es)
    #print("Es_std:", Es_std)
    iters, = Es.shape
    print("Number of iterations:", iters)

    Es_numpy = Es.to(device=torch.device("cpu")).numpy()
    iters = np.arange(1, iters + 1)
    plt.plot(iters, Es_numpy)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iters")
    plt.ylabel("$E$")
    #plt.savefig(checkpoint_dir + "Es.pdf")
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
    from base_dist import FreeFermionHO2D
    from MLP import MLP
    from equivariant_funs import Backflow
    from potentials import HO, CoulombPairPotential

    nup, ndown = 6, 0
    device = torch.device("cuda:1")

    basedist = FreeFermionHO2D(nup, ndown, device=device)

    D_hidden_eta = D_hidden_mu = 50
    eta = MLP(1, D_hidden_eta)
    eta.init_zeros()
    mu = MLP(1, D_hidden_mu)
    mu.init_zeros()
    #mu = None
    v = Backflow(eta, mu=mu)

    t_span = (0., 1.)

    sp_potential = HO()
    Z = 0.5
    pair_potential = CoulombPairPotential(Z)

    cnf = CNF(basedist, v, t_span, pair_potential, sp_potential=sp_potential)
    cnf.to(device=device)
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-2)

    batch = 8000
    base_iter = 0

    checkpoint_dir = "datas/FermionHO2D/init_zeros/" + \
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
        cnf.load_state_dict(states["nn_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        Es = states["Es"]
        Es_std = states["Es_std"]
    else:
        print("Start from scratch...")
        Es = torch.empty(0, device=device)
        Es_std = torch.empty(0, device=device)

    #plot_iterations(Es, Es_std)
    
    #eta, mu = cnf.backflow_potential()
    #plot_backflow_potential(eta, mu, device)
    #exit(0)
    # ==============================================================================

    print("batch =", batch)
    iter_num = 1000
    print("iter_num:", iter_num)

    new_Es = torch.empty(iter_num, device=device)
    new_Es_std = torch.empty(iter_num, device=device)
    Es = torch.cat((Es, new_Es))
    Es_std = torch.cat((Es_std, new_Es_std))

    import time
    for i in range(base_iter + 1, base_iter + iter_num + 1):
        start = time.time()

        gradE = cnf(batch)
        optimizer.zero_grad()
        gradE.backward()
        optimizer.step()

        speed = (time.time() - start) * 100 / 3600
        print("iter: %03d" % i, "E:", cnf.E, "E_std:", cnf.E_std, 
                "Instant speed (hours per 100 iters):", speed)

        Es[i - 1] = cnf.E
        Es_std[i - 1] = cnf.E_std

        nn_state_dict = cnf.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        states = {"nn_state_dict": nn_state_dict, 
                "optimizer_state_dict": optimizer_state_dict, 
                "Es": Es[:i], 
                "Es_std": Es_std[:i],
                }
        checkpoint = checkpoint_dir + "iters_%04d.chkp" % i 
        torch.save(states, checkpoint)
        #print("States saved to the checkpoint file: %s" % checkpoint)
