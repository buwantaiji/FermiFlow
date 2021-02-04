import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import HO, CoulombPairPotential
from VMC import BetaVMC

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finite-temperature variational Monte Carlo simulation")

    parser.add_argument("--beta", type=float, default=2.0, help="inverse temperature")
    parser.add_argument("--nup", type=int, default=6, help="number of spin-up electrons")
    parser.add_argument("--ndown", type=int, default=0, help="number of spin-down electrons")
    parser.add_argument("--Z", type=float, default=0.5, help="Coulomb interaction strength")

    parser.add_argument("--deltaE", type=float, default=2.0, help="energy cutoff")
    parser.add_argument("--cuda", type=int, default=1, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")
    parser.add_argument("--boltzmann", action="store_true", help="initialize the state probabilities using Boltzmann distribution, otherwise using random Gaussian.")

    parser.add_argument("--baseiter", type=int, default=0, help="base iteration step")
    parser.add_argument("--analyze", action="store_true", help="analyze the data already obtained, instead of computing new iterations")
    parser.add_argument("--iternum", type=int, default=1000, help="number of new iterations")
    parser.add_argument("--batch", type=int, default=8000, help="batch size")
    
    args = parser.parse_args()

    device = torch.device("cuda:%d" % args.cuda)

    orbitals = HO2D()
    basedist = FreeFermion(device=device)

    eta = MLP(1, args.Deta)
    eta.init_zeros()
    if not args.nomu:
        mu = MLP(1, args.Dmu)
        mu.init_zeros()
    else:
        mu = None
    v = Backflow(eta, mu=mu)

    t_span = (args.t0, args.t1)

    cnf = CNF(v, t_span)

    sp_potential = HO()
    pair_potential = CoulombPairPotential(args.Z)

    model = BetaVMC(args.beta, args.nup, args.ndown, args.deltaE, args.boltzmann,
                    orbitals, basedist, cnf, pair_potential, sp_potential=sp_potential)
    model.to(device=device)

    print("beta = %.1f, nup = %d, ndown = %d, Z = %.1f" % (args.beta, args.nup, args.ndown, args.Z))
    print("deltaE = %.1f, total number of states = %d" % (args.deltaE, model.Nstates))
    print("State probabilities initialized with " +
            ("Boltzmann distribution." if args.boltzmann else "random Gaussian."))

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

    checkpoint_dir = "datas/BetaFermionHO2D/init_zeros/" + \
            "beta_%.1f_" % args.beta + \
            "nup_%d_ndown_%d_" % (args.nup, args.ndown) + \
            "deltaE_%.1f_" % args.deltaE + \
           ("boltzmann_" if args.boltzmann else "") + \
           ("cuda_%d_" % device.index if device.type == "cuda" else "cpu_") + \
            "Deta_%d_" % args.Deta + \
            "Dmu_%s_" % (args.Dmu if not args.nomu else None) + \
            "T0_%.1f_T1_%.1f_" % t_span + \
            "batch_%d_" % args.batch + \
            "Z_%.1f/" % args.Z
            
    checkpoint = checkpoint_dir + "iters_%04d.chkp" % args.baseiter 

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
        Ss = states["Ss"]
        Ss_analytical = states["Ss_analytical"]
    else:
        print("Start from scratch...")
        Fs = torch.empty(0, device=device)
        Fs_std = torch.empty(0, device=device)
        Es = torch.empty(0, device=device)
        Es_std = torch.empty(0, device=device)
        Ss = torch.empty(0, device=device)
        Ss_analytical = torch.empty(0, device=device)
    # ==============================================================================

    if args.analyze:
        print("Analyze the data already obtained.")

        from plots import *
        plot_iterations(Fs, Fs_std, Es, Es_std, Ss, Ss_analytical, 
                        savefig=False, savedir=checkpoint_dir)
        
        plot_backflow_potential(model, device, savefig=False, savedir=checkpoint_dir)

        energylevels_batch = 8000
        plot_energylevels(model, energylevels_batch, device, checkpoint_dir, savefig=False)

        density_batch = 800000
        plot_density(model, density_batch, savefig=False, savedir=checkpoint_dir)
    else:
        print("Compute new iterations. batch = %d, iternum = %d." % (args.batch, args.iternum))

        new_Fs = torch.empty(args.iternum, device=device)
        new_Fs_std = torch.empty(args.iternum, device=device)
        new_Es = torch.empty(args.iternum, device=device)
        new_Es_std = torch.empty(args.iternum, device=device)
        new_Ss = torch.empty(args.iternum, device=device)
        new_Ss_analytical = torch.empty(args.iternum, device=device)
        Fs = torch.cat((Fs, new_Fs))
        Fs_std = torch.cat((Fs_std, new_Fs_std))
        Es = torch.cat((Es, new_Es))
        Es_std = torch.cat((Es_std, new_Es_std))
        Ss = torch.cat((Ss, new_Ss))
        Ss_analytical = torch.cat((Ss_analytical, new_Ss_analytical))

        import time
        for i in range(args.baseiter + 1, args.baseiter + args.iternum + 1):
            start = time.time()

            gradF_phi, gradF_theta = model(args.batch)
            optimizer.zero_grad()
            gradF_phi.backward()
            gradF_theta.backward()
            optimizer.step()

            speed = (time.time() - start) * 100 / 3600
            print("iter: %03d" % i, "F:", model.F, "F_std:", model.F_std, 
                                    "E:", model.E, "E_std:", model.E_std, 
                                    "S:", model.S, "S_analytical:", model.S_analytical,
                    "Instant speed (hours per 100 iters):", speed)

            Fs[i - 1] = model.F
            Fs_std[i - 1] = model.F_std
            Es[i - 1] = model.E
            Es_std[i - 1] = model.E_std
            Ss[i - 1] = model.S
            Ss_analytical[i - 1] = model.S_analytical

            nn_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            states = {"nn_state_dict": nn_state_dict, 
                    "optimizer_state_dict": optimizer_state_dict, 
                    "Fs": Fs[:i], 
                    "Fs_std": Fs_std[:i],
                    "Es": Es[:i], 
                    "Es_std": Es_std[:i],
                    "Ss": Ss[:i], 
                    "Ss_analytical": Ss_analytical[:i],
                    }
            checkpoint = checkpoint_dir + "iters_%04d.chkp" % i 
            torch.save(states, checkpoint)
            #print("States saved to the checkpoint file: %s" % checkpoint)
