import torch
torch.set_default_dtype(torch.float64)

from orbitals import HO2D
from base_dist import FreeFermion

from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF

from potentials import HO, CoulombPairPotential
from VMC import GSVMC

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ground-state variational Monte Carlo simulation")

    parser.add_argument("--nup", type=int, default=6, help="number of spin-up electrons")
    parser.add_argument("--ndown", type=int, default=0, help="number of spin-down electrons")
    parser.add_argument("--Z", type=float, default=0.5, help="Coulomb interaction strength")

    parser.add_argument("--cuda", type=int, default=1, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")

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

    model = GSVMC(args.nup, args.ndown, orbitals, basedist, cnf, 
                    pair_potential, sp_potential=sp_potential)
    model.to(device=device)
    print("nup = %d, ndown = %d, Z = %.1f" % (args.nup, args.ndown, args.Z))


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    checkpoint_prefix = "/data1/xieh/FlowVMC/master/FermionHO2D/init_zeros/"
    data_dir = "nup_%d_ndown_%d_" % (args.nup, args.ndown) + \
              ("cuda_%d_" % device.index if device.type == "cuda" else "cpu_") + \
               "Deta_%d_" % args.Deta + \
               "Dmu_%s_" % (args.Dmu if not args.nomu else None) + \
               "T0_%.1f_T1_%.1f_" % t_span + \
               "batch_%d_" % args.batch + \
               "Z_%.1f/" % args.Z
    checkpoint_dir = checkpoint_prefix + data_dir
    checkpoint = checkpoint_dir + "iters_%04d.chkp" % args.baseiter 

    # ==============================================================================
    # Load the model and optimizer states from a checkpoint file, if any.
    import os
    if os.path.exists(checkpoint):
        print("Load checkpoint file: %s" % checkpoint)
        states = torch.load(checkpoint)
        model.load_state_dict(states["nn_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        Es = states["Es"]
        Es_std = states["Es_std"]
    else:
        print("Start from scratch...")
        Es = torch.empty(0, device=device)
        Es_std = torch.empty(0, device=device)
    # ==============================================================================

    if args.analyze:
        print("Analyze the data already obtained.")
        from plots import *
        savedir = "datas/FermionHO2D/init_zeros/" + data_dir
        if not os.path.exists(savedir): os.makedirs(savedir)

        plot_iterations_GS(Es, Es_std, savefig=True, savedir=savedir)
        plot_backflow_potential(model, device, savefig=True, savedir=savedir)
    else:
        print("Compute new iterations. batch = %d, iternum = %d." % (args.batch, args.iternum))

        new_Es = torch.empty(args.iternum, device=device)
        new_Es_std = torch.empty(args.iternum, device=device)
        Es = torch.cat((Es, new_Es))
        Es_std = torch.cat((Es_std, new_Es_std))

        import time
        for i in range(args.baseiter + 1, args.baseiter + args.iternum + 1):
            start = time.time()

            gradE = model(args.batch)
            optimizer.zero_grad()
            gradE.backward()
            optimizer.step()

            speed = (time.time() - start) * 100 / 3600
            print("iter: %03d" % i, "E:", model.E, "E_std:", model.E_std, 
                    "Instant speed (hours per 100 iters):", speed)

            Es[i - 1] = model.E
            Es_std[i - 1] = model.E_std

            nn_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            states = {"nn_state_dict": nn_state_dict, 
                    "optimizer_state_dict": optimizer_state_dict, 
                    "Es": Es[:i], 
                    "Es_std": Es_std[:i],
                    }
            checkpoint = checkpoint_dir + "iters_%04d.chkp" % i 
            torch.save(states, checkpoint)
