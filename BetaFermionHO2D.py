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
    parser.add_argument("--cuda", type=int, default=0, help="GPU device number")
    parser.add_argument("--Deta", type=int, default=50, help="hidden layer size in the MLP representation of two-body backflow potential eta")
    parser.add_argument("--nomu", action="store_true", help="do not use the one-body backflow potential mu")
    parser.add_argument("--Dmu", type=int, default=50, help="hidden layer size in the MLP representation of one-body backflow potential mu")
    parser.add_argument("--t0", type=float, default=0.0, help="starting time")
    parser.add_argument("--t1", type=float, default=1.0, help="ending time")
    parser.add_argument("--boltzmann", action="store_true", help="initialize the state probabilities using Boltzmann distribution, otherwise using random Gaussian.")

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print("batch = %d, iternum = %d." % (args.batch, args.iternum))

    import time
    for i in range(1, args.iternum + 1):
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
