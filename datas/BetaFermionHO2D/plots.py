import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import matplotlib.pyplot as plt
import os

from orbitals import HO2D
from base_dist import FreeFermion
from MLP import MLP
from equivariant_funs import Backflow
from flow import CNF
from potentials import HO, CoulombPairPotential
from VMC import BetaVMC

def load_model(beta, nup, ndown, Z,
        deltaE, cuda, Deta, nomu, Dmu, t0, t1, boltzmann,
        baseiter, batch):
    device = torch.device("cuda:%d" % cuda)
    orbitals = HO2D()
    basedist = FreeFermion(device=device)

    eta = MLP(1, Deta)
    eta.init_zeros()
    if not nomu:
        mu = MLP(1, Dmu)
        mu.init_zeros()
    else:
        mu = None
    v = Backflow(eta, mu=mu)

    t_span = (t0, t1)
    cnf = CNF(v, t_span)
    sp_potential = HO()
    pair_potential = CoulombPairPotential(Z)

    model = BetaVMC(beta, nup, ndown, deltaE, boltzmann,
                    orbitals, basedist, cnf, pair_potential, sp_potential=sp_potential)
    model.to(device=device)

    print("beta = %.1f, nup = %d, ndown = %d, Z = %.1f" % (beta, nup, ndown, Z))
    print("deltaE = %.1f, total number of states = %d" % (deltaE, model.Nstates))
    print("State probabilities initialized with " +
            ("Boltzmann distribution." if boltzmann else "random Gaussian."))

    checkpoint_dir = "init_zeros/" + \
            "beta_%.1f_" % beta + \
            "nup_%d_ndown_%d_" % (nup, ndown) + \
            "deltaE_%.1f_" % deltaE + \
           ("boltzmann_" if boltzmann else "") + \
           ("cuda_%d_" % device.index if device.type == "cuda" else "cpu_") + \
            "Deta_%d_" % Deta + \
            "Dmu_%s_" % (Dmu if not nomu else None) + \
            "T0_%.1f_T1_%.1f_" % t_span + \
            "batch_%d_" % batch + \
            "Z_%.1f/" % Z
            
    checkpoint = checkpoint_dir + "iters_%04d.chkp" % baseiter 

    if os.path.exists(checkpoint):
        print("Load checkpoint file: %s" % checkpoint)
        states = torch.load(checkpoint)
        model.load_state_dict(states["nn_state_dict"])
        return model, states, checkpoint_dir
    else:
        raise ValueError("Checkpoint file does not exist: %s" % checkpoint)

def plot_iterations(Fs, Fs_std, Es, Es_std, Ss, Ss_analytical, ax_entropy_iterations, label):
    #print("Fs:", Fs)
    #print("Fs_std:", Fs_std)
    print("Es:", Es)
    #print("Es_std:", Es_std)
    print("entropy (analytical):", Ss_analytical)

    assert Fs.shape == Es.shape
    iters, = Es.shape
    print("Number of iterations:", iters)

    iters = np.arange(1, iters + 1)
    
    Ss_analytical_numpy = Ss_analytical.to(device=torch.device("cpu")).numpy()
    ax_entropy_iterations.plot(iters, Ss_analytical_numpy, label=label)

def entropy_from_flow(beta, batch, checkpoint_dir):
    from torch.distributions.categorical import Categorical

    filename = checkpoint_dir + "energylevels_batch_%d.pt" % batch
    if os.path.exists(filename):
        print("Load energy level data file: %s" % filename)
        energies = torch.load(filename)
        Es_flow = energies["Es_flow"]
        dist = Categorical(logits=-beta * Es_flow)
        return dist.entropy().item()
    else:
        raise ValueError("Energy level data file does not exist: %s" % filename)

def plot_backflow_potential(model, cuda, ax, color, eta_label, mu_label, r_max=20.0):
    eta, mu = model.cnf.backflow_potential()
    r = np.linspace(0., r_max, num=int(r_max * 100))
    device = torch.device("cuda:%d" % cuda)
    eta_r = eta( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
    ax.plot(r, eta_r, color=color, label=eta_label)
    if mu is not None:
        mu_r = mu( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
        ax.plot(r, mu_r, "--", color=color, label=mu_label)

def plot_density(ax1, ax2, x, label, rmax=5.0, bins=300):
    x = x.cpu().numpy()
    rs = np.linalg.norm(x, axis=-1)
    n = rs.shape[-1]
    hist, bin_edges = np.histogram(rs, bins=bins, range=(0.0, rmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    ax1.plot(bin_centers, n * hist, label=label)
    ax2.plot(bin_centers, n * hist / (2 * np.pi * bin_centers), label=label)

if __name__ == "__main__":
    beta = 10.0
    #nup, ndown = 3, 0
    #Zs = (2.0, 4.0, 6.0, 8.0)
    #cudas = (0, 1, 2, 3)
    #plot_indices = (0, 1, 2, 3)
    #nup, ndown = 4, 0
    #Zs = (2.0, 4.0, 6.0, 8.0)
    #cudas = (6, 6, 4, 7)
    #plot_indices = (0, 1, 2, 3)
    nup, ndown = 6, 0
    Zs = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    cudas = (1, 4, 5, 6, 0, 1, 2, 3, 4, 7)
    plot_indices = (1, 3, 5, 7, 9)

    deltaE = 2.0
    boltzmann = True
    Deta = 50
    nomu = False
    Dmu = 50
    t0, t1 = 0., 1.
    batch = 8000
    baseiter = 3000

    fig_entropy_iterations = plt.figure()
    ax_entropy_iterations = fig_entropy_iterations.add_subplot(111)
    fig_backflow = plt.figure()
    ax_backflow = fig_backflow.add_subplot(111)
    fig_density1 = plt.figure()
    ax_density1 = fig_density1.add_subplot(111)
    fig_density2 = plt.figure()
    ax_density2 = fig_density2.add_subplot(111)

    thermo_quantity_labels = ("Z", "F", "F_std", "E", "E_std", "S", "S_analytical", "S_flow")
    thermo_quantities = []
    energylevels_batch = 8000
    density_batch = 800000
    for idx, Z, cuda in zip(range(len(Zs)), Zs, cudas):
        model, states, checkpoint_dir = load_model(beta, nup, ndown, Z,
                deltaE, cuda, Deta, nomu, Dmu, t0, t1, boltzmann, baseiter, batch)
        Fs, Fs_std, Es, Es_std, Ss, Ss_analytical = states["Fs"], states["Fs_std"], \
                states["Es"], states["Es_std"], states["Ss"], states["Ss_analytical"]

        F, F_std, E, E_std, S, S_analytical = Fs[-1].item(), Fs_std[-1].item(), \
                Es[-1].item(), Es_std[-1].item(), Ss[-1].item(), Ss_analytical[-1].item()
        S_flow = entropy_from_flow(beta, energylevels_batch, checkpoint_dir)
        thermo_quantities.append(np.array([Z, F, F_std, E, E_std, S, S_analytical, S_flow]))

        if idx in plot_indices:
            plot_iterations(Fs, Fs_std, Es, Es_std, Ss, Ss_analytical,
                    ax_entropy_iterations, "$Z = %.1f$" % Z)

            color = next(ax_backflow._get_lines.prop_cycler)['color']
            plot_backflow_potential(model, cuda, ax_backflow, color,
                    "$\eta(r), Z = %.1f$" % Z, r"$\xi(r), Z = %.1f$" % Z)

            if idx == plot_indices[0]:
                old_weights = model.log_state_weights
                model.log_state_weights = torch.nn.Parameter(
                        -model.beta * (model.Es_original - model.Es_original[0]))
                x, _ = model.sample((density_batch,))
                plot_density(ax_density1, ax_density2, x, "no interaction")
                model.log_state_weights = old_weights
            _, x = model.sample((density_batch,))
            plot_density(ax_density1, ax_density2, x, "$Z = %.1f$" % Z)

    params_str = "beta_%.1f_" % beta + \
            "nup_%d_ndown_%d_" % (nup, ndown) + \
            "deltaE_%.1f_" % deltaE + \
           ("boltzmann_" if boltzmann else "") + \
            "Deta_%d_" % Deta + \
            "Dmu_%s_" % (Dmu if not nomu else None) + \
            "T0_%.1f_T1_%.1f_" % (t0, t1) + \
            "batch_%d" % batch
    sumup_file = "sumup_%s.txt" % params_str

    np.savetxt(sumup_file, thermo_quantities, fmt="%18.15f",
            header="%16s %18s %18s %18s %18s %18s %18s %18s" % thermo_quantity_labels)

    ax_entropy_iterations.set_xscale("log")
    ax_entropy_iterations.set_xlabel("Iters", size=18)
    ax_entropy_iterations.set_ylabel("Entropy", size=18)
    ax_entropy_iterations.legend()
    fig_entropy_iterations.tight_layout()
    #fig_entropy_iterations.savefig("entropy_%s.pdf" % params_str)

    ax_backflow.set_xlabel("$r$")
    ax_backflow.set_ylabel("Backflow potential", size=18)
    ax_backflow.grid(True)
    ax_backflow.legend()
    fig_backflow.tight_layout()
    #fig_backflow.savefig("backflow_%s.pdf" % params_str)

    ax_density1.set_xlabel("$r$")
    ax_density1.set_ylabel(r"$2 \pi r \rho(r)$")
    ax_density1.legend()
    fig_density1.tight_layout()
    #fig_density1.savefig("density1_%s.pdf" % params_str)

    ax_density2.set_xlabel("$r$")
    ax_density2.set_ylabel(r"$\rho(r)$")
    ax_density2.legend()
    fig_density2.tight_layout()
    #fig_density2.savefig("density2_%s.pdf" % params_str)

    plt.show()
