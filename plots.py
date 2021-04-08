import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import matplotlib.pyplot as plt

def plot_iterations_GS(Es, Es_std, savefig=False, savedir=None):
    print("Es:", Es)
    #print("Es_std:", Es_std)
    iters, = Es.shape
    print("Number of iterations:", iters)

    iters = np.arange(1, iters + 1)
    Es_numpy = Es.cpu().numpy()
    plt.plot(iters, Es_numpy)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iters", size=18)
    plt.ylabel("$E$", size=18)
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "Es.pdf")
    plt.show()

def plot_iterations(Fs, Fs_std, Es, Es_std, Ss, Ss_analytical, S_flow,
                    savefig=False, savedir=None):
    #print("Fs:", Fs)
    #print("Fs_std:", Fs_std)
    print("Es:", Es)
    #print("Es_std:", Es_std)
    print("entropy (analytical):", Ss_analytical)

    assert Fs.shape == Es.shape
    iters, = Es.shape
    print("Number of iterations:", iters)

    print("F:", Fs[-1].item(), "F_std:", Fs_std[-1].item(),
          "E:", Es[-1].item(), "E_std:", Es_std[-1].item(),
          "S:", Ss[-1].item(), "S_analytical:", Ss_analytical[-1].item())

    iters = np.arange(1, iters + 1)
    Fs_numpy, Es_numpy = Fs.cpu().numpy(), Es.cpu().numpy()
    plt.plot(iters, Fs_numpy, label="$F$")
    plt.plot(iters, Es_numpy, label="$E$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iters", size=18)
    plt.ylabel("Observable", size=18)
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "observable.pdf")
    plt.show()
    
    Ss_numpy, Ss_analytical_numpy = Ss.cpu().numpy(), Ss_analytical.cpu().numpy()
    plt.plot(iters, Ss_numpy, label="MC sampling")
    plt.plot(iters, Ss_analytical_numpy, label="analytical")
    if S_flow is not None: plt.plot(iters, S_flow * np.ones_like(iters), label="flow")
    plt.xscale("log")
    plt.xlabel("Iters", size=18)
    plt.ylabel("Entropy", size=18)
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "entropy.pdf")
    plt.show()

def plot_backflow_potential(model, device, r_max=20.0,
                            savefig=False, savedir=None):
    eta, mu = model.cnf.backflow_potential()
    r = np.linspace(0., r_max, num=int(r_max * 100))
    eta_r = eta( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
    plt.plot(r, eta_r, label="$\eta(r)$")
    if mu is not None:
        mu_r = mu( torch.from_numpy(r).to(device=device)[:, None] )[:, 0].detach().cpu().numpy()
        plt.plot(r, mu_r, label=r"$\xi(r)$")
    plt.xlabel("$r$")
    plt.ylabel("Backflow potential", size=18)
    #plt.title("$\\xi^{e-e}_i = \\sum_{j \\neq i} \\eta(|r_i - r_j|) (r_i - r_j)$" +
              #("\t\t$\\xi^{e-n}_i = \\mu(|r_i|) r_i$" if mu is not None else ""))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "backflow.pdf")
    plt.show()

def plot_energylevels(model, batch, device, loaddir, savedir, savefig=False):
    import os

    # Load or compute from scratch the energy level data Es_flow and Es_std_flow.
    filename = loaddir + "energylevels_batch_%d.pt" % batch
    if os.path.exists(filename):
        print("Load energy data file: %s" % filename)
        energies = torch.load(filename)
        Es_flow = energies["Es_flow"]
        Es_std_flow = energies["Es_std_flow"]
    else:
        print("Compute the energy data...")
        Es_flow, Es_std_flow = model.compute_energies((batch,), device)
        energies = {"Es_flow": Es_flow, 
                    "Es_std_flow": Es_std_flow, 
                    }
        torch.save(energies, filename)
        print("Energy data saved to file: %s" % filename)

    # Compute entropy from the energy level data Es_flow.
    from torch.distributions.categorical import Categorical
    dist = Categorical(logits=-model.beta * Es_flow)
    S_flow = dist.entropy().item()

    # Compute the energy level data Es_state_weights.
    log_state_weights = model.log_state_weights.detach()
    log_state_weights = log_state_weights - log_state_weights[0]
    Es_state_weights =  -log_state_weights / model.beta + Es_flow[0]

    # Some prints.
    print("Es_original:", model.Es_original)
    print("Es_flow:", Es_flow)
    #print("Es_std_flow:", Es_std_flow)
    print("Es_state_weights:", Es_state_weights)
    print("Es_state_weights - Es_flow:", Es_state_weights - Es_flow)

    # Plot the data.
    figname = savedir + "energylevels_batch_%d.pdf" % batch
    _plot_energylevels(model.Es_original, Es_flow, Es_state_weights, figname, savefig)
    return S_flow

def _plot_energylevels(Es_original, Es_flow, Es_state_weights, figname, savefig):
    xcenter_original, xcenter_flow, xcenter_state_weights = 0.0, 2.0, 4.0
    color_original, color_flow, color_state_weights = "red", "green", "blue"
    halfwidth = 0.5
    N = 200
    x_original = np.linspace(xcenter_original - halfwidth, xcenter_original + halfwidth, num=N)
    x_flow = np.linspace(xcenter_flow - halfwidth, xcenter_flow + halfwidth, num=N)
    x_state_weights = np.linspace(xcenter_state_weights - halfwidth, xcenter_state_weights + halfwidth, num=N)
    for E_original, E_flow, E_state_weights in zip(
            Es_original.cpu().numpy(), Es_flow.cpu().numpy(), Es_state_weights.cpu().numpy()):
        plt.plot(x_original, E_original * np.ones(N), lw=0.5, color=color_original)
        plt.plot(x_flow, E_flow * np.ones(N), lw=0.5, color=color_flow)
        plt.plot(x_state_weights, E_state_weights * np.ones(N), lw=0.5, color=color_state_weights)
    plt.xticks((xcenter_original, xcenter_flow, xcenter_state_weights), 
               ("base", "flow", "state weights"))
    plt.ylabel("$E$")
    #plt.ylim(41, 44)       # N = 10, Z = 0.5
    #plt.ylim(43, 46)       # N = 10, Z = 0.6
    #plt.ylim(45.5, 48)     # N = 10, Z = 0.7
    #plt.ylim(47.5, 50)     # N = 10, Z = 0.8
    #plt.ylim(50, 52.5)     # N = 10, Z = 0.9
    #plt.ylim(52, 54.5)     # N = 10, Z = 1.0
    #plt.ylim(71, 73)       # N = 10, Z = 2.0
    #plt.ylim(87, 90)       # N = 10, Z = 3.0
    #plt.ylim(103, 106)     # N = 10, Z = 4.0
    #plt.ylim(117.5, 120)   # N = 10, Z = 5.0
    #plt.ylim(131, 134)     # N = 10, Z = 6.0
    plt.ylim(143.5, 146.5)  # N = 10, Z = 7.0
    #plt.ylim(156, 159)     # N = 10, Z = 8.0
    plt.tight_layout()
    if savefig: plt.savefig(figname)
    plt.show()

def plot_density(model, batch, times=1, rmax=5.0, bins=300, savefig=False, savedir=None):
    for i in range(times):
        old_weights = model.log_state_weights
        model.log_state_weights = torch.nn.Parameter(
                -model.beta * (model.Es_original - model.Es_original[0]))
        x0_new, _ = model.sample((batch,))
        model.log_state_weights = old_weights
        _, x1_new = model.sample((batch,))

        x0 = x0_new if i==0 else torch.cat((x0, x0_new), dim=0)
        x1 = x1_new if i==0 else torch.cat((x1, x1_new), dim=0)

    print("x0.shape:", x0.shape)
    print("x1.shape:", x1.shape)
    x0, x1 = x0.cpu().numpy(), x1.cpu().numpy()

    rs0, rs1 = np.linalg.norm(x0, axis=-1), np.linalg.norm(x1, axis=-1)
    n = rs0.shape[-1]
    hist0, bin_edges0 = np.histogram(rs0, bins=bins, range=(0.0, rmax), density=True)
    hist1, bin_edges1 = np.histogram(rs1, bins=bins, range=(0.0, rmax), density=True)
    bin_centers0 = (bin_edges0[:-1] + bin_edges0[1:])/2
    bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:])/2

    plt.plot(bin_centers0, n * hist0, label="no interaction")
    plt.plot(bin_centers1, n * hist1, label="flow")
    plt.xlabel("$r$")
    plt.ylabel(r"$2 \pi r \rho(r)$")
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "density1.pdf")
    plt.show()

    plt.plot(bin_centers0, n * hist0 / (2 * np.pi * bin_centers0), label="no interaction")
    plt.plot(bin_centers1, n * hist1 / (2 * np.pi * bin_centers1), label="flow")
    plt.xlabel("$r$")
    plt.ylabel(r"$\rho(r)$")
    plt.legend()
    plt.tight_layout()
    if savefig: plt.savefig(savedir + "density2.pdf")
    plt.show()

    xs, ys = x1[..., 0].flatten(), x1[..., 1].flatten()
    H, xedges, yedges = np.histogram2d(xs, ys, bins=2*bins, range=((-rmax, rmax), (-rmax, rmax)),
                density=True)
    plt.imshow(n * H, interpolation="nearest", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                cmap="inferno", vmin=0, vmax=0.7)
    if savefig: plt.savefig(savedir + "density2D.pdf")
    plt.show()
