import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Energy data of the present work.
data = {}
filename = "sumup_beta_6.0_nup_10_ndown_0_deltaE_2.0_boltzmann_Deta_50_Dmu_50_T0_0.0_T1_1.0_batch_8000.txt"
Z, F, F_error, E, E_error, S, S_analytical, S_flow = np.loadtxt(filename, unpack=True)
data = {"Z": Z, "F": F, "F_error": F_error, "E": E, "E_error": E_error,
                "S": S, "S_analytical": S_analytical, "S_flow": S_flow}
print("Z:", Z, "\nE:", E, "\nE_error:", E_error)


# Plot the energy data.
fig, ax = plt.subplots()
indices = [0, 5, 6, 7, 8, 9, 10, 11, 12]
Z, E, E_error = Z[indices], E[indices], E_error[indices]
ax.errorbar(Z, E, yerr=E_error/np.sqrt(8000), capsize=7, marker="o")
ax.set_xlabel("$\kappa$")
ax.set_ylabel("$E$")
ax.set_xticks(tuple(range(1, 9)))


def plot_density2D(ax, loaddir, batch, rmax=5.0, bins=500):
    filename = loaddir + "coordinates_flow_batch_%d.npy" % batch
    x = np.load(filename)
    n = x.shape[-2]

    # Plot density in the 2D x-y plane.
    xs, ys = x[..., 0].flatten(), x[..., 1].flatten()
    H, xedges, yedges = np.histogram2d(xs, ys, bins=2*bins, range=((-rmax, rmax), (-rmax, rmax)),
                density=True)
    ax.imshow(n * H, interpolation="nearest", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                cmap="inferno", vmin=0, vmax=0.7)
    ax.set_xticks([])
    ax.set_yticks([])

left1, bottom1, width1, height1 = 0.12, 0.45, 0.3, 0.3
ax1 = fig.add_axes((left1, bottom1, width1, height1))
loaddir1 = "/data1/xieh/FlowVMC/master/BetaFermionHO2D/" + \
        "beta_6.0_nup_10_ndown_0_deltaE_2.0_boltzmann_cuda_0_Deta_50_Dmu_50_T0_0.0_T1_1.0_batch_8000_Z_0.5/"
plot_density2D(ax1, loaddir1, 2000000)
ax.arrow(1.0, 78, -0.4, -28, head_width=0.1, head_length=5, facecolor="black")

left2, bottom2, width2, height2 = 0.7, 0.45, 0.3, 0.3
ax2 = fig.add_axes((left2, bottom2, width2, height2))
loaddir2 = "/data1/xieh/FlowVMC/master/BetaFermionHO2D/" + \
        "beta_6.0_nup_10_ndown_0_deltaE_2.0_boltzmann_cuda_1_Deta_50_Dmu_50_T0_0.0_T1_1.0_batch_8000_Z_8.0/"
plot_density2D(ax2, loaddir2, 2000000)
ax.arrow(7.5, 130, 0.38, 18, head_width=0.1, head_length=5, facecolor="black")

plt.tight_layout()
fig.savefig("energy_data2.pdf", dpi=800)
plt.show()
