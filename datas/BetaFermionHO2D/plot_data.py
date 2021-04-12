import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Energy data of the paper: J. Chem. Phys. 153, 234104 (2020).
files = np.array(["PRE", "PBPIMC", "eta0.1", "eta0.2", "eta0", "extrapolated"])
labels = np.array(["PRE", "PBPIMC", "$\eta = 0.1$", "$\eta = 0.2$", "$\eta = 0$", "extrapolated"])
data_jcp = {}
print("==== data of the jcp paper ====")
for filename in files:
    usecols = (0, 9, 10) if filename != "extrapolated" else (0, 1, 2)
    beta, E, E_error = np.loadtxt("jcp_data/%s.txt" % filename, usecols=usecols, unpack=True)
    data_jcp[filename] = {"beta": beta, "E": E, "E_error": E_error}
    print("---- %s ----" % filename)
    print("beta:", beta, "\nE:", E, "\nE_error:", E_error)


# Energy data of the present work.
data = {}
deltaEs = [4]
print("==== data of the present work ====")
for deltaE in deltaEs:
    filename = "sumup_Z_0.5_nup_6_ndown_0_deltaE_%.1f_boltzmann_Deta_50_Dmu_50_T0_0.0_T1_1.0_batch_8000.txt" % deltaE
    beta, F, F_error, E, E_error, S, S_analytical = np.loadtxt(filename, unpack=True)
    data[deltaE] = {"beta": beta, "F": F, "F_error": F_error, "E": E, "E_error": E_error,
                    "S": S, "S_analytical": S_analytical}
    print("---- deltaE = %.1f ----" % deltaE)
    print("beta:", beta, "\nE:", E, "\nE_error:", E_error)
beta_inf = 10.0
data["inf"] = {"E": 18.181598714718717, "E_error": 0.3591216807094323}


# Plot the energy data.
fig, (ax, ax_inf) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]})
indices = [1, 5]
markers = ["s", "o"]
colors = ["green", "blue"]
for filename, label, marker, color in zip(files[indices], labels[indices], markers, colors):
    beta, E, E_error = data_jcp[filename]["beta"], data_jcp[filename]["E"], data_jcp[filename]["E_error"]
    # Only take the data points for beta>=2.
    beta, E, E_error = beta[3:], E[3:], E_error[3:]
    ax.errorbar(beta, E, yerr=E_error, linestyle="None", capsize=7, color=color,
                marker=marker, markerfacecolor="None", label=label)
deltaE = 4
beta_new, E, E_error = data[deltaE]["beta"], data[deltaE]["E"], data[deltaE]["E_error"]
ax.errorbar(beta_new, E, yerr=E_error/np.sqrt(8000),
            linestyle="None", capsize=7, marker="o", color="red", label="FermiFlow")
ax.set_xlim(right=beta_new[-1] + 0.5)

ax_inf.errorbar(beta_inf, data["inf"]["E"], yerr=data["inf"]["E_error"]/np.sqrt(8000),
            linestyle="None", capsize=7, marker="o", color="red")
ax_inf.set_xlim(beta_inf - 1, beta_inf + 0.2)
ax_inf.set_xticks((beta_inf,))
ax_inf.set_xticklabels(("$\infty$",))
ax_inf.tick_params(axis="y", left=False, right=False)

ax.spines['right'].set_visible(False)
ax_inf.spines['left'].set_visible(False)
fig.legend(loc=(0.65, 0.65))

ax.set_xlabel('.', color=(0, 0, 0, 0))
ax.set_ylabel('.', color=(0, 0, 0, 0))
fig.text(0.55, 0.04, r"$\beta$", va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])
fig.text(0.04, 0.56, "$E$", va='center', ha='center', rotation='vertical', fontsize=mpl.rcParams['axes.labelsize'])

dx, dy = .015, .04
y = np.linspace(-dy, dy, num=50)
x = -dx * np.sin(np.pi / dy * y)
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot(1 + x, y, **kwargs)
ax.plot(1 + x, 1 + y, **kwargs)
kwargs.update(transform=ax_inf.transAxes)
ax_inf.plot(4*x, y, **kwargs)
ax_inf.plot(4*x, 1 + y, **kwargs)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
#fig.savefig("energy_data.pdf")
plt.show()
