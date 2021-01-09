import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Energy data of the paper: J. Chem. Phys. 153, 234104 (2020).
files = np.array(["PRE", "PBPIMC", "eta0.1", "eta0.2", "eta0", "extrapolated"])
labels = np.array(["PRE", "PBPIMC", "$\eta = 0.1$", "$\eta = 0.2$", "$\eta = 0$", "extrapolated"])
data_jcp = {}
for filename in files:
    usecols = (0, 9, 10) if filename != "extrapolated" else (0, 1, 2)
    beta, E, E_error = np.loadtxt("jcp_data/%s.txt" % filename, usecols=usecols, unpack=True)
    data_jcp[filename] = {"beta": beta, "E": E, "E_error": E_error}
    print("---- %s ----" % filename)
    print("beta:", beta, "\nE:", E, "\nE_error:", E_error)


# Energy data of the present work.
data = {}
beta_new = np.array([2, 2.5, 3, 4, 5, 6])
deltaEs = [2, 3, 4]
data[2] = {"E": np.array([19.123831532650104, 18.895257341518963, 18.699338958445537, 
                          18.449204759931455, 18.31013292766241, 18.24566486399798])}
data[3] = {"E": np.array([19.352662698631693, 19.001804338661042, 18.762737052698064, 
                          18.439585756324526, 18.30518341332807, 18.26166008103948])}
data[4] = {"E": np.array([19.451894148637724, 19.051535381949076, 18.768763247436326, 
                          18.439925946041527, 18.29076179102448, 18.260959379998507])}
beta_inf = 10.0
data["inf"] = {"E": 18.19107329214244}


# Plot the energy data.
fig, (ax, ax_inf) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]})
indices = [1, 4, 5]
markers = ["s", "D", "v"]
for filename, label, marker in zip(files[indices], labels[indices], markers):
    beta, E = data_jcp[filename]["beta"], data_jcp[filename]["E"]
    ax.plot(beta, E, linestyle="None", marker=marker, markerfacecolor="None", label=label)
for deltaE in deltaEs:
    ax.plot(beta_new, data[deltaE]["E"], linestyle="None", marker="o",
            label=r"$\Delta E_{\textrm{cut}} = %d$" % deltaE)
ax.set_xlim(right=beta_new[-1] + 0.5)

ax_inf.plot(beta_inf, data["inf"]["E"], linestyle="None", marker="o", label="ground state")
ax_inf.set_xlim(beta_inf - 1, beta_inf + 0.2)
ax_inf.set_xticks((beta_inf,))
ax_inf.set_xticklabels(("$\infty$",))
ax_inf.tick_params(axis="y", left=False, right=False)

ax.spines['right'].set_visible(False)
ax_inf.spines['left'].set_visible(False)
fig.legend(loc=(0.65, 0.39))

ax.set_xlabel('.', color=(0, 0, 0, 0))
ax.set_ylabel('.', color=(0, 0, 0, 0))
fig.text(0.53, 0.04, r"$\beta$", va='center', ha='center', fontsize=mpl.rcParams['axes.labelsize'])
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
plt.show()
