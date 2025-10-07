"""Plot warm starting results."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "lines.linewidth": 2,
    }
)


files_cold = [
    "timing_results/timing_64_40_12_pcg=0.0001_warmstart=False",
    "timing_results/timing_64_40_12_pcg=1e-08_warmstart=False",
    "timing_results/timing_64_40_12_pcg=1e-12_warmstart=False",
]
files_warm = [
    "timing_results/timing_64_40_12_pcg=0.0001_warmstart=True",
    "timing_results/timing_64_40_12_pcg=1e-08_warmstart=True",
    "timing_results/timing_64_40_12_pcg=1e-12_warmstart=True",
]
pcg_tols = [1e-4, 1e-8, 1e-12]

data = {
    "cold": {pcg_tol: {} for pcg_tol in pcg_tols},
    "warm": {pcg_tol: {} for pcg_tol in pcg_tols},
}


for i, file in enumerate(files_cold):
    name = file.replace(".npy", "")
    pcg_tol = pcg_tols[i]
    data["cold"][pcg_tol]["fwd"] = np.load(file + "/diffmpc_cuda:0_fwd.npy")
    data["cold"][pcg_tol]["bwd"] = np.load(file + "/diffmpc_cuda:0_bwd.npy")
for i, file in enumerate(files_warm):
    name = file.replace(".npy", "")
    pcg_tol = pcg_tols[i]
    data["warm"][pcg_tol]["fwd"] = np.load(file + "/diffmpc_cuda:0_fwd.npy")
    data["warm"][pcg_tol]["bwd"] = np.load(file + "/diffmpc_cuda:0_bwd.npy")


fig, axs = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
for ax in axs:
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xscale("log")
    ax.set_xticks(pcg_tols)
for i, fwdbwd in enumerate(["fwd", "bwd"]):
    color = "b"
    speedups = np.array(
        [
            (data["cold"][pcg_tol][fwdbwd] - data["warm"][pcg_tol][fwdbwd])
            / data["cold"][pcg_tol][fwdbwd]
            for pcg_tol in pcg_tols
        ]
    )

    speedups *= 1e2
    axs[i].plot(pcg_tols, np.mean(speedups, axis=1), color=color, linewidth=2)
    p_low = np.percentile(speedups, 2.5, axis=-1)
    p_high = np.percentile(speedups, 97.5, axis=-1)
    axs[i].fill_between(pcg_tols, p_low, p_high, color=color, alpha=0.2)

for ax in axs:
    ax.grid(True)

axs[0].set_ylabel(r"Speedup (\%)", fontsize=16)
axs[0].set_xlabel(r"PCG Tolerance $\epsilon$", fontsize=16)
axs[1].set_xlabel(r"PCG Tolerance $\epsilon$", fontsize=16)
axs[0].set_title(r"Forward Pass", fontsize=16)
axs[1].set_title(r"Backward Pass", fontsize=16)
plt.tight_layout()

for i, fwdbwd in enumerate(["fwd", "bwd"]):
    for pcg_tol in pcg_tols:
        color = "b"
        speedup = np.array(
            [
                (data["cold"][pcg_tol][fwdbwd] - data["warm"][pcg_tol][fwdbwd])
                / data["cold"][pcg_tol][fwdbwd]
            ]
        )
        speedup *= 1e2
        print(
            f"{fwdbwd} speedup at toleps = {pcg_tol}: {np.mean(speedup)} +/- sigma="
            f" {np.std(speedup)}"
        )

plt.show()
