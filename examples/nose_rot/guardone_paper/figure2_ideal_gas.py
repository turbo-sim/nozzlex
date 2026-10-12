"""
Reproduce Figure 2 of Gajoni & Guardone (JFM 2023).

Ideal polytropic gas with constant specific heats, planar radial
equilibrium. Eq. (2.8):

    M(r~) = M_i / sqrt( (1 + (gamma-1)/2 * M_i^2) * r~^2 - (gamma-1)/2 * M_i^2 )

with r~ = r/r_i in [1, r_e/r_i], r_e/r_i = 5 (paper's Fig. 2 caption).

Fluids / gamma values (paper Sec. 3.1):
    N2   : gamma = 1.4
    CO2  : gamma = 1.29
    MM   : gamma = 1.026   (siloxane, hexamethyldisiloxane)

M_i = 0.5, 1, 1.5, 2 (paper's Fig. 2).
"""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from common import ideal_gas_mach_profile

jxp.set_plot_options(grid=False, minor_ticks=False)

GAMMAS = {
    "N2": 1.4,
    "CO2": 1.29,
    "MM": 1.026,
}

M_I_VALUES = [0.5, 1.0, 1.5, 2.0]
R_TILDE_E = 5.0
N_POINTS = 300


def main():
    r_tilde = np.linspace(1.0, R_TILDE_E, N_POINTS)

    fig, ax = plt.subplots(figsize=(6, 5))
    linestyles = {
        "N2": "-",
        "CO2": "--",
        "MM": ":",
    }

    for fluid, gamma in GAMMAS.items():
        for M_i in M_I_VALUES:
            M = ideal_gas_mach_profile(M_i, gamma, r_tilde)
            ax.plot(r_tilde, M, linestyle=linestyles[fluid], color="black", lw=1.0)

    # One legend entry per fluid (paper style)
    for fluid, ls in linestyles.items():
        ax.plot([], [], linestyle=ls, color="black", lw=1.0, label=fluid)

    ax.set_xlabel(r"$r/r_i$", fontsize=16)
    ax.set_ylabel(r"$M$", fontsize=16)
    ax.set_xlim(1, R_TILDE_E)
    ax.set_ylim(0, 2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="upper right", fontsize=16)
    ax.set_title("Reproduction of Gajoni & Guardone (2023), Figure 2\n"
                  r"ideal polytropic gas, $M_i$ = 0.5, 1, 1.5, 2")
    plt.tight_layout()
    fig.savefig("figure2_ideal_gas.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
