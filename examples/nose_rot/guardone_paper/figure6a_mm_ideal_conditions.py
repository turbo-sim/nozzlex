"""
Reproduce Figure 6a of Gajoni & Guardone (JFM 2023): MM (hexamethyldi-
siloxane) in planar radial equilibrium, ideal-like stagnation regime.

    Pt/Pc = 0.5
    Tt/Tc = 0.92, 0.95, 1.0, 1.1   (solid curves)
    M_i   = 1.75                   (supersonic inlet)
    r~ integrated from 1 to 5

Dotted curves: ideal-gas comparison model (Eq. 2.8), using
gamma_ideal = lim_{P->0} cp(Tc,P)/cv(Tc,P) (Eq. 3.1).

Even though Gamma < 1 throughout this stagnation regime (MM is a High
Molecular Complexity fluid), the flow states stay in the J<0 region (see
Eq. 2.5: J = 1-Gamma-1/M^2), so no non-monotone (non-ideal gasdynamic)
Mach behaviour is expected here -- only a quantitative (non-ideal
thermodynamic) deviation from the ideal-gas curves. This script also
prints Gamma and J along each path as a check.
"""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from common import (
    get_fluid, gamma_of_rho, find_rho_for_mach, integrate_radial_equilibrium,
    ideal_gas_gamma, ideal_gas_mach_profile, label_curves_staggered,
)

jxp.set_plot_options(grid=False, minor_ticks=False)

FLUID_NAME = "MM"
M_I = 1.75
R_TILDE_END = 5.0
PT_OVER_PC = 0.5
TT_OVER_TC_VALUES = [0.92, 0.95, 1.0, 1.1]


def main():
    fluid = get_fluid(FLUID_NAME)
    Tc, Pc = fluid.critical_point.T, fluid.critical_point.p
    gamma_ideal = ideal_gas_gamma(fluid, Tc)
    print(f"MM: Tc={Tc:.2f} K, Pc={Pc:.1f} Pa, gamma_ideal(Tc)={gamma_ideal:.4f}")

    Pt = PT_OVER_PC * Pc
    n = len(TT_OVER_TC_VALUES)
    r_labels = np.linspace(1.5, R_TILDE_END - 0.3, n)

    fig, ax = plt.subplots(figsize=(7, 6))
    curves = []

    for Tt_over_Tc in TT_OVER_TC_VALUES:
        Tt = Tt_over_Tc * Tc
        stag = fluid.get_state(jxp.PT_INPUTS, Pt, Tt)
        s0, h0, rho_stag = stag.s, stag.h, stag.rho

        rho_i = find_rho_for_mach(fluid, M_I, s0, h0, rho_stag)
        r_tilde, rho, M = integrate_radial_equilibrium(fluid, s0, h0, rho_i, R_TILDE_END)
        ax.plot(r_tilde, M, "-", lw=1.0, color="black")

        M_ideal = ideal_gas_mach_profile(M_I, gamma_ideal, r_tilde)
        ax.plot(r_tilde, M_ideal, ":", lw=1.0, color="black")

        # Check: Gamma < 1 and J < 0 throughout (no crossing), as expected
        # for this "ideal-like" stagnation regime (Pt/Pc=0.5).
        Gamma_path = np.array([gamma_of_rho(fluid, r, s0) for r in rho])
        J_path = 1.0 - Gamma_path - 1.0 / M**2
        print(f"Tt/Tc={Tt_over_Tc}: Gamma in [{Gamma_path.min():.3f}, "
              f"{Gamma_path.max():.3f}], J in [{J_path.min():.3f}, "
              f"{J_path.max():.3f}] (all < 0: {np.all(J_path < 0)})")

        curves.append((Tt_over_Tc, r_tilde, M))

    label_curves_staggered(ax, curves, r_labels)

    ax.set_xlabel(r"$r/r_i$", fontsize=16)
    ax.set_ylabel(r"$M$", fontsize=16)
    ax.set_xlim(1, R_TILDE_END)
    ax.set_ylim(0, 2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title("Reproduction of Gajoni & Guardone (2023), Figure 6a -- MM\n"
                  r"$P_t/P_c=0.5$, $M_i=1.75$ "
                  "(solid: real fluid, dotted: ideal-gas model)")
    plt.tight_layout()
    fig.savefig("figure6a_mm_ideal_conditions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
