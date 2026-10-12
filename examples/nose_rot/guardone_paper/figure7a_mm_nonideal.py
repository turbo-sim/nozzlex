"""
Reproduce Figure 7a of Gajoni & Guardone (JFM 2023): MM (hexamethyldi-
siloxane) in planar radial equilibrium, NON-ideal stagnation regime.

    Pt/Pc = 2.0
    Tt/Tc = 1.05, 1.07, 1.1, 1.2   (solid curves)
    M_i   = 1.75                   (supersonic inlet)
    r~ integrated from 1 to 5

Dotted curves: ideal-gas comparison model (Eq. 2.8), using
gamma_ideal = lim_{P->0} cp(Tc,P)/cv(Tc,P) (Eq. 3.1).

Unlike figure6a (Pt/Pc=0.5), here the flow states reach the J>0 region
in supersonic conditions for low Tt/Tc (closer to the non-ideal region),
so a NON-MONOTONE Mach number profile is expected for the lowest
Tt/Tc values -- the Mach number dips then rises again with r~, unlike
the monotonically-decreasing curves of figure6a. At higher Tt/Tc, the
stagnation state moves away from the non-ideal region and the profile
approaches the ideal-gas curve again.
"""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from common import (
    get_fluid, gamma_of_rho, find_rho_for_mach, integrate_radial_equilibrium,
    ideal_gas_gamma, ideal_gas_mach_profile, label_curves_crossing,
)

jxp.set_plot_options(grid=False, minor_ticks=False)

FLUID_NAME = "MM"
M_I = 1.75
R_TILDE_END = 5.0
PT_OVER_PC = 2.0
TT_OVER_TC_VALUES = [1.05, 1.07, 1.1, 1.2]


def main():
    fluid = get_fluid(FLUID_NAME)
    Tc, Pc = fluid.critical_point.T, fluid.critical_point.p
    gamma_ideal = ideal_gas_gamma(fluid, Tc)
    print(f"MM: Tc={Tc:.2f} K, Pc={Pc:.1f} Pa, gamma_ideal(Tc)={gamma_ideal:.4f}")

    Pt = PT_OVER_PC * Pc

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

        Gamma_path = np.array([gamma_of_rho(fluid, r, s0) for r in rho])
        J_path = 1.0 - Gamma_path - 1.0 / M**2
        monotone = np.all(np.diff(M) <= 1e-9)
        print(f"Tt/Tc={Tt_over_Tc}: Gamma in [{Gamma_path.min():.3f}, "
              f"{Gamma_path.max():.3f}], J in [{J_path.min():.3f}, "
              f"{J_path.max():.3f}], M {'monotone' if monotone else 'NON-MONOTONE'}")

        curves.append((Tt_over_Tc, r_tilde, M))

    # These curves CROSS each other (that's the non-monotone effect), so
    # use the crossing-aware labeler instead of the staggered one.
    label_curves_crossing(ax, curves, r_range=(1.3, R_TILDE_END - 0.3))

    ax.set_xlabel(r"$r/r_i$", fontsize=16)
    ax.set_ylabel(r"$M$", fontsize=16)
    ax.set_xlim(1, R_TILDE_END)
    ax.set_ylim(0, 2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title("Reproduction of Gajoni & Guardone (2023), Figure 7a -- MM\n"
                  r"$P_t/P_c=2$, $M_i=1.75$ "
                  "(solid: real fluid, dotted: ideal-gas model)")
    plt.tight_layout()
    fig.savefig("figure7a_mm_nonideal.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
