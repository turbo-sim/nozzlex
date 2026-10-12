"""
Reproduce Figure 8a of Gajoni & Guardone (JFM 2023): D6 (dodecamethyl-
cyclohexasiloxane), a BZT (Bethe-Zel'dovich-Thompson) fluid, in
non-classical planar radial equilibrium.

    Pt/Pc = 1.1171
    Tt/Tc = 1.0094
    M_i   = 1.2   (subsonic -- non-classical BZT behaviour occurs
                   subsonically, unlike the HMC/MM cases which need
                   supersonic conditions; M_i read off the paper's
                   Fig. 8a y-axis, not stated explicitly in the text)
    r~ integrated from 1 to 5

For BZT fluids, Gamma < 0 is admissible somewhere in the envelope, which
allows J = 1-Gamma-1/M^2 > 0 even in SUBSONIC conditions (see Eq. 2.5
discussion) -- unlike HMC fluids like MM, which need M>1 for J>0. This
produces the non-classical non-monotone Mach profile that is the point
of this figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from common import (
    get_fluid, gamma_of_rho, find_rho_for_mach, integrate_radial_equilibrium,
)

jxp.set_plot_options(grid=False, minor_ticks=False)

FLUID_NAME = "D6"
M_I = 1.2
R_TILDE_END = 5.0
PT_OVER_PC = 1.1171
TT_OVER_TC = 1.0094


def main():
    fluid = get_fluid(FLUID_NAME)
    Tc, Pc = fluid.critical_point.T, fluid.critical_point.p

    Pt = PT_OVER_PC * Pc
    Tt = TT_OVER_TC * Tc
    stag = fluid.get_state(jxp.PT_INPUTS, Pt, Tt)
    s0, h0, rho_stag = stag.s, stag.h, stag.rho

    rho_i = find_rho_for_mach(fluid, M_I, s0, h0, rho_stag)
    r_tilde, rho, M = integrate_radial_equilibrium(fluid, s0, h0, rho_i, R_TILDE_END)

    Gamma_path = np.array([gamma_of_rho(fluid, r, s0) for r in rho])
    J_path = 1.0 - Gamma_path - 1.0 / M**2
    monotone = np.all(np.diff(M) <= 1e-9)
    print(f"D6: Tc={Tc:.2f} K, Pc={Pc:.1f} Pa")
    print(f"Gamma in [{Gamma_path.min():.3f}, {Gamma_path.max():.3f}] "
          f"(Gamma<0 anywhere: {np.any(Gamma_path < 0)})")
    print(f"J in [{J_path.min():.3f}, {J_path.max():.3f}], "
          f"M {'monotone' if monotone else 'NON-MONOTONE'}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(r_tilde, M, "-", lw=1.2, color="black")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")

    ax.set_xlabel(r"$r/r_i$", fontsize=16)
    ax.set_ylabel(r"$M$", fontsize=16)
    ax.set_xlim(1, R_TILDE_END)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title("Reproduction of Gajoni & Guardone (2023), Figure 8a -- D6 (BZT)\n"
                  rf"$P_t/P_c={PT_OVER_PC}$, $T_t/T_c={TT_OVER_TC}$, $M_i={M_I}$")
    plt.tight_layout()
    fig.savefig("figure8a_d6_bzt.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
