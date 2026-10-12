"""
Reproduce Figure 4 of Gajoni & Guardone (JFM 2023): non-ideal planar
radial equilibrium for CO2, integrating Eq. (2.9) with RK45, from r~=1
(M=M_i) to r~=3.

Two panels, both with M_i = 0.5:
    (a) Pt/Pc = 0.5 (ideal-like), Tt/Tc = 0.95, 1, 1.1, 1.3
    (b) Pt/Pc = 2   (non-ideal),  Tt/Tc = 1.05, 1.11, 1.15, 1.25

Each panel also shows the ideal-gas analytic solution (Eq. 2.8, dotted)
for comparison, using gamma_ideal = lim_{P->0} cp(Tc,P)/cv(Tc,P) (Eq. 3.1).
"""

import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp

from common import (
    get_fluid, find_rho_for_mach, integrate_radial_equilibrium,
    ideal_gas_gamma, ideal_gas_mach_profile,
)

jxp.set_plot_options(grid=False, minor_ticks=False)

FLUID_NAME = "CO2"
M_I = 0.5
R_TILDE_END = 3.0


def run_panel(ax, fluid, Pt_over_Pc, Tt_over_Tc_values, gamma_ideal, Tc, Pc,
              collapsed=False):
    """
    collapsed=True: curves are (nearly) on top of each other (panel a), so a
    single note is used instead of per-curve labels, which would overlap.

    Otherwise, each curve gets its own label placed directly on it, at a
    different r-position per curve (staggered), so labels fall in the gaps
    between curves rather than stacking on top of one another or of the
    lines themselves.
    """
    Pt = Pt_over_Pc * Pc
    n = len(Tt_over_Tc_values)
    r_labels = np.linspace(1.75, R_TILDE_END - 0.15, n)
    curves = []

    for Tt_over_Tc in Tt_over_Tc_values:
        Tt = Tt_over_Tc * Tc
        stag = fluid.get_state(jxp.PT_INPUTS, Pt, Tt)
        s0, h0, rho_stag = stag.s, stag.h, stag.rho

        rho_i = find_rho_for_mach(fluid, M_I, s0, h0, rho_stag)
        r_tilde, rho, M = integrate_radial_equilibrium(fluid, s0, h0, rho_i, R_TILDE_END)
        ax.plot(r_tilde, M, "-", lw=1.0, color="black")

        M_ideal = ideal_gas_mach_profile(M_I, gamma_ideal, r_tilde)
        ax.plot(r_tilde, M_ideal, ":", lw=1.0, color="black")

        curves.append((Tt_over_Tc, r_tilde, M))

    if collapsed:
        r_tilde_ref = np.linspace(1.0, R_TILDE_END, 50)
        M_ref = ideal_gas_mach_profile(M_I, gamma_ideal, r_tilde_ref)
        M_at_2 = np.interp(2.0, r_tilde_ref, M_ref)
        ax.annotate(
            r"$T_t/T_c=$" + ", ".join(str(v) for v in Tt_over_Tc_values),
            xy=(2.0, M_at_2), xytext=(2.15, M_at_2 + 0.10),
            fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="black"),
        )
    else:
        # Order curves by their M value (nested, non-crossing) so each
        # label can be nudged to sit in the gap above its own curve
        # without touching a neighboring one.
        curves_sorted = sorted(curves, key=lambda c: np.interp(2.0, c[1], c[2]))
        for i, (Tt_over_Tc, r_tilde, M) in enumerate(curves_sorted):
            r_lab = r_labels[i]
            M_lab = np.interp(r_lab, r_tilde, M)
            # The lowest curve is hemmed in from above by its neighbor, so
            # its label points down into the open space below instead.
            dy = -20 if i == 0 else 22
            va = "top" if i == 0 else "bottom"
            ax.annotate(rf"${Tt_over_Tc}$", xy=(r_lab, M_lab),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=10, ha="center", va=va,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))

    ax.set_xlabel(r"$r/r_i$", fontsize=14)
    ax.set_ylabel(r"$M$", fontsize=14)
    ax.set_xlim(1, R_TILDE_END)
    ax.tick_params(axis="both", labelsize=13)


def main():
    fluid = get_fluid(FLUID_NAME)
    Tc, Pc = fluid.critical_point.T, fluid.critical_point.p
    gamma_ideal = ideal_gas_gamma(fluid, Tc)
    print(f"CO2: Tc={Tc:.2f} K, Pc={Pc:.1f} Pa, gamma_ideal(Tc)={gamma_ideal:.4f}")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))

    run_panel(ax_a, fluid, 0.5, [0.95, 1.0, 1.1, 1.3], gamma_ideal, Tc, Pc,
              collapsed=True)
    ax_a.set_title(r"(a) $P_t/P_c=0.5$", fontsize=14)

    run_panel(ax_b, fluid, 2.0, [1.05, 1.11, 1.15, 1.25], gamma_ideal, Tc, Pc)
    ax_b.set_title(r"(b) $P_t/P_c=2$", fontsize=14)

    fig.suptitle("Reproduction of Gajoni & Guardone (2023), Figure 4 -- CO2, "
                  r"$M_i=0.5$" "\n(solid: real fluid, dotted: ideal-gas model)")
    plt.tight_layout()
    fig.savefig("figure4_co2_nonideal.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
