"""Overview plots (2x3 profile grid + T-s diagram) for a batch of inverse-design cases."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt

LABEL_FONTSIZE = 16


def plot_design_overview(
    fluid,
    results: List[Dict[str, Any]],
    case_labels: Sequence[str],
    colormap: str = "plasma",
    lw: float = 2.0,
):
    """Plot the 2x3 profile grid (Height, Pressure, Quality / Velocity, Mach, Density)
    and the T-s diagram for a batch of nozzle expansion cases.

    Parameters
    ----------
    fluid : jaxprop Fluid
        Used for the T-s background diagram and the critical-point axis limits.
    results : list of dict
        One dict per case, with keys 'x_norm', 'p', 'Q', 'area_ratio', 'u', 'mach',
        'rho', 's', 'T' (each a 1D array/sequence, all the same length per case).
        'area_ratio' is the dimensionless A/A* profile (no assumed duct width).
        Optionally also 'iso_x_norm', 'iso_p', 'iso_Q', 'iso_area_ratio', 'iso_u',
        'iso_mach', 'iso_rho' — the closed-form isentropic (frictionless)
        cross-check, drawn as a dashed black line per case.
    case_labels : sequence of str
        Legend label for each case, same length/order as `results`.

    Returns
    -------
    fig, axes, fig_ts, ax_ts
    """
    plt.rcParams.update({
        'axes.labelsize': LABEL_FONTSIZE,
        'axes.titlesize': LABEL_FONTSIZE,
        'legend.fontsize': LABEL_FONTSIZE,
    })

    num_cases = len(results)
    color_map = cm.get_cmap(colormap, num_cases)

    # --- Profile grid: Height, Pressure, Quality / Velocity, Mach, Density ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    ((ax_h, ax_p, ax_q), (ax_u, ax_m, ax_rho)) = axes
    for ax in axes.flat:
        ax.set_box_aspect(0.7)

    fig_ts, ax_ts = fluid.plot_phase_diagram("s", "T")
    fig_ts.set_size_inches(5, 4)

    case_handles = []
    iso_handles = []
    p_min_all = float("inf")
    s_min_all, s_max_all = float("inf"), -float("inf")
    T_out_min_all = float("inf")

    for i, res in enumerate(results):
        case_color = color_map(i)
        x_norm, p, Q, area_ratio = res['x_norm'], res['p'], res['Q'], res['area_ratio']
        u, mach, rho, s, T = res['u'], res['mach'], res['rho'], res['s'], res['T']

        ax_h.plot(x_norm, area_ratio / 2, color=case_color, lw=lw)
        ax_h.plot(x_norm, -area_ratio / 2, color=case_color, lw=lw)
        line, = ax_p.plot(x_norm, p, color=case_color, lw=lw, label=case_labels[i])
        case_handles.append(line)
        ax_q.plot(x_norm, Q, color=case_color, lw=lw)
        ax_u.plot(x_norm, u, color=case_color, lw=lw)
        ax_m.plot(x_norm, mach, color=case_color, lw=lw)
        ax_rho.plot(x_norm, rho, color=case_color, lw=lw)

        ax_ts.plot(s, T, color=case_color, lw=lw, zorder=1)
        ax_ts.plot(s[0], T[0], 'o', color=case_color, ms=7, mec='black', zorder=2)
        ax_ts.plot(s[-1], T[-1], 'o', color=case_color, ms=7, mec='black', zorder=2)

        if 'iso_x_norm' in res:
            iso_x = res['iso_x_norm']
            iso_line, = ax_p.plot(iso_x, res['iso_p'], 'k:', lw=1.5,
                                  label='Isentropic' if not iso_handles else None)
            if not iso_handles:
                iso_handles.append(iso_line)
            ax_h.plot(iso_x, res['iso_area_ratio'] / 2, 'k:', lw=1.5)
            ax_h.plot(iso_x, -res['iso_area_ratio'] / 2, 'k:', lw=1.5)
            ax_q.plot(iso_x, res['iso_Q'], 'k:', lw=1.5)
            ax_u.plot(iso_x, res['iso_u'], 'k:', lw=1.5)
            ax_m.plot(iso_x, res['iso_mach'], 'k:', lw=1.5)
            ax_rho.plot(iso_x, res['iso_rho'], 'k:', lw=1.5)

        p_min_all = min(p_min_all, min(p))
        s_min_all = min(s_min_all, min(s))
        s_max_all = max(s_max_all, max(s))
        T_out_min_all = min(T_out_min_all, T[-1])

    # --- Formatting ---
    ax_h.set_ylabel('$A/A^*$ (-)')
    ax_p.set_ylabel('$P$ (Pa)')
    ax_p.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax_q.set_ylabel('$Q$ (-)')
    ax_u.set_ylabel('$u$ (m/s)'); ax_u.set_xlabel('$x/L$ (-)')
    ax_m.set_ylabel('$M$ (-)'); ax_m.set_xlabel('$x/L$ (-)')
    ax_rho.set_ylabel(r'$\rho$ (kg/m³)'); ax_rho.set_xlabel('$x/L$ (-)')

    for ax in axes.flat:
        ax.set_xlim(0.0, 1.0)
    ax_p.set_ylim(bottom=0.5 * p_min_all)
    ax_q.set_ylim(0.0, 1.0)
    ax_u.set_ylim(bottom=0.0)
    ax_m.set_ylim(bottom=0.0)
    ax_rho.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    all_handles = case_handles + iso_handles
    fig.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=len(all_handles), fontsize=LABEL_FONTSIZE, frameon=True)

    s_margin = 0.1 * (s_max_all - s_min_all)
    ax_ts.set_xlim(s_min_all - s_margin, fluid.critical_point.s)
    ax_ts.set_ylim(0.8 * T_out_min_all, fluid.critical_point.T)
    fig_ts.tight_layout()

    return fig, axes, fig_ts, ax_ts
