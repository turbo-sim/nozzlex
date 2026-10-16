import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import nozzlex as nzx

# ======================================================
# CONFIGURATION
# ======================================================
CASES = [1]          # one or more case numbers, e.g. [1, 3, 6, 8]
SAVE_FIGS = True
AREA_AVG_X_TARGET = 0.309528       # which x-station to show in the plot markers and error table

# Case number -> mixture ratio used in the HEM pickle filename
CASE_MIXTURE_RATIO = {
    1: 22.45,
    2: 29.26,
    3: 37.96,
    4: 48.85,
    5: 60.76,
    6: 67.88,
    7: 76.46,
    8: 94.48,
}

AREA_AVERAGE_CFD = {
    1: {0.309528: 153.05597, 0.311713: 154.00821},
    2: {0.309528: 138.45216, 0.311713: 138.35198},
    3: {0.309528: 125.92418, 0.311713: 124.83196},
    4: {0.309528: 115.43095, 0.311713: 113.42556},
    5: {0.309528: 107.53303, 0.311713: 104.77947},
    6: {0.309528: 103.90974, 0.311713: 100.79233},
    7: {0.309528: 100.29081, 0.311713: 96.794877},
    8: {0.309528: 94.499621, 0.311713: 90.361867},
}

CFD_ROOT = os.path.join("..", "..", "..", "output_data_cfd")
HEM_ROOT = os.path.join("..", "..", "..", "hem_solutions")


# ======================================================
# HELPERS
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cfd(case):
    tag  = f"case_{case:03d}_a"
    base = os.path.join(CFD_ROOT, tag, "simulation_latest")

    p_df = nzx.graphics.parse_fluent_xy(os.path.join(base, f"{tag}_pressure.xy"))
    v_df = nzx.graphics.parse_fluent_xy(os.path.join(base, f"{tag}_velocity.xy"))

    x_p = p_df["axis"]["Position"].values
    p   = p_df["axis"]["Static Pressure"].values / 1e3      # Pa -> kPa

    x_v = v_df["axis"]["Position"].values
    # Pick the first non-Position column (handles 'Velocity Magnitude' or 'X Velocity')
    vel_col = [c for c in v_df["axis"].columns if c != "Position"][0]
    v = v_df["axis"][vel_col].values

    return x_p, p, x_v, v


def load_hem(case):
    R    = CASE_MIXTURE_RATIO[case]
    path = os.path.join(HEM_ROOT, f"solution_{R:g}.pkl")
    sol  = load_pickle(path)
    x = np.asarray(sol["x"])
    p = np.asarray(sol["p"]) / 1e3     # Pa -> kPa
    v = np.asarray(sol["v"])
    return x, p, v


# ======================================================
# PLOT
# ======================================================
nzx.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
ax2 = ax1.twinx()

ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")
ax2.set_ylabel("Velocity [m/s]")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, case in enumerate(CASES):
    color = colors[i % len(colors)]
    R     = CASE_MIXTURE_RATIO[case]
    label = f"R={R:g}"

    try:
        x_p, p_cfd, x_v, v_cfd = load_cfd(case)
        ax1.plot(x_p, p_cfd, color=color, linestyle="-",  linewidth=1.05, label=f"CFD ({label})")
        ax2.plot(x_v, v_cfd, color=color, linestyle="-",  linewidth=1.05)
    except Exception as e:
        print(f"CFD case {case}: {e}")

    try:
        x, p_hem, v_hem = load_hem(case)
        ax1.plot(x, p_hem, color=color, linestyle="--", linewidth=1.05, label=f"HEM ({label})")
        ax2.plot(x, v_hem, color=color, linestyle="--", linewidth=1.05)
    except Exception as e:
        print(f"HEM case {case}: {e}")

    if case in AREA_AVERAGE_CFD:
        aa_dict = AREA_AVERAGE_CFD[case]
        aa_x = np.array([AREA_AVG_X_TARGET])
        aa_v = np.array([aa_dict[AREA_AVG_X_TARGET]])
        ax2.plot(aa_x, aa_v, "^", color=color, markersize=6, zorder=5,
                 label=f"CFD area-avg vel ({label})")

        print(f"\n--- Area-average velocity errors (case {case}, R={R:g}) ---")
        print(f"  {'x [m]':<12} {'CFD_aa':>10} {'CFD_ax':>10} {'err_ax':>12} {'HEM':>10} {'err_HEM':>12}")

        try:
            v_cfd_ax_at_aa = np.interp(aa_x, x_v, v_cfd)
        except Exception:
            v_cfd_ax_at_aa = np.full_like(aa_x, np.nan)

        try:
            v_hem_at_aa = np.interp(aa_x, x, v_hem)
        except Exception:
            v_hem_at_aa = np.full_like(aa_x, np.nan)

        for xi, v_aa, v_ax, v_h in zip(aa_x, aa_v, v_cfd_ax_at_aa, v_hem_at_aa):
            err_ax  = (v_ax - v_aa) / v_aa * 100
            err_hem = (v_h  - v_aa) / v_aa * 100
            print(f"  {xi:<12.6f} {v_aa:>10.4f} {v_ax:>10.4f} {err_ax:>+11.2f}% {v_h:>10.4f} {err_hem:>+11.2f}%")

ax1.set_ylim([-100, 2100])
ax2.set_ylim([-5, 165])
ax1.legend(loc="center left", fontsize=9)
fig.tight_layout(pad=1)

if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), "comparison_cfd_hem")
    nzx.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
