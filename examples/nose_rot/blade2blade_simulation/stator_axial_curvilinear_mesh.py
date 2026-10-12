"""
Curvilinear (meanline-following) quasi-orthogonal mesh for the axial
stator passage -- replaces the fixed constant-x cuts in
stator_axial_mesh.py / stator_axial_slc_solve.py.

Why: with fixed vertical cuts, the flow turning from 0 deg to 70 deg means
the cuts become increasingly oblique to the flow. Mass flux through a cut
scales with cos(Phi - Phi_cut), which collapses toward zero as the flow
direction departs from the cut's normal -- exactly the degeneracy Gajoni's
bend avoids by switching from Cartesian to polar (radially-oriented) cuts.
For an arbitrarily-curved passage (no fixed center to use for polar
coordinates), the general fix is to orient each cut perpendicular to the
LOCAL mean-flow direction, found from a mean camberline through the
passage, parameterized by its own arc length.

Construction:
  1. Build wall1(x), wall2(x) as before (stator_axial_mesh.py).
  2. Mean camberline = 0.5*(wall1+wall2) on a dense x grid.
  3. Arc-length parameterize the meanline, resample to Q_STATIONS
     uniformly-spaced-in-s stations.
  4. At each station: local tangent angle Phi_mean(s) (heading of the
     meanline), local normal direction n_hat(s).
  5. Wall offsets t1(s), t2(s): project each wall's point (at the same x
     used to build that meanline station) onto n_hat(s) -- an
     approximation to a true line-curve intersection, accurate as long as
     the walls are locally smooth relative to the passage width (true
     here), much simpler than root-finding an exact intersection.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml

from stator_axial_mesh import (
    load_blade, split_surfaces, resample_by_x, translate, extend_straight,
    check_combo, BLADE_FILE, N_EXT, EXT_FRAC,
)

Q_STATIONS = 80
N_DENSE = 600  # dense x-grid used to build/arclength-parameterize the meanline


def three_point_derivatives(f, x):
    n = len(x)
    fp = np.zeros(n)
    fpp = np.zeros(n)
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        fp[i] = (f[i + 1] - f[i - 1]) / (h1 + h2)
        fpp[i] = 2.0 * (h1 * f[i + 1] - (h1 + h2) * f[i] + h2 * f[i - 1]) \
            / (h1 * h2 * (h1 + h2))
    fp[0] = (f[1] - f[0]) / (x[1] - x[0])
    fp[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    fpp[0] = fpp[1] if n > 2 else 0.0
    fpp[-1] = fpp[-2] if n > 2 else 0.0
    return fp, fpp


def build_walls():
    case_file = sorted(glob.glob("meanline_stage_axial_*.yaml"))[-1]
    with open(case_file) as f:
        cfg = yaml.safe_load(f)
    geom = cfg["geometry"]["stator"]
    spacing = geom["spacing"]
    height = geom["height"]

    x, y = load_blade(BLADE_FILE)
    surfaces = split_surfaces(x, y)
    x_common = np.linspace(x.min() + 1e-5, x.max() - 1e-5, 30)
    candidates = [
        check_combo(surfaces, spacing, l1, l2, sign, x_common)
        for l1 in ("A", "B") for l2 in ("A", "B") if l2 != l1
        for sign in (+1.0, -1.0)
    ]
    valid = [c for c in candidates if c["frac_inside"] < 1e-6]
    best = min(valid, key=lambda c: abs(c["mean_width"] - spacing))

    x1, y1 = surfaces[best["label1"]]
    x2, y2 = surfaces[best["label2"]]

    angle_in = np.deg2rad(geom["metal_angle_in"])
    angle_out = np.deg2rad(geom["metal_angle_out"])
    dir_in = np.array([np.cos(angle_in), np.sin(angle_in)])
    dir_out = np.array([np.cos(angle_out), np.sin(angle_out)])

    ext_len = EXT_FRAC * spacing
    x1e, y1e = extend_straight(x1, y1, ext_len, N_EXT, dir_in, dir_out)
    x2e, y2e = extend_straight(x2, y2, ext_len, N_EXT, dir_in, dir_out)
    x2e, y2e = translate(x2e, y2e, best["sign"] * spacing)

    # Use the OVERLAP of both extended walls' x-ranges, not just wall1's:
    # the two surfaces were trimmed by different amounts near their tips
    # (build_walls trims each independently wherever it goes near-vertical,
    # and that trim depth differs between surfaces), so their extensions
    # -- built from different starting points but the same length -- don't
    # end at exactly the same x. Building x_dense from wall1's range alone
    # let it run past wall2's actual data, silently extrapolating (flat
    # clamp) wall2 over the last ~1% of the domain -- which is exactly the
    # pinch that showed up at the outlet end.
    x_lo = max(x1e.min(), x2e.min()) + 1e-6
    x_hi = min(x1e.max(), x2e.max()) - 1e-6
    x_dense = np.linspace(x_lo, x_hi, N_DENSE)
    _, yW1_dense = resample_by_x(x1e, y1e, x_dense)
    _, yW2_dense = resample_by_x(x2e, y2e, x_dense)
    if np.mean(yW2_dense - yW1_dense) < 0.0:
        yW1_dense, yW2_dense = yW2_dense, yW1_dense

    # Smooth the walls themselves here, not just the derived camberline:
    # these are what's actually drawn and what the cut endpoints are
    # pinned to, so a kink here shows up directly regardless of how
    # smooth the camberline/cut-angle signal is. Same short window as
    # the camberline smoothing (much shorter than the blade's real
    # curvature length scale), applied identically to both walls so they
    # stay a consistent pitch apart.
    from scipy.signal import savgol_filter
    window = max(11, 2 * (N_DENSE // 120) + 1)
    yW1_dense = savgol_filter(yW1_dense, window, 3)
    yW2_dense = savgol_filter(yW2_dense, window, 3)

    return x_dense, yW1_dense, yW2_dense, spacing, height, cfg


def build_curvilinear_mesh(Q_STATIONS=Q_STATIONS):
    x_dense, yW1, yW2, spacing, height, cfg = build_walls()

    # Mean camberline on the dense grid, then arc-length parameterize it.
    x_mean = x_dense
    y_mean = 0.5 * (yW1 + yW2)

    # Smooth the meanline signal itself before differentiating it for cut
    # directions. The quasi-orthogonal cut angle comes from a SECOND
    # derivative of this curve, which is far more sensitive to small
    # imperfections than the curve's visible shape -- geometric fixes to
    # the walls (matching the true tip tangent, blending the extension)
    # got the visible kink out of the walls themselves, but a residual,
    # much smaller wiggle still showed up as a visible kink in the cut
    # directions. A Savitzky-Golay filter with a window much shorter than
    # the blade's real curvature length scale (~0.03 m radius) kills that
    # residual without touching the genuine chord-scale curvature.
    from scipy.signal import savgol_filter
    window = max(11, 2 * (N_DENSE // 120) + 1)  # odd window, ~0.3% of N_DENSE
    y_mean = savgol_filter(y_mean, window, 3)

    ds_dense = np.hypot(np.diff(x_mean), np.diff(y_mean))
    s_dense = np.concatenate([[0.0], np.cumsum(ds_dense)])

    s_stations = np.linspace(0.0, s_dense[-1], Q_STATIONS)
    xm = np.interp(s_stations, s_dense, x_mean)
    ym = np.interp(s_stations, s_dense, y_mean)
    x_of_s = np.interp(s_stations, s_dense, x_dense)  # to re-evaluate the walls at matching x

    xm_p, _ = three_point_derivatives(xm, s_stations)
    ym_p, _ = three_point_derivatives(ym, s_stations)
    norm = np.hypot(xm_p, ym_p)
    tangent = np.stack([xm_p / norm, ym_p / norm], axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # rotate +90 deg
    Phi_mean = np.arctan2(tangent[:, 1], tangent[:, 0])

    _, yW1_at_s = resample_by_x(x_dense, yW1, x_of_s)
    _, yW2_at_s = resample_by_x(x_dense, yW2, x_of_s)

    v1 = np.stack([x_of_s - xm, yW1_at_s - ym], axis=1)
    v2 = np.stack([x_of_s - xm, yW2_at_s - ym], axis=1)
    t1 = np.einsum("ij,ij->i", v1, normal)
    t2 = np.einsum("ij,ij->i", v2, normal)
    if np.mean(t2 - t1) < 0.0:
        t1, t2 = t2, t1

    return dict(s=s_stations, xm=xm, ym=ym, tangent=tangent, normal=normal,
                Phi_mean=Phi_mean, t1=t1, t2=t2, spacing=spacing, height=height,
                cfg=cfg, x_dense=x_dense, yW1=yW1, yW2=yW2)


if __name__ == "__main__":
    mesh = build_curvilinear_mesh()
    s, xm, ym = mesh["s"], mesh["xm"], mesh["ym"]
    normal, t1, t2 = mesh["normal"], mesh["t1"], mesh["t2"]

    N_STREAMLINES = 15
    frac = np.linspace(0.0, 1.0, N_STREAMLINES)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mesh["x_dense"], mesh["yW1"], color="tab:blue", lw=2.0, label="wall 1")
    ax.plot(mesh["x_dense"], mesh["yW2"], color="tab:orange", lw=2.0, label="wall 2")
    ax.plot(xm, ym, "k--", lw=1.0, alpha=0.6, label="mean camberline")

    for k in range(len(s)):
        p1 = np.array([xm[k], ym[k]]) + t1[k] * normal[k]
        p2 = np.array([xm[k], ym[k]]) + t2[k] * normal[k]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.6)

    for f in frac:
        xs = xm + (t1 + f * (t2 - t1)) * normal[:, 0]
        ys = ym + (t1 + f * (t2 - t1)) * normal[:, 1]
        ax.plot(xs, ys, color="green", lw=0.8, alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("axial direction [m]")
    ax.set_ylabel("tangential direction [m]")
    ax.set_title("Axial stator passage: curvilinear (flow-following) quasi-orthogonals")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(pad=1)
    fig.savefig("stator_axial_curvilinear_mesh.png", dpi=150)
    print("saved stator_axial_curvilinear_mesh.png")
