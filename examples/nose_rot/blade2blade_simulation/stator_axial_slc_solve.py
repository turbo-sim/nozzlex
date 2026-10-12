"""
Blade-to-blade SLC solve for the axial stator passage built in
stator_axial_mesh.py, reusing the numerical machinery proven out this
session on the Gajoni & Guardone bend problem (nozzlex/examples/nose_rot/
guardone_paper/figure10_slc.py): same three-nested-loop structure
(m-loop / U-loop / r-loop), same zero-first-pass + 1-2-1 spatial
smoothing + adaptive relaxation + U-resampling-after-reposition fixes.

Two structural differences from the Gajoni case:
  - Geometry is Cartesian throughout (x = axial position is the single
    marching coordinate for the whole domain -- inlet extension, blade
    passage, outlet extension). No bend/polar branch is needed: unlike
    Gajoni's duct, the turning here comes entirely from the shape of the
    two walls in the (x,y) plane, not from a change of coordinate system.
  - The two walls are NOT a fixed distance apart (Gajoni's r_i, r_e were
    constants): here wall1(x), wall2(x) both vary with x, since the
    blade passage genuinely converges/diverges. Streamline positions are
    pinned to the LOCAL wall position at every station, not just at the
    domain ends.

Mass flow target and inlet state come from the mean-line design
(meanline_stage_axial_*.yaml): the domain starts one full pitch upstream
of the leading edge, where the flow is still uniform and axial, so the
inlet condition is simply the mean-line's station-1 (stator inlet)
velocity, imposed uniformly -- no separate boundary-layer/equilibration
treatment needed there.
"""

import glob
import numpy as np
from scipy.interpolate import interp1d
import yaml

from stator_axial_mesh import (
    load_blade, split_surfaces, resample_by_x, translate, extend_straight,
    estimate_tangent, check_combo, BLADE_FILE, N_EXT, EXT_FRAC,
)

GAMMA = 1.4
R_GAS = 287.05  # air, J/(kg K)

N = 15            # streamlines across the passage
Q_STATIONS = 80   # quasi-orthogonal stations along the passage

TOL_M = TOL_U = TOL_R = 1e-3
MAX_M = MAX_U = MAX_R = 500
STALL_WINDOW = 15


def rho_of_speed(u, T0, P0):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    return (P0 * (T / T0) ** (GAMMA / (GAMMA - 1.0))) / (R_GAS * T)


def mach_of_speed(u, T0):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    c = np.sqrt(GAMMA * R_GAS * T)
    return u / c


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


def build_passage(Q_STATIONS):
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

    x_full = np.linspace(x1e.min() + 1e-6, x1e.max() - 1e-6, Q_STATIONS)
    _, yW1 = resample_by_x(x1e, y1e, x_full)
    _, yW2 = resample_by_x(x2e, y2e, x_full)

    if np.mean(yW2 - yW1) < 0.0:
        # wall labels are correct (verified non-crossing, non-enclosing),
        # just swapped relative to which one is numerically "lower" --
        # width = yW2 - yW1 must be positive for the mass-flow sign
        # convention used throughout the solve.
        yW1, yW2 = yW2, yW1

    return x_full, yW1, yW2, spacing, height, cfg


def run_case(debug=False):
    x_full, yW1, yW2, spacing, height, cfg = build_passage(Q_STATIONS)
    geom = cfg["geometry"]["stator"]
    Q = Q_STATIONS
    dx = x_full[1] - x_full[0]

    stator_in = cfg["flow_stations"][1]  # station 0 is the stagnation ref, 1 is the real inlet
    T0 = cfg["inputs"]["inlet_property_2"]
    P0 = cfg["inputs"]["inlet_property_1"]
    u_in = stator_in["v"]
    u_max = 0.999 * np.sqrt(2.0 * GAMMA / (GAMMA - 1.0) * R_GAS * T0)

    rho_in = rho_of_speed(u_in, T0, P0)
    width_in = yW2[0] - yW1[0]
    m_bar = rho_in * u_in * width_in * height
    m_s = m_bar / (N - 1)

    Pos = np.zeros((N, Q))
    for k in range(Q):
        Pos[:, k] = np.linspace(yW1[k], yW2[k], N)
    U = np.full((N, Q), u_in)

    def geometry():
        Phi = np.zeros((N, Q))
        R = np.full((N, Q), 1e12)
        for j in range(N):
            yp, ypp = three_point_derivatives(Pos[j, :], x_full)
            Phi[j, :] = np.arctan(yp)
            R[j, :] = (1.0 + yp**2) ** 1.5 / np.where(np.abs(ypp) > 1e-9, ypp, 1e-9)
        return Phi, R

    err_r_history = []
    for r_iter in range(MAX_R):
        Phi, R = geometry()
        first_pass = r_iter == 0
        A = np.zeros((N, Q)) if first_pass else np.cos(Phi) / R

        # Not in the Gajoni thesis (no such artifact there -- its bend has
        # constant, smooth curvature by construction). Here the finite
        # difference stencil straddling the extension/blade-surface
        # junction sees an effectively discontinuous slope at the scale of
        # one station spacing, from the rounded LE/TE nose cap's tight
        # point spacing, producing curvature values far beyond anything
        # physically real (~1700 vs. ~1/leading_edge_radius~170). Clamp A
        # to that physical bound rather than chase exact geometric C1
        # continuity, which this blade's continuous camber doesn't afford.
        A_max = 1.0 / geom["leading_edge_radius"]
        A = np.clip(A, -A_max, A_max)

        n_smooth = 4
        if not first_pass:
            for _ in range(n_smooth):
                A_smooth = A.copy()
                for j in range(N):
                    A_smooth[j, 1:-1] = 0.25 * A[j, :-2] + 0.5 * A[j, 1:-1] + 0.25 * A[j, 2:]
                A = A_smooth
        A[:, 0] = 0.0
        A[:, -1] = 0.0

        for u_iter in range(MAX_U):
            l = np.zeros((N, Q))
            l[:, 1:] = np.cumsum(dx / np.maximum(np.cos(Phi[:, :-1]), 1e-3), axis=1)

            B = np.zeros((N, Q))
            if not first_pass:
                for j in range(N):
                    dUdl, _ = three_point_derivatives(U[j, :], l[j, :])
                    B[j, :] = np.sin(Phi[j, :]) * dUdl
                for _ in range(n_smooth):
                    B_smooth = B.copy()
                    for j in range(N):
                        B_smooth[j, 1:-1] = 0.25 * B[j, :-2] + 0.5 * B[j, 1:-1] + 0.25 * B[j, 2:]
                    B = B_smooth
                B[:, 0] = 0.0
                B[:, -1] = 0.0

            U_prev_uloop = U.copy()

            for k in range(1, Q):
                Uk = U[:, k].copy()
                for m_iter in range(MAX_M):
                    Uk = np.empty(N)
                    Uk[0] = U[0, k]
                    for j in range(1, N):
                        dr = Pos[j, k] - Pos[j - 1, k]
                        Abar = 0.5 * (A[j, k] + A[j - 1, k])
                        expo = np.clip(-Abar * dr, -50.0, 50.0)
                        Uk[j] = (Uk[j - 1] + 0.5 * dr * B[j - 1, k]) * np.exp(expo) \
                            + 0.5 * dr * B[j, k]
                        Uk[j] = np.clip(Uk[j], 1e-3, 0.9999 * u_max)

                    Uperp = Uk * np.cos(Phi[:, k])
                    rho_k = rho_of_speed(Uk, T0, P0)
                    m_k = np.trapezoid(rho_k * Uperp, Pos[:, k]) * height
                    err_m = abs(m_k - m_bar) / m_bar
                    if err_m < TOL_M:
                        break
                    factor = 2.0 - m_k / m_bar
                    if m_k < m_bar:
                        factor = min(1.2, factor)
                    else:
                        factor = max(0.8, factor)
                    U[0, k] = np.clip(factor * U[0, k], 1e-3, 0.9999 * u_max)
                U[:, k] = Uk

            err_U = np.max(np.abs(U - U_prev_uloop) / np.maximum(np.abs(U_prev_uloop), 1e-9))
            if err_U < TOL_U:
                break
        print(f"    r_iter={r_iter} u_iter={u_iter} err_U={err_U:.3e}", flush=True)

        Pos_target = Pos.copy()
        for k in range(1, Q):
            rho_k = rho_of_speed(U[:, k], T0, P0)
            Uperp = U[:, k] * np.cos(Phi[:, k])
            flux = rho_k * Uperp
            cum = np.zeros(N)
            cum[1:] = np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(Pos[:, k]))
            target_cum = np.arange(N) * m_s
            inv = interp1d(cum, Pos[:, k], bounds_error=False,
                            fill_value=(Pos[0, k], Pos[-1, k]))
            Pos_target[1:-1, k] = inv(target_cum[1:-1])

        T_now = np.maximum(T0 - U**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
        M_now = U / np.sqrt(GAMMA * R_GAS * T_now)
        Ave_mach = np.mean(M_now, axis=0)
        width = yW2 - yW1
        M_damping = np.maximum(1.0 - Ave_mach**2, 0.15)
        RFr = 1.0 / (1.0 + 0.25 * M_damping * (width / dx))

        Pos_new = Pos + RFr[None, :] * (Pos_target - Pos)
        Pos_new[0, :] = yW1
        Pos_new[-1, :] = yW2

        U_new = U.copy()
        for k in range(1, Q):
            interp_u = interp1d(Pos[:, k], U[:, k], bounds_error=False,
                                 fill_value=(U[0, k], U[-1, k]))
            U_new[:, k] = interp_u(Pos_new[:, k])
        U = U_new

        err_r = np.max(np.abs(Pos_new - Pos) / np.maximum(np.abs(Pos), 1e-9))
        Pos = Pos_new
        if not first_pass and err_r < TOL_R:
            print(f"  converged: r_iter={r_iter} err_r={err_r:.3e}")
            break

        err_r_history.append(err_r)
        if not first_pass and len(err_r_history) > STALL_WINDOW:
            if err_r_history[-1] > 0.8 * err_r_history[-STALL_WINDOW]:
                print(f"  [stalled: err_r={err_r:.3e} not improving over "
                      f"{STALL_WINDOW} iters -- stopping early]")
                break
    else:
        print(f"  [hit MAX_R={MAX_R} without converging, err_r={err_r:.3e}]")

    M_field = mach_of_speed(U, T0)

    if debug:
        return dict(x=x_full, yW1=yW1, yW2=yW2, Pos=Pos, U=U, M_field=M_field,
                    Phi=Phi, R=R, r_iter=r_iter, err_r=err_r, m_bar=m_bar,
                    T0=T0, P0=P0)
    return x_full, Pos, M_field


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out = run_case(debug=True)
    x = out["x"]
    M = out["M_field"]

    print(f"Mach at wall1 (final station): {M[0, -1]:.3f}")
    print(f"Mach at wall2 (final station): {M[-1, -1]:.3f}")
    print(f"Mach at mid streamline (final station): {M[N // 2, -1]:.3f}")
    print(f"Target stator-exit Mach (mean-line): "
          f"{sorted(glob.glob('meanline_stage_axial_*.yaml')) and '(see yaml station 2)'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for j in range(0, N, max(1, N // 6)):
        ax1.plot(x, M[j, :], lw=1.0)
    ax1.plot(x, M[0, :], "b-", lw=2, label="wall 1")
    ax1.plot(x, M[-1, :], "orange", lw=2, label="wall 2")
    ax1.set_xlabel("axial position [m]")
    ax1.set_ylabel("Mach number")
    ax1.legend()
    ax1.set_title("Mach number along streamlines")

    ax2.plot(x, out["yW1"], "b-", lw=2)
    ax2.plot(x, out["yW2"], "orange", lw=2)
    for j in range(N):
        ax2.plot(x, out["Pos"][j, :], color="green", lw=0.6, alpha=0.7)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("axial position [m]")
    ax2.set_ylabel("tangential position [m]")
    ax2.set_title("Converged streamlines")

    fig.tight_layout(pad=1)
    fig.savefig("stator_axial_slc_result.png", dpi=150)
    print("saved stator_axial_slc_result.png")
