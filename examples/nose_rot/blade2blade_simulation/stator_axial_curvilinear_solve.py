"""
Blade-to-blade SLC solve on the curvilinear (flow-following) mesh from
stator_axial_curvilinear_mesh.py.

Generalizes the Gajoni-style three-loop solver (m-loop/U-loop/r-loop,
reused from figure10_slc.py and stator_axial_slc_solve.py) to a passage
whose quasi-orthogonal cuts tilt to follow the local mean-flow direction,
instead of staying fixed (which degenerates once the flow turns far from
the fixed reference -- the bug found and fixed earlier this session).

Key generalizations vs. the constant-x solver:
  - Pos[j,k] is now the signed distance along the LOCAL normal direction
    at station k (from the curvilinear mesh), not a y-coordinate. The
    physical position is xm[k] + Pos[j,k]*normal[k].
  - Phi[j,k] (streamline heading) and R[j,k] (radius of curvature) come
    from the PARAMETRIC curvature formulas applied to each streamline's
    actual (X,Y) path vs. arc-length station s -- this single formula
    covers what Gajoni needed two branches (Cartesian/polar) for.
  - The physics (A, B, mass flux) must be evaluated relative to the LOCAL
    mean-flow direction Phi_mean[k], not the absolute heading: using
    Phi_rel = Phi - Phi_mean keeps cos(Phi_rel) close to 1 even as the
    passage turns 70 deg, which is the whole point of this rework (using
    absolute Phi here would silently reintroduce the same degeneracy the
    curvilinear mesh was built to avoid).
"""

import numpy as np
from scipy.interpolate import interp1d

from stator_axial_curvilinear_mesh import build_curvilinear_mesh, three_point_derivatives

GAMMA = 1.4
R_GAS = 287.05  # air, J/(kg K)

N = 15
Q_STATIONS = 80

TOL_M = TOL_U = TOL_R = 1e-3
MAX_M = MAX_U = MAX_R = 500
STALL_WINDOW = 15


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rho_of_speed(u, T0, P0):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    return (P0 * (T / T0) ** (GAMMA / (GAMMA - 1.0))) / (R_GAS * T)


def mach_of_speed(u, T0):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    c = np.sqrt(GAMMA * R_GAS * T)
    return u / c


def run_case(debug=False):
    mesh = build_curvilinear_mesh(Q_STATIONS=Q_STATIONS)
    s = mesh["s"]
    xm, ym = mesh["xm"], mesh["ym"]
    normal = mesh["normal"]
    Phi_mean = mesh["Phi_mean"]
    t1, t2 = mesh["t1"], mesh["t2"]
    spacing, height, cfg = mesh["spacing"], mesh["height"], mesh["cfg"]
    geom = cfg["geometry"]["stator"]
    Q = Q_STATIONS

    stator_in = cfg["flow_stations"][1]
    T0 = cfg["inputs"]["inlet_property_2"]
    P0 = cfg["inputs"]["inlet_property_1"]
    u_in = stator_in["v"]
    u_max = 0.999 * np.sqrt(2.0 * GAMMA / (GAMMA - 1.0) * R_GAS * T0)

    rho_in = rho_of_speed(u_in, T0, P0)
    width_in = t2[0] - t1[0]
    m_bar = rho_in * u_in * width_in * height
    m_s = m_bar / (N - 1)

    Pos = np.zeros((N, Q))
    for k in range(Q):
        Pos[:, k] = np.linspace(t1[k], t2[k], N)
    U = np.full((N, Q), u_in)

    A_max = 1.0 / geom["leading_edge_radius"]

    def physical_xy():
        # xy[j,k,:] = mean-line point + Pos[j,k]*normal[k]
        return (np.stack([xm, ym], axis=1)[None, :, :]
                + Pos[:, :, None] * normal[None, :, :])

    def geometry():
        xy = physical_xy()
        Phi = np.zeros((N, Q))
        R = np.full((N, Q), 1e12)
        for j in range(N):
            Xp, Xpp = three_point_derivatives(xy[j, :, 0], s)
            Yp, Ypp = three_point_derivatives(xy[j, :, 1], s)
            Phi[j, :] = np.arctan2(Yp, Xp)
            num = (Xp**2 + Yp**2) ** 1.5
            den = Xp * Ypp - Yp * Xpp
            R[j, :] = num / np.where(np.abs(den) > 1e-9, den, 1e-9)
        Phi_rel = wrap_angle(Phi - Phi_mean[None, :])
        return Phi, Phi_rel, R, xy

    err_r_history = []
    for r_iter in range(MAX_R):
        Phi, Phi_rel, R, xy = geometry()
        first_pass = r_iter == 0
        A = np.zeros((N, Q)) if first_pass else np.cos(Phi_rel) / R
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
            l[:, 1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), axis=1)

            B = np.zeros((N, Q))
            if not first_pass:
                for j in range(N):
                    dUdl, _ = three_point_derivatives(U[j, :], l[j, :])
                    B[j, :] = np.sin(Phi_rel[j, :]) * dUdl
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

                    Uperp = Uk * np.cos(Phi_rel[:, k])
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
            Uperp = U[:, k] * np.cos(Phi_rel[:, k])
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
        width = t2 - t1
        ds = np.gradient(s)
        M_damping = np.maximum(1.0 - Ave_mach**2, 0.15)
        RFr = 1.0 / (1.0 + 0.25 * M_damping * (width / np.maximum(ds, 1e-9)))

        Pos_new = Pos + RFr[None, :] * (Pos_target - Pos)
        Pos_new[0, :] = t1
        Pos_new[-1, :] = t2

        U_new = U.copy()
        for k in range(1, Q):
            interp_u = interp1d(Pos[:, k], U[:, k], bounds_error=False,
                                 fill_value=(U[0, k], U[-1, k]))
            U_new[:, k] = interp_u(Pos_new[:, k])
        U = U_new

        err_r = np.max(np.abs(Pos_new - Pos) / np.maximum(np.abs(width)[None, :], 1e-9))
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
        return dict(s=s, xm=xm, ym=ym, normal=normal, t1=t1, t2=t2,
                    Pos=Pos, U=U, M_field=M_field, Phi=Phi, R=R,
                    r_iter=r_iter, err_r=err_r, m_bar=m_bar, T0=T0, P0=P0)
    return s, Pos, M_field


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    out = run_case(debug=True)
    s = out["s"]
    M = out["M_field"]

    print(f"Mach at wall1 (final station): {M[0, -1]:.3f}")
    print(f"Mach at wall2 (final station): {M[-1, -1]:.3f}")
    print(f"Mach at mid streamline (final station): {M[N // 2, -1]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for j in range(0, N, max(1, N // 6)):
        ax1.plot(s, M[j, :], lw=1.0)
    ax1.plot(s, M[0, :], "b-", lw=2, label="wall 1")
    ax1.plot(s, M[-1, :], "orange", lw=2, label="wall 2")
    ax1.set_xlabel("meanline arc length [m]")
    ax1.set_ylabel("Mach number")
    ax1.legend()
    ax1.set_title("Mach number along streamlines")

    xy = (np.stack([out["xm"], out["ym"]], axis=1)[None, :, :]
          + out["Pos"][:, :, None] * out["normal"][None, :, :])
    for j in range(N):
        ax2.plot(xy[j, :, 0], xy[j, :, 1], color="green", lw=0.6, alpha=0.7)
    ax2.plot(xy[0, :, 0], xy[0, :, 1], "b-", lw=2)
    ax2.plot(xy[-1, :, 0], xy[-1, :, 1], "orange", lw=2)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("axial position [m]")
    ax2.set_ylabel("tangential position [m]")
    ax2.set_title("Converged streamlines")

    fig.tight_layout(pad=1)
    fig.savefig("stator_axial_curvilinear_result.png", dpi=150)
    print("saved stator_axial_curvilinear_result.png")
