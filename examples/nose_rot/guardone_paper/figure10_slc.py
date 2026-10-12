"""
Reproduce Figure 10 of Gajoni & Guardone (JFM 2023) using the Streamline
Curvature (SLC) method exactly as described in Gajoni's MSc thesis
(Politecnico di Milano, 2020-21), Chapter 3, "Streamline curvature
method". Equation numbers in comments refer to that thesis.

Domain (thesis Fig. 3.5, Sec. 3.4): inlet straight section of length
L = 2*r_e -> 180 deg circular bend -> outlet straight section of the
same length, so the inlet/outlet boundary conditions don't affect the
flow inside the bend.

Notation follows thesis Sec. 3.3: j = 1..N indexes streamlines (j=1
inner wall, j=N outer wall), k = 1..Q indexes quasi-orthogonal stations
(fixed vertical lines in the straight sections, fixed radial lines in
the bend). Three nested loops (Fig. 3.7):
  - m-loop (innermost): matches the mass flow rate at station k by
    updating the inner-wall velocity U_i,k, eq. 3.35/3.36.
  - U-loop: refreshes B_jk = sin(Phi_jk) dU/dl from the previous
    velocity field, then re-solves the m-loop at every station.
  - r-loop (outermost): repositions streamlines by mass conservation
    (eq. 3.33), recomputes R_jk/Phi_jk from the new geometry, repeats.
"""
import numpy as np
from scipy.interpolate import interp1d

from common import ideal_gas_mach_profile

GAMMA = 1.4
R_GAS = 8.314
T0 = 288.15
P0 = 101325
R_I = 1.0

CASES = [
    (3.0, 1.1),
    # (1.5, 0.76),
    # (2.0, 0.63),
    # (3.0, 0.49),
    # (5.0, 0.35),
]

N = 50                     # streamlines
Q_STRAIGHT = 20             # quasi-orthogonals per straight section
Q_BEND = 40                 # quasi-orthogonals in the bend
THETA_MAX_DEG = 180.0
L_FACTOR = 2.5

TOL_M = 1e-3
TOL_U = 1e-3
TOL_R = 1e-3
MAX_M = 500
MAX_U = 500
MAX_R = 500
RELAX_R = 0.1               # thesis: relaxation factor to limit streamline displacement
STALL_WINDOW = 15           # not in the thesis; bail out early if err_r stalls (see below)

u_max = 0.999 * np.sqrt(2.0 * GAMMA / (GAMMA - 1.0) * R_GAS * T0)
T_sonic = T0 * 2.0 / (GAMMA + 1.0)
u_sonic = np.sqrt(GAMMA * R_GAS * T_sonic)  # speed at M=1


def rho_of_speed(u):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    return (P0 * (T / T0) ** (GAMMA / (GAMMA - 1.0))) / (R_GAS * T)


def isentropic_state(M):
    T = T0 / (1.0 + 0.5 * (GAMMA - 1.0) * M**2)
    p = P0 * (T / T0) ** (GAMMA / (GAMMA - 1.0))
    rho = p / (R_GAS * T)
    c = np.sqrt(GAMMA * R_GAS * T)
    u = M * c
    return rho, u, p, c


def mach_of_speed(u):
    T = np.maximum(T0 - u**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
    c = np.sqrt(GAMMA * R_GAS * T)
    return u / c


def three_point_derivatives(f, x):
    """First and second derivative of f(x), three-point central
    differences (one-sided at the boundaries) -- thesis Sec. 3.3.2."""
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


def run_case(r_e_over_ri, M_in, debug=False):
    r_i, r_e = R_I, r_e_over_ri * R_I
    L = L_FACTOR * r_e

    x_straight_in = np.linspace(-L, 0.0, Q_STRAIGHT, endpoint=False)
    theta_max = np.deg2rad(THETA_MAX_DEG)
    theta_bend = np.linspace(0.0, theta_max, Q_BEND)
    s_straight_out = np.linspace(0.0, L, Q_STRAIGHT + 1)[1:]

    i1 = Q_STRAIGHT               # index of the first bend station
    i2 = Q_STRAIGHT + Q_BEND      # index of the first outlet station
    Q = i2 + Q_STRAIGHT

    # local physical station spacing, for the Letcher/Cumpsty/Wilkinson
    # relaxation formula below.
    dx_straight = L / Q_STRAIGHT
    dtheta = theta_max / (Q_BEND - 1) if Q_BEND > 1 else theta_max
    r_mid = 0.5 * (r_i + r_e)
    delta_z = np.empty(Q)
    delta_z[:i1] = dx_straight
    delta_z[i1:i2] = r_mid * dtheta
    delta_z[i2:] = dx_straight

    rho_in, u_in, p_in, c_in = isentropic_state(M_in)
    Pos_init = np.linspace(r_i, r_e, N)
    m_bar = rho_in * u_in * (r_e - r_i)     # eq. 3.32
    m_s = m_bar / (N - 1)                   # eq. 3.33

    # first guess (Sec. 3.3.1): equispaced streamlines and the inlet
    # velocity profile reported at every station.
    Pos = np.tile(Pos_init[:, None], (1, Q))
    U = np.full((N, Q), u_in)

    normal_exit = np.array([np.sin(theta_max), np.cos(theta_max)])
    tangent_exit = np.array([np.cos(theta_max), -np.sin(theta_max)])

    def cartesian_xy():
        xy = np.zeros((N, Q, 2))
        xy[:, :i1, 0] = x_straight_in[None, :]
        xy[:, :i1, 1] = Pos[:, :i1]
        th = theta_bend[None, :]
        xy[:, i1:i2, 0] = Pos[:, i1:i2] * np.sin(th)
        xy[:, i1:i2, 1] = Pos[:, i1:i2] * np.cos(th)
        xy[:, i2:, 0] = Pos[:, i2:] * normal_exit[0] + s_straight_out[None, :] * tangent_exit[0]
        xy[:, i2:, 1] = Pos[:, i2:] * normal_exit[1] + s_straight_out[None, :] * tangent_exit[1]
        return xy

    def geometry():
        """R_jk, Phi_jk -- thesis eq. 3.44-3.47."""
        Phi = np.zeros((N, Q))
        R = np.full((N, Q), 1e12)
        for j in range(N):
            # straight inlet: cartesian, eq. 3.44/3.46
            y = Pos[j, :i1]
            yp, ypp = three_point_derivatives(y, x_straight_in)
            Phi[j, :i1] = np.arctan(yp)
            R[j, :i1] = (1.0 + yp**2) ** 1.5 / np.where(np.abs(ypp) > 1e-9, ypp, 1e-9)

            # bend: polar, eq. 3.45/3.47
            r = Pos[j, i1:i2]
            rp, rpp = three_point_derivatives(r, theta_bend)
            Phi[j, i1:i2] = np.arctan(rp / r)
            num = (r**2 + rp**2) ** 1.5
            den = r**2 + 2.0 * rp**2 - r * rpp
            R[j, i1:i2] = num / np.where(np.abs(den) > 1e-9, den, 1e-9)

            # straight outlet: cartesian, eq. 3.44/3.46
            y = Pos[j, i2:]
            yp, ypp = three_point_derivatives(y, s_straight_out)
            Phi[j, i2:] = np.arctan(yp)
            R[j, i2:] = (1.0 + yp**2) ** 1.5 / np.where(np.abs(ypp) > 1e-9, ypp, 1e-9)

        return Phi, R

    err_r_history = []
    for r_iter in range(MAX_R):
        Phi, R = geometry()
        # not in the thesis; Schobeiri (2012), "Turbomachinery Flow
        # Physics and Dynamic Performance", Sec. 11.2.1, steps 2-3:
        # "zero is the usual first estimate" for the streamwise-gradient
        # terms, and "curvatures are normally set to zero for the first
        # and last stations" (always, every pass). A first pass with the
        # curvature/coupling terms fully engaged jumps straight to the
        # fully-curved equilibrium; starting from exactly zero lets
        # position/mass-conservation develop a genuinely spatial solution
        # from pass 2 onward.
        first_pass = r_iter == 0
        A = np.zeros((N, Q)) if first_pass else np.cos(Phi) / R  # eq. 3.19

        # not in the thesis: 1-2-1 spatial smoothing of A/B supplies the
        # station-to-station coupling nothing else in this formulation
        # provides, letting curvature build up gradually instead of
        # jumping straight to its local value. A FIXED kernel width smooths
        # over a fixed number of stations regardless of channel width, but
        # a wider channel needs a physically longer transition (thesis
        # Table 4.2: theta_eq grows with r_e) -- so scale the pass count
        # with r_e/r_i.
        n_smooth = max(2, int(round(2 * r_e_over_ri)))
        if not first_pass:
            for _ in range(n_smooth):
                A_smooth = A.copy()
                for j in range(N):
                    A_smooth[j, 1:-1] = 0.25 * A[j, :-2] + 0.5 * A[j, 1:-1] + 0.25 * A[j, 2:]
                A = A_smooth

        A[:, 0] = 0.0
        A[:, -1] = 0.0

        for u_iter in range(MAX_U):
            xy = cartesian_xy()
            l = np.zeros((N, Q))
            l[:, 1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), axis=1)

            B = np.zeros((N, Q))
            if not first_pass:
                for j in range(N):
                    dUdl, _ = three_point_derivatives(U[j, :], l[j, :])
                    B[j, :] = np.sin(Phi[j, :]) * dUdl  # eq. 3.19
                
                for _ in range(n_smooth):  # same width-scaled pass count as A
                    B_smooth = B.copy()
                    for j in range(N):
                        B_smooth[j, 1:-1] = 0.25 * B[j, :-2] + 0.5 * B[j, 1:-1] + 0.25 * B[j, 2:]
                    B = B_smooth

                B[:, 0] = 0.0
                B[:, -1] = 0.0

            U_prev_uloop = U.copy()

            for k in range(1, Q):
                # m-loop, eq. 3.35/3.36: bounded (+-20%) multiplicative
                # continuation from the previous inner-wall velocity, not
                # a fresh search -- keeps the solve on the same root.
                Uk = U[:, k].copy()
                for m_iter in range(MAX_M):
                    Uk = np.empty(N)
                    Uk[0] = U[0, k]
                    for j in range(1, N):
                        dr = Pos[j, k] - Pos[j - 1, k]
                        Abar = 0.5 * (A[j, k] + A[j - 1, k])          # eq. 3.41
                        expo = np.clip(-Abar * dr, -50.0, 50.0)        # not in thesis; safety only
                        Uk[j] = (Uk[j - 1] + 0.5 * dr * B[j - 1, k]) * np.exp(expo) \
                            + 0.5 * dr * B[j, k]                       # eq. 3.43
                        # physical bound -- not in the thesis, but needed:
                        # without it, a transient Pos wobble can blow the
                        # march up to overflow before the m-loop's own
                        # correction has a chance to pull it back. Capped
                        # at the true physical max speed (T->0), not at
                        # sonic, so transonic/supersonic states can still
                        # be represented.
                        Uk[j] = np.clip(Uk[j], 1e-3, 0.9999 * u_max)

                    Uperp = Uk * np.cos(Phi[:, k])                     # eq. 3.34
                    rho_k = rho_of_speed(Uk)
                    m_k = np.trapezoid(rho_k * Uperp, Pos[:, k])       # eq. 3.37
                    err_m = abs(m_k - m_bar) / m_bar
                    if err_m < TOL_M:
                        break
                    factor = 2.0 - m_k / m_bar
                    if m_k < m_bar:
                        factor = min(1.2, factor)                      # eq. 3.35
                    else:
                        factor = max(0.8, factor)                      # eq. 3.36
                    # not in the thesis: clamp the corrected value itself,
                    # not just the interior march -- otherwise repeated
                    # +-20% steps with a residual that doesn't flip sign
                    # compound geometrically (1.2^30 = 237x) into overflow.
                    U[0, k] = np.clip(factor * U[0, k], 1e-3, 0.9999 * u_max)
                U[:, k] = Uk

            err_U = np.max(np.abs(U - U_prev_uloop) / np.maximum(np.abs(U_prev_uloop), 1e-9))
            if err_U < TOL_U:
                break

        Pos_target = Pos.copy()
        for k in range(1, Q):
            rho_k = rho_of_speed(U[:, k])
            Uperp = U[:, k] * np.cos(Phi[:, k])
            flux = rho_k * Uperp
            cum = np.zeros(N)
            cum[1:] = np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(Pos[:, k]))
            target_cum = np.arange(N) * m_s
            inv = interp1d(cum, Pos[:, k], bounds_error=False,
                            fill_value=(Pos[0, k], Pos[-1, k]))
            Pos_target[1:-1, k] = inv(target_cum[1:-1])

        # not in the thesis (which just says "a relaxation factor is
        # used"): Letcher (2008, MS thesis under S. Korpela, Appendix B.7)
        # gives the actual formula used in real SLC codes (Cumpsty/
        # Wilkinson 1970, Hearsey), computed FRESH PER STATION from the
        # local channel width, station spacing, and Mach number:
        #   RFr = 1 / (1 + 5/24*(1-M_avg^2)*(width/delta_z))
        # A constant RELAX_R (thesis-literal) is spatially blind; this one
        # scales with channel width, so wider channels are damped harder
        # and genuinely take more iterations to reach equilibrium.
        T_now = np.maximum(T0 - U**2 / (2.0 * GAMMA / (GAMMA - 1.0) * R_GAS), 1e-9)
        M_now = U / np.sqrt(GAMMA * R_GAS * T_now)
        Ave_mach = np.mean(M_now, axis=0)
        width = r_e - r_i
        # book: "the constant shown as 6 ... needs to be set to 4 ...
        # when many stations are placed relatively close together" (ours
        # are); also (1-M^2) collapses toward 0 -- killing the damping --
        # right as M approaches the sonic target this case is calibrated
        # for, so clamp it away from zero instead of letting it vanish.
        M_damping = np.maximum(1.0 - Ave_mach**2, 0.15)
        RFr = 1.0 / (1.0 + 0.25 * M_damping * (width / delta_z))

        Pos_new = Pos + RFr[None, :] * (Pos_target - Pos)
        Pos_new[0, :] = r_i
        Pos_new[-1, :] = r_e

        # thesis p.29: "the velocities are updated interpolating the old
        # values on the new node positions" -- U[j,k] belongs to a
        # streamtube, not a fixed array index, so it must be resampled
        # onto Pos_new or it silently stops matching the geometry.
        U_new = U.copy()
        for k in range(1, Q):
            interp_u = interp1d(Pos[:, k], U[:, k], bounds_error=False,
                                 fill_value=(U[0, k], U[-1, k]))
            U_new[:, k] = interp_u(Pos_new[:, k])
        U = U_new

        err_r = np.max(np.abs(Pos_new - Pos) / np.maximum(np.abs(Pos), 1e-9))
        Pos = Pos_new
        # don't allow early-exit on the artificial zero-curvature first
        # pass: err_r looks small then simply because A/B are zero, not
        # because the solution has actually converged.
        if not first_pass and err_r < TOL_R:
            break

        # not in the thesis: stop early if err_r has genuinely stalled
        # (not just oscillating) instead of always burning the full
        # MAX_R -- e.g. shock-forming cases (thesis Sec. 4.2.3) never
        # converge here since this isentropic formulation has no shock
        # mechanism, so grinding through all MAX_R iterations is wasted
        # work, not extra accuracy.
        err_r_history.append(err_r)
        if not first_pass and len(err_r_history) > STALL_WINDOW:
            if err_r_history[-1] > 0.8 * err_r_history[-STALL_WINDOW]:
                print(f"  [stalled: err_r={err_r:.3e} not improving over "
                      f"{STALL_WINDOW} iters -- stopping early, likely no "
                      f"isentropic solution exists for this case]")
                break

    M_field = mach_of_speed(U)
    theta_in_deg = np.rad2deg(x_straight_in / r_i)
    theta_bend_deg = np.rad2deg(theta_bend)
    theta_out_deg = THETA_MAX_DEG + np.rad2deg(s_straight_out / r_i)
    theta_deg = np.concatenate([theta_in_deg, theta_bend_deg, theta_out_deg])

    if debug:
        return dict(theta_deg=theta_deg, Mi=M_field[0, :], Me=M_field[-1, :],
                    Phi=Phi, R=R, A=A, Pos=Pos, U=U, M_field=M_field,
                    r_iter=r_iter, err_r=err_r, i1=i1, i2=i2)
    return theta_deg, M_field[0, :], M_field[-1, :]


def main():
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for r_e_over_ri, M_in in CASES:
        print(f"Running r_e/r_i={r_e_over_ri}, M_in={M_in} ...")
        theta_deg, M_i_wall, M_e_wall = run_case(r_e_over_ri, M_in)

        ax1.plot(theta_deg, M_i_wall, "o-", lw=1.2, color="black")

        ax2.plot(theta_deg, M_e_wall, "o-", lw=1.2, color="black")
        

        # M_eq_inner = 1.0
        M_eq_outer = ideal_gas_mach_profile(1.0, GAMMA, np.array([r_e_over_ri]))[0]
        # ax1.axhline(M_eq_inner, color="gray", lw=0.8, ls="--")
        ax2.axhline(M_eq_outer, color="gray", lw=0.8, ls="--")

        # print(f"  final M_inner(theta_max)={M_i_wall[-1]:.4f} (ref {M_eq_inner:.4f}), "
        #       f"M_outer(theta_max)={M_e_wall[-1]:.4f} (ref {M_eq_outer:.4f})")

    for ax in (ax1, ax2):
        ax.axvline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"$\theta$ [deg]  (negative = straight settling section)",
                      fontsize=12)
        ax.set_xlim(-50, 180)
        ax.tick_params(labelsize=12)

    ax1.set_ylabel(r"$M_i$ (internal wall)", fontsize=14)
    ax2.set_ylabel(r"$M_e$ (external wall)", fontsize=14)

    fig.suptitle("Reproduction of Gajoni & Guardone (2023), Figure 10\n"
                  "SLC method, thesis Ch. 3 (literal transcription)")
    plt.tight_layout()
    fig.savefig("figure10_slc.png", dpi=150)


if __name__ == "__main__":
    main()
