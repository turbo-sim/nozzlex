import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import jaxprop as jxp
import matplotlib.pyplot as plt
import nurbspy.jax as nrb

jxp.set_plot_options(grid=False)


# TODO CHECK this chaptgpt conversation about how to integrate the ODE. the recommendation is to use meridional (arclength) coordinate for the geometyr functions and integrate in physicla space
# https://chatgpt.com/c/6903b75b-be7c-8325-938e-6647f18ab50e


# -----------------------------------------------------------------------------
# Geometry generator
# -----------------------------------------------------------------------------
def make_vaneless_channel_geometry(geometry, tol=1e-6):
    """
    Construct a parametric representation of a 2D axisymmetric vaneless channel
    (e.g., diffuser, inter-blade passage, U-turn, 90-degree bend) using two coupled NURBS
    curves:

      - a channel midline curve defining the (z-r) coordinates in the meridional plane,
      - a channel width curve defining the local width b(s) along the channel.

    The function returns a callable geometric evaluator `geom_handle(s)` that
    provides all local geometric quantities as continuous functions of the
    meridional arclength coordinate `s`.

    ---------------------------------------------------------------------------
    Construction procedure
    ---------------------------------------------------------------------------
    1. The **channel midline** is constructed from four control points:
           (z_in, r_in),
           (z_in + td_in*cos(phi_in), r_in + td_in*sin(phi_in)),
           (z_out - td_out*cos(phi_out), r_out - td_out*sin(phi_out)),
           (z_out, r_out)
       where `td_in` and `td_out` are pseudo-tangential control distances and
       `phi_in`, `phi_out` are inlet and outlet inclination angles.

    2. The midline curve is reparametrized by meridional **arclength** s using
       numerical ODE integration of ds/du = ||C'(u)||, yielding the mapping
       u(s) and total arclength s_total.

    3. The **channel width** curve is constructed in (s, b)
       space, with the first coordinate representing the physical meridional
       distance along the midline, and the second the full channel height b(s).

       For each s, b(s) is evaluated by inverting the NURBS parameterization
       via the mapping u_b(s).

    4. The final evaluator returns geometric quantities such as coordinates,
       normal vector, cross-sectional area, and local slope angle as functions
       of arclength.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    geometry : dict
        Dictionary containing geometric parameters:
            {
                "z_in", "z_out" : inlet/outlet axial coordinates [m],
                "r_in", "r_out" : inlet/outlet radii [m],
                "b_in", "b_out" : inlet/outlet channel heights [m],
                "phi_in", "phi_out" : inlet/outlet wall angles [rad],
                "td_in", "td_out" : inlet/outlet tangent distances [m]
            }

    tol : float, optional
        Relative and absolute tolerance for the arclength ODE integration
        (default: 1e-6).

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    geom_handle : callable
        A JAX-compatible function `geom_handle(s)` that returns a dictionary
        of geometric quantities evaluated at arclength positions `s`
        (scalar or array-like):

            {
                "z"        : axial coordinate z(s),
                "r"        : radial coordinate r(s),
                "b"        : local channel height b(s),
                "A"        : cross-sectional area A(s) = 2π·r·b,
                "phi"      : local meridional inclination atan(dr/ds, dz/ds),
                "dzds"     : derivative dz/ds,
                "drds"     : derivative dr/ds,
                "dbds"     : derivative db/ds,
                "dAds"     : derivative dA/ds,
                "z_shroud" : shroud wall axial coordinate z + n_z·b/2,
                "r_shroud" : should wall radial coordinate r + n_r·b/2,
                "z_hub"    : hub wall axial coordinate z - n_z·b/2,
                "r_hub"    : hub wall radial coordinate r - n_r·b/2,
                "s_total"  : total meridional arclength
            }

        The callable is fully compatible with `jax.jit`, `jax.vmap`,
        and automatic differentiation.

    s_total : float
        The total meridional arclength of the channel midline.
    """

    # Construct the channel midline curve
    z = jnp.array(
        [
            geometry["z_in"],
            geometry["z_in"] + geometry["td_in"] * jnp.cos(geometry["phi_in"]),
            geometry["z_out"] - geometry["td_out"] * jnp.cos(geometry["phi_out"]),
            geometry["z_out"],
        ]
    )

    r = jnp.array(
        [
            geometry["r_in"],
            geometry["r_in"] + geometry["td_in"] * jnp.sin(geometry["phi_in"]),
            geometry["r_out"] - geometry["td_out"] * jnp.sin(geometry["phi_out"]),
            geometry["r_out"],
        ]
    )

    P_midline = jnp.vstack([z, r])
    channel_midline = nrb.NurbsCurve(control_points=P_midline, degree=2)

    # Reparametrize midline by arclength
    u_of_s, s_total = channel_midline.reparametrize_by_arclength(tol=tol)

    # Construct the channel width curve
    P_width = jnp.array(
        [
            [0.00 * s_total, geometry["b_in"]],
            # [0.30*s_total, geometry["b_in"]],
            # [0.70*s_total, geometry["b_out"]],
            [1.00 * s_total, geometry["b_out"]],
        ]
    ).T
    channel_width = nrb.NurbsCurve(control_points=P_width)

    # Reparametrize channel width by coordinate
    u_of_x = channel_width.reparametrize_by_coordinate(dim=0)

    # Define geometry evaluation function
    def geom_handle(s):
        """Return geometric properties at given arclength s (scalar or array)."""

        # Parameterize back to u-coordinate
        s = jnp.atleast_1d(s)
        u_midline = u_of_s(s)
        u_width = u_of_x(s)

        # Compute channel midline geometry and derivatives
        zr = channel_midline.get_value(u_midline)
        n = channel_midline.get_normal(u_midline)
        dzr_du = channel_midline.get_derivative(u_midline, order=1)
        dzr_du_mag = jnp.linalg.norm(dzr_du, axis=0)
        dzr_ds = dzr_du / dzr_du_mag
        z, r = zr[0], zr[1]
        dzds, drds = dzr_ds[0], dzr_ds[1]
        phi = jnp.arctan2(drds, dzds)

        # Compute channel width and derivative
        _, b = channel_width.get_value(u_width)
        dsb_du = channel_width.get_derivative(u_width, order=1)
        dsdu = dsb_du[0, :]  # first coordinate derivative
        dbdu = dsb_du[1, :]  # second coordinate derivative
        dbds = dbdu / dsdu

        # Compute area and its derivative
        A = 2.0 * jnp.pi * r * b
        dAds = 2.0 * jnp.pi * (drds * b + r * dbds)

        # Compute hub/shroud wall coordinates
        z_hub = z - 0.5 * n[0] * b
        r_hub = r - 0.5 * n[1] * b
        z_shroud = z + 0.5 * n[0] * b
        r_shroud = r + 0.5 * n[1] * b

        # Combine results
        geom_dict = {
            "z": z,
            "r": r,
            "b": b,
            "A": A,
            "phi": phi,
            "dzds": dzds,
            "drds": drds,
            "dbds": dbds,
            "dAds": dAds,
            "z_shroud": z_shroud,
            "r_shroud": r_shroud,
            "z_hub": z_hub,
            "r_hub": r_hub,
            "s_total": s_total,
        }

        # If s was scalar, return scalar values instead of 1-element arrays
        geom_dict = {
            k: (v.squeeze() if v.shape == (1,) else v) for k, v in geom_dict.items()
        }
        return geom_dict

    geom_handle.s_total = 0.999 * s_total

    return jax.jit(geom_handle)


def plot_vaneless_channel(
    geom_handle,
    n_points: int = 200,
    plot_midline: bool = True,
    plot_hub: bool = True,
    plot_shroud: bool = True,
    connect_edges: bool = True,
    fig=None,
    ax=None,
):
    """
    Plot the meridional channel geometry using a geometry handle from
    make_vaneless_channel_geometry().

    Parameters
    ----------
    geom_handle : callable
        Geometry evaluator function geom_handle(s) → dict of geometry properties.
    n_points : int, optional
        Number of evaluation points along the channel (default 200).
    plot_midline : bool, optional
        Whether to plot the channel midline (default True).
    plot_hub : bool, optional
        Whether to plot the hub wall (default True).
    plot_shroud : bool, optional
        Whether to plot the shroud wall (default True).
    connect_edges : bool, optional
        Whether to connect inlet and outlet edges (default True).
    fig, ax : matplotlib Figure and Axes, optional
        Existing figure/axes to plot on. If None, new ones are created.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    # Create figure and axes if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Axial coordinate z [m]")
        ax.set_ylabel("Radial coordinate r [m]")

    # Sample the geometry along the midline
    s = jnp.linspace(0.0, geom_handle.s_total, n_points)
    geom = geom_handle(s)

    # Extract geometry data
    z_mid = geom["z"]
    r_mid = geom["r"]
    z_shroud = geom["z_shroud"]
    r_shroud = geom["r_shroud"]
    z_hub = geom["z_hub"]
    r_hub = geom["r_hub"]

    # Plot requested curves
    if plot_hub:
        ax.plot(z_hub, r_hub, color="black", linestyle="-")
    if plot_shroud:
        ax.plot(z_shroud, r_shroud, color="black", linestyle="-")
    if plot_midline:
        ax.plot(z_mid, r_mid, color="black", linestyle="-.")

    # Optionally connect inlet/outlet edges
    if connect_edges:
        ax.plot(
            [z_hub[0], z_shroud[0]],
            [r_hub[0], r_shroud[0]],
            color="black",
            linestyle="-",
        )
        ax.plot(
            [z_hub[-1], z_shroud[-1]],
            [r_hub[-1], r_shroud[-1]],
            color="black",
            linestyle="-",
        )

    return fig, ax


# -----------------------------------------------------------------------------
# Main API to the vaneless diffuser model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def solve_vaneless_channel_model(
    params,
    fluid,
    geom_handle,
    solver_name: str = "Dopri5",
    adjoint_name: str = "DirectAdjoint",
    number_of_points: int | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-9,
):
    """Evaluate one-dimensional flow in an vaneless channel"""

    # Rename parameters
    p0_in = params["p0_in"]
    T0_in = params["T0_in"]
    v_in = params["v_in"]
    alpha_in = params["alpha_in"]
    Cf = params["Cf"]
    q_w = params["q_w"]
    geom = params["geometry"]

    # Compute initial conditions for ODE system
    state0_in = fluid.get_state(jxp.PT_INPUTS, p0_in, T0_in)
    h_in = state0_in.h - 0.5 * v_in**2
    s_in = state0_in.s
    state_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)
    p_in = state_in["p"]
    d_in = state_in["d"]
    v_m_in = v_in * jnp.cos(alpha_in)
    v_t_in = v_in * jnp.sin(alpha_in)
    y0 = jnp.array([v_m_in, v_t_in, d_in, p_in, 0.0, 0.0])

    # Group the ODE system constant parameters
    args = (Cf, q_w, p0_in, p_in, fluid, geom_handle)
    s_max = geom_handle.s_total

    # Create and configure the solver
    solver = jxp.make_diffrax_solver(solver_name)
    adjoint = jxp.make_diffrax_adjoint(adjoint_name)
    term = dfx.ODETerm(evaluate_vaneless_channel_ode)
    ctrl = dfx.PIDController(rtol=rtol, atol=atol)
    if number_of_points is not None:
        ts = jnp.linspace(0.0, s_max, number_of_points)
        saveat = dfx.SaveAt(ts=ts, dense=False, fn=postprocess_ode)
    else:
        saveat = dfx.SaveAt(t1=True, fn=postprocess_ode)

    # Solve the ODE system
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=s_max,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        max_steps=200,
        adjoint=adjoint,
    )

    return sol


# -----------------------------------------------------------------------------
# Right hand side of the diffuser ODE system
# -----------------------------------------------------------------------------
def evaluate_vaneless_channel_ode(t, y, args):
    # Rename from ODE terminology to physical variables
    meridional_length = t
    v_m, v_t, d, p, s_gen, theta = y
    Cf, q_w, p0_in, p_in, fluid, geom_handle = args

    # Calculate velocity
    v = jnp.sqrt(v_t**2 + v_m**2)
    alpha = jnp.arctan2(v_t, v_m)

    # NEW – query geometry at local meridional coordinate s
    geom = geom_handle(meridional_length)
    r = geom["r"]
    b = geom["b"]
    A = geom["A"]
    dAds = geom["dAds"]
    phi = geom["phi"]

    # Calculate thermodynamic state
    state = fluid.get_state(jxp.DmassP_INPUTS, d, p)
    a = state["a"]
    h = state["h"]
    s = state["s"]
    T = state["T"]
    G = state["gruneisen"]
    h0 = h + 0.5 * v**2

    # Stress at the wall
    tau_w = Cf * d * v**2 / 2

    # Compute coefficient matrix
    M = jnp.asarray(
        [
            [d, 0.0, v_m, 0.0],
            [d * v_m, 0.0, 0.0, 1.0],
            [0.0, d * v_m, 0.0, 0.0],
            [0.0, 0.0, -(a**2), d],
        ]
    )

    # Compute source term
    S = jnp.asarray(
        [
            -d * v_m * (dAds / A),
            d * v_t**2 / r * jnp.sin(phi) - 2 * tau_w / b * jnp.cos(alpha),
            -d * v_t * v_m / r * jnp.sin(phi) - 2 * tau_w / b * jnp.sin(alpha),
            (2 / b) * (G / v_m) * (tau_w * v + q_w),
        ]
    )

    # Compute solution
    core = jnp.linalg.solve(M, S)  # dy = [dv_m, dv_t, d, p]
    sigma = 2.0 / b * (tau_w * v)
    s_gen_dot = sigma / (d * v_m) / T  # Entropy generation
    theta_dot = (v_t / v_m) / r  # Streamline wrapping angle (only for phi=pi/2)

    return jnp.concatenate([core, jnp.array([s_gen_dot, theta_dot])])


def postprocess_ode(t, y, args):
    """compute derived outputs at save times"""
    Cf, q_w, p0_in, p_in, fluid, geom_handle = args
    v_m, v_t, d, p, s_gen, theta = y
    v = jnp.sqrt(v_t**2 + v_m**2)
    state = fluid.get_state(jxp.DmassP_INPUTS, d, p)
    a = state["a"]

    # Evaluate geometry at current position
    geom = geom_handle(t)
    r = geom["r"]
    b = geom["b"]
    A = geom["A"]
    phi = geom["phi"]

    # Evaluate inlet geometry for normalization
    geom0 = geom_handle(0.0)
    r_in = geom0["r"]
    b_in = geom0["b"]

    # Merge everything in a single output dictionary
    out = {
        # --- flow variables ---
        "v_t": v_t,
        "v_m": v_m,
        "v": v,
        "Ma": v / a,
        "Ma_m": v_m / a,
        "Ma_t": v_t / a,
        "alpha": jnp.arctan2(v_t, v_m),
        "d": d,
        "p": p,
        "s": state["s"],
        "s_gen": s_gen,
        "h": state["h"],
        "h0": state["h"] + 0.5 * v**2,
        "theta": theta,
        "Cp": (p - p_in) / (p0_in - p_in),
        # --- geometry quantities ---
        "z": geom["z"],
        "r": geom["r"],
        "b": geom["b"],
        "A": geom["A"],
        "phi": geom["phi"],
        "dzds": geom["dzds"],
        "drds": geom["drds"],
        "dbds": geom["dbds"],
        "dAds": geom["dAds"],
        "z_shroud": geom["z_shroud"],
        "r_shroud": geom["r_shroud"],
        "z_hub": geom["z_hub"],
        "r_hub": geom["r_hub"],
        "m": t,
        "radius_ratio": r / r_in,
        "area_ratio": A / (2.0 * jnp.pi * r_in * b_in),
    }

    return out
