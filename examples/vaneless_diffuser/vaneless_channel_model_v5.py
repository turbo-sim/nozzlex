import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import jaxprop as jxp
import optimistix as optx
import nurbspy.jax as nrb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolor

jxp.set_plot_options(grid=False)


# TODO CHECK this chaptgpt conversation about how to integrate the ODE. the recommendation is to use meridional (arclength) coordinate for the geometyr functions and integrate in physicla space
# https://chatgpt.com/c/6903b75b-be7c-8325-938e-6647f18ab50e


class SolverParams(eqx.Module):
    solver_name: str = "Dopri5"
    adjoint_name: str = "DirectAdjoint"
    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 1000
    n_points: int = 100
    throw: bool = True


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
    geometry["td_in"] = jnp.maximum(1e-3, geometry["td_in"])
    geometry["td_out"] = jnp.maximum(1e-3, geometry["td_out"])
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
    channel_midline = nrb.NurbsCurve(control_points=P_midline, degree=3)

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
        curvature = channel_midline.get_curvature(u_midline)
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
            "curvature": curvature,
            "s_total": s_total,
        }

        # If s was scalar, return scalar values instead of 1-element arrays
        geom_dict = {
            k: (v.squeeze() if v.shape == (1,) else v) for k, v in geom_dict.items()
        }
        return geom_dict

    geom_handle.nurbs_midline = channel_midline
    geom_handle.nurbs_width = channel_width

    return jax.jit(geom_handle)


def plot_vaneless_channel(
    geom_handle,
    fig=None,
    ax=None,
    n_points: int = 200,
    plot_midline: bool = True,
    plot_hub: bool = True,
    plot_shroud: bool = True,
    plot_inlet: bool = True,
    plot_outlet: bool = True,
    plot_control_points: bool = False,
):
    """
    Plot the meridional channel geometry using a geometry handle from
    make_vaneless_channel_geometry().

    Parameters
    ----------
    geom_handle : callable
        Geometry evaluator function geom_handle(s) → dict of geometry properties.
    fig, ax : matplotlib Figure and Axes, optional
        Existing figure/axes to plot on. If None, new ones are created.
    n_points : int, optional
        Number of evaluation points along the channel (default 200).
    plot_midline : bool, optional
        Whether to plot the channel midline (default True).
    plot_hub : bool, optional
        Whether to plot the hub wall (default True).
    plot_shroud : bool, optional
        Whether to plot the shroud wall (default True).
    plot_inlet : bool, optional
        Whether to plot the inlet edge (default True).
    plot_outlet : bool, optional
        Whether to plot the outlet edge (default True).
    plot_control_points : bool, optional
        Whether to to plot the control points of the midline NURBS (default False).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """

    # Sample the geometry along the midline
    s = jnp.linspace(0.0, geom_handle(s=0.0)["s_total"], n_points)
    geom = geom_handle(s)

    # Extract geometry data
    z_mid = geom["z"]
    r_mid = geom["r"]
    z_shroud = geom["z_shroud"]
    r_shroud = geom["r_shroud"]
    z_hub = geom["z_hub"]
    r_hub = geom["r_hub"]

    # Compute y-axis (radial) limits
    r_max = jnp.max(r_shroud)
    r_top = 1.1 * r_max  # 10% margin above the shroud radius
    z_min = jnp.max(z_shroud)

    # Create figure and axes if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z$ $-$ Axial coordinate  [m]")
        ax.set_ylabel(r"$r$ $-$ Radial coordinate [m]")

        decimals = 2
        fmt = mticker.FormatStrFormatter(f"%.{decimals}f")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

        r_max = float(jnp.max(r_shroud))
        r_top = 1.1 * r_max
        z_min = float(jnp.min(jnp.concatenate([z_hub, z_shroud])))
        z_max = float(jnp.max(jnp.concatenate([z_hub, z_shroud])))
        dz = 0.1 * (z_max - z_min)
        z_left = z_min - dz
        z_right = z_max + dz
        ax.set_xlim(left=z_left, right=z_right)
        ax.set_ylim(bottom=0.0, top=r_top)

    # Plot requested items
    if plot_hub:
        ax.plot(z_hub, r_hub, color="black", linestyle="-")

    if plot_shroud:
        ax.plot(z_shroud, r_shroud, color="black", linestyle="-")

    if plot_midline:
        ax.plot(z_mid, r_mid, color="black", linestyle="-.")

    if plot_inlet:
        ax.plot(
            [z_hub[0], z_shroud[0]],
            [r_hub[0], r_shroud[0]],
            color="black",
            linestyle="-",
        )
    if plot_outlet:
        ax.plot(
            [z_hub[-1], z_shroud[-1]],
            [r_hub[-1], r_shroud[-1]],
            color="black",
            linestyle="-",
        )

    if plot_control_points:
        P = geom_handle.nurbs_midline.P
        zc, rc = P[0], P[1]
        ax.plot(zc, rc, "ro", linestyle="-.")

    return fig, ax


def plot_vaneless_channel_contour(
    geom_handle,
    solution: dict,
    var_name: str,
    fig=None,
    ax=None,
    levels=100,
    cmap=None,
    cbar: bool = True,
    label: str | None = None,
):
    """
    Plot a 2D contour of a 1D solution variable along the meridional channel walls.

    Parameters
    ----------
    geom_handle : callable
        Geometry evaluator function geom_handle(s) → dict of geometry properties.
    solution : dict
        Output dictionary from the diffuser solver or postprocessing function.
        Must contain 'm' (meridional coordinate) and the requested variable.
    var_name : str
        Name of the variable in the solution dictionary to visualize
        (e.g. "p", "Ma", "Cp", "v_m").
    fig, ax : matplotlib Figure and Axes, optional
        Existing figure/axes to plot on. If None, new ones are created.
    cmap : matplotlib Colormap, optional
        Colormap object to use for the contour fill (default: plt.cm.magma).
    label : str, optional
        Label for the colorbar.
    cbar : bool, optional
        Whether to include a colorbar (default True).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Figure and axes with the plotted contours.
    """

    # Extract data from the solution dictionary
    if "m" not in solution:
        raise KeyError(
            "Solution dictionary must contain key 'm' (meridional coordinate)."
        )
    if var_name not in solution:
        raise KeyError(f"Variable '{var_name}' not found in solution dictionary.")

    if cmap is None:
        cmap = mcolor.LinearSegmentedColormap.from_list(
            "soft_Blues", plt.cm.Blues(jnp.linspace(0.3, 0.9, 256))
        )

    s_vec = solution["m"]
    var_vec = solution[var_name]

    # Evaluate geometry
    geom = geom_handle(s_vec)
    z_hub, r_hub = geom["z_hub"], geom["r_hub"]
    z_shroud, r_shroud = geom["z_shroud"], geom["r_shroud"]

    # Build 2D interpolation grid between hub and shroud
    spanwise = jnp.linspace(0.0, 1.0, 50)  # increase for smoother interpolation
    z_grid = jnp.outer(z_hub, 1 - spanwise) + jnp.outer(z_shroud, spanwise)
    r_grid = jnp.outer(r_hub, 1 - spanwise) + jnp.outer(r_shroud, spanwise)
    var_grid = jnp.outer(var_vec, jnp.ones_like(spanwise))

    # Create figure and axes if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$z$ $-$ Axial coordinate [m]")
        ax.set_ylabel(r"$r$ $-$ Radial coordinate [m]")

    # Plot the filled contour
    cf = ax.contourf(
        z_grid,
        r_grid,
        var_grid,
        levels=levels,
        cmap=cmap,
    )

    # Optional colorbar
    if cbar:
        cb = fig.colorbar(cf, ax=ax)
        cb.set_label(label if label else var_name)

    return fig, ax


# -----------------------------------------------------------------------------
# Main API to the vaneless diffuser model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def solve_vaneless_channel_model(params, fluid, geom_handle, solver_params):
    r"""
    Integrate the one-dimensional steady flow equations through a vaneless
    diffuser or return channel using the enthalpy-pressure formulation.

    The model solves the coupled differential system governing the mass, meridional
    momentum, tangential momentum, and total energy balances.

    The solver performs numerical integration from the inlet to the diffuser exit
    using an adaptive high-order Diffrax ODE integrator. At each integration point,
    derived flow quantities (Mach number, density, entropy, geometric parameters, etc.)
    are also evaluated and stored.

    Parameters
    ----------
    params : dict
        Flow and boundary condition parameters, including:
        - ``p_in`` : inlet static pressure [Pa]
        - ``h_in`` : inlet static enthalpy [J/kg]
        - ``v_in`` : inlet velocity magnitude [m/s]
        - ``alpha_in`` : inlet flow angle (radians)
        - ``C_f`` : wall friction coefficient [-]
        - ``q_w`` : wall heat flux [W/m²]
    fluid : jaxprop.Fluid
        Thermodynamic fluid model exposing the method
        ``get_state(INPUT_TYPE, var1, var2)`` for evaluating
        fluid properties such as density, enthalpy, entropy, and speed of sound.
    geom_handle : callable
        Geometry handle returning a dictionary of local geometric quantities
        (radius, width, area, derivatives, inclination) as a function of the
        meridional coordinate. Obtained from function `make_vaneless_channel_geometry()`
    solver_params : object
        Configuration for the Diffrax integrator, including the solver name,
        adjoint type, tolerances (``rtol``, ``atol``), number of save points,
        and maximum iteration count.

    Returns
    -------
    solution : dict of ndarrays
        Integrated solution array containing all computed variables
    """

    # Compute initial conditions for ODE system
    p_in = params["p_in"]
    h_in = params["h_in"]
    v_in = params["v_in"]
    alpha_in = params["alpha_in"]
    v_m_in = v_in * jnp.cos(alpha_in)
    v_t_in = v_in * jnp.sin(alpha_in)


    # Compute inlet stagnation pressure for Cp calculations
    state_in = fluid.get_state(jxp.HmassP_INPUTS, h_in, p_in)
    h0_in = h_in + 0.5 * v_in**2
    s0_in = state_in.s
    p0_in = fluid.get_state(jxp.HmassSmass_INPUTS, h0_in, s0_in).p
    params["p0_in"] = p0_in

    y0 = jnp.array([v_m_in, v_t_in, h_in, p_in, p0_in, 0.0, 0.0])

    # Integration termination
    s_max = geom_handle(s=0.0)["s_total"]

    # Group the ODE system constant parameters
    args = (params, fluid, geom_handle, s_max)

    # Create and configure the solver
    func_rhs = lambda t, y, args: evaluate_vaneless_channel_ode(t, y, args)[0]
    func_all = lambda t, y, args: evaluate_vaneless_channel_ode(t, y, args)[1]
    solver = jxp.make_diffrax_solver(solver_params.solver_name)
    adjoint = jxp.make_diffrax_adjoint(solver_params.adjoint_name)
    term = dfx.ODETerm(func_rhs)
    ctrl = dfx.PIDController(rtol=solver_params.rtol, atol=solver_params.atol)
    ts = jnp.linspace(0.0, s_max, solver_params.n_points)
    saveat = dfx.SaveAt(ts=ts, dense=False, fn=func_all)

    # Solve the ODE system
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=s_max,
        dt0=None,
        y0=y0,
        saveat=saveat,
        stepsize_controller=ctrl,
        args=args,
        max_steps=solver_params.max_steps,
        adjoint=adjoint,
        throw=solver_params.throw
    )

    return solution.ys


# -----------------------------------------------------------------------------
# Right hand side of the diffuser ODE system
# -----------------------------------------------------------------------------
def evaluate_vaneless_channel_ode(t, y, args):
    r"""
    Evaluate the right-hand side of the one-dimensional vaneless diffuser model
    using the enthalpy-pressure formulation.

    This function computes the differential evolution of the meridional,
    tangential, thermodynamic, and geometric flow variables along the channel
    arclength coordinate :math:`m` (or :math:`s`). It represents the local
    steady-flow momentum and energy balances in axisymmetric coordinates,
    following the model described in:

        R. Agromayor, B. Müller, and L. O. Nord, “One-dimensional annular diffuser model
        for preliminary turbomachinery design,” International Journal of Turbomachinery,
        Propulsion and Power, vol. 4, no. 3, p. 31, 2019, doi: 10.3390/ijtpp4030031.

    The governing equations are identical to those in the cited paper but
    reformulated in terms of the thermodynamic variables :math:`(h, p)` instead
    of :math:`(\rho, p)`.

    The system can be written compactly as:

    .. math::

        A\,\frac{dU}{dm} = S

    where

    .. math::
        A =
        \begin{bmatrix}
        \tfrac{\rho}{v_m} & 0 & -\,\tfrac{\rho G}{a^{2}} & \tfrac{1+G}{a^{2}} \\
        \rho v_m & 0 & 0 & 1 \\
        0 & \rho v_m & 0 & 0 \\
        v_m & v_\theta & 1 & 0
        \end{bmatrix},
        \quad
        U =
        \begin{bmatrix}
        v_m \\ v_\theta \\ h \\ p
        \end{bmatrix},
        \quad
        S =
        \begin{bmatrix}
        -\,\tfrac{\rho}{A}\tfrac{dA}{dm} \\[0.6em]
        \tfrac{\rho v_\theta^2}{r}\sin\phi - \tfrac{2\tau_w}{b}\cos\alpha \\[0.6em]
        -\,\tfrac{\rho v_\theta v_m}{r}\sin\phi - \tfrac{2\tau_w}{b}\sin\alpha \\[0.6em]
        \tfrac{2\dot q_w}{b}
        \end{bmatrix}.

    Here,
    - :math:`v_m, v_\theta` are the meridional and tangential velocities,
    - :math:`h, p` are the static enthalpy and pressure,
    - :math:`A = 2\pi r b` is the meridional flow area,
    - :math:`\tau_w` is the wall shear stress,
    - :math:`\dot q_w` is the wall heat flux,
    - :math:`a` is the local speed of sound,
    - :math:`G` is the Grüneisen parameter

    The dependent variables evolve according to:

    .. math::
        \frac{dU}{dm}
        =
        A^{-1} S,

    which is solved numerically at each spatial location.

    The last two terms of the returned state vector represent:
    - :math:`\dot{s}_{gen}` — entropy generation rate per unit mass flow,
    - :math:`\dot{\theta}` — circumferential wrapping rate of the streamline.

    Parameters
    ----------
    t : float
        Meridional coordinate :math:`m` or :math:`s` (integration variable).
    y : array_like
        State vector `[v_m, v_t, h, p, s_gen, theta]`.
    args : tuple
        Model parameters `(C_f, q_w, p0_in, p_in, fluid, geom_handle)`.

    Returns
    -------
    rhs : ndarray
        Time derivative of the state vector
        `[dv_m/dm, dv_t/dm, dh/dm, dp/dm, ds_gen/dm, dtheta/dm]`.
    out : dict
        Dictionary containing derived thermodynamic, kinematic, and geometric
        quantities at the current meridional location, including Mach numbers,
        entropy, total enthalpy, static-to-total pressure ratio, and local
        geometry descriptors such as area, radius, and inclination angle.
    """

    # Rename from ODE terminology to physical variables
    params, fluid, geom_handle, L_total = args
    m_coord = jnp.minimum(t, L_total - 1e-6) # Prevent NURBS extrapolation
    v_m, v_t, h, p, p0, s_gen, theta = y

    # jax.debug.print(
    #     "t={:.3e}, v_m={:.3e}, v_t={:.3e}, h={:.3e}, p={:.3e}, s_gen={:.3e}, theta={:.3e}",
    #     t, v_m, v_t, h, p, s_gen, theta
    # )

    # Calculate velocity magnitude and direction
    v = jnp.sqrt(v_t**2 + v_m**2)
    alpha = jnp.arctan2(v_t, v_m)

    # Query geometry at local meridional coordinate
    geom = geom_handle(m_coord)
    r = geom["r"]
    b = geom["b"]
    A = geom["A"]
    dAds = geom["dAds"]
    phi = geom["phi"]
    curvature = geom["curvature"]

    # Calculate thermodynamic state
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    a = state["a"]
    d = state["d"]
    T = state["T"]
    s = state["s"]
    mu = state["viscosity"]
    G = state["gruneisen"]

    # Calculate otal pressure at current location
    h0 = h + 0.5 * v **2
    p0_bis = fluid.get_state(jxp.HmassSmass_INPUTS, h0, s).p

    # Compute skin friction according to Aungier loss model
    D_h = 2.0 * b
    Re = jnp.maximum(d * v * D_h / mu, 1.0)
    roughness = params["roughness"]
    alpha_in = params["alpha_in"]
    b_in = params["geometry"]["b_in"]
    cf_W = get_cf_wall(Re, roughness, D_h)
    cf_D, E = get_cf_diffusion(b, A, dAds, alpha_in, b_in, L_total)
    cf_C = get_cf_curvature(b, curvature, alpha)
    cf_total = cf_W + cf_D + cf_C
    # cf_total = jnp.asarray(0.0)
    tau_w = 0.5 * cf_total * d * v**2

    # Heat transfer at the wall (to be improved in the future)
    q_w = params["q_w"]

    # Compute coefficient matrix
    M = jnp.asarray(
        [
            [d / v_m, 0.0, -G / a**2, (1.0 + G) / a**2],
            [d * v_m, 0.0, 0.0, 1.0],
            [0.0, d * v_m, 0.0, 0.0],
            [v_m, v_t, 1.0, 0.0],
        ]
    )

    # Compute source term
    S = jnp.asarray(
        [
            -d * (dAds / A),
            d * v_t**2 / r * jnp.sin(phi) - 2 * tau_w / b * jnp.cos(alpha),
            -d * v_t * v_m / r * jnp.sin(phi) - 2 * tau_w / b * jnp.sin(alpha),
            (2 / b) * q_w,
        ]
    )

    # Compute solution
    dv_m, dv_t, dh, dp = jnp.linalg.solve(M, S)
    # dp0 = dp + d * v_m * dv_m + d * v_t * dv_t


    dp0 = - (2.0 * tau_w) / (b * jnp.cos(alpha))
    s_gen = 2.0 / b * (tau_w * v)
    ds = s_gen / (d * v_m) / T    
    dtheta = (v_t / v_m) / r  # Streamline wrapping angle
    rhs = jnp.array([dv_m, dv_t, dh, dp, dp0, ds, dtheta])

    # Compute derived output quantities
    geom0 = geom_handle(0.0)
    A_in = geom0["A"]
    r_in = geom0["r"]
    p_in = params["p_in"]
    p0_in = params["p0_in"]

    # Compute pressure recovery and breakdown
    Cp = (p - p_in) / (p0_in - p_in)
    dCp_kinetic = (p0_bis - p) / (p0_in - p_in)
    dCp_loss = (p0_in - p0_bis) / (p0_in - p_in)
    dCp_loss_wall = dCp_loss * cf_W / cf_total
    dCp_loss_diff = dCp_loss * cf_D / cf_total
    dCp_loss_curv = dCp_loss * cf_C / cf_total

    # Store esults
    out = {
        # --- flow variables ---
        "v_t": v_t,
        "v_m": v_m,
        "v": v,
        "Ma": v / a,
        "Ma_m": v_m / a,
        "Ma_t": v_t / a,
        "alpha": alpha,
        "alpha_deg": jnp.rad2deg(alpha),
        "d": d,
        "p": p,
        "s": state["s"],
        "p0": p0,
        "p0_bis": p0_bis,
        "s_gen": s_gen,
        "h": state["h"],
        "h0": state["h"] + 0.5 * v**2,
        "theta": theta,
        "Cp": Cp,
        "dCp_kinetic": dCp_kinetic,
        "dCp_loss": dCp_loss,
        "dCp_loss_wall": dCp_loss_wall,
        "dCp_loss_diffusion": dCp_loss_diff,
        "dCp_loss_curvature": dCp_loss_curv,
        # --- geometry quantities ---
        "m": t,
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
        "radius_ratio": r / r_in,
        "area_ratio": A / A_in,
        "Re": Re,
        "cf_total": cf_total,
        "cf_wall": cf_W,
        "cf_diffusion": cf_D,
        "cf_curvature": cf_C,
        "E_diffusion": E,
    }

    return rhs, out


# def get_cf_wall(Re, roughness, diameter):
#     """Fanning friction factor from Haaland correlation."""
#     term = 6.9 / jnp.maximum(Re, 1.0) + (roughness / diameter / 3.7) ** 1.11
#     f_D = (-1.8 * jnp.log10(term)) ** -2  # Darcy friction factor
#     return f_D / 4.0  # Convert to Fanning friction factor

def get_cf_wall(Re, roughness, diameter):
    """Fanning friction factor with smooth laminar-turbulent transition."""
    # Laminar friction factor (Cf = 16/Re for Fanning)
    Cf_laminar = 16.0 / jnp.maximum(Re, 1.0)
    
    # Turbulent friction factor (Haaland correlation)
    term = 6.9 / jnp.maximum(Re, 1.0) + (roughness / diameter / 3.7) ** 1.11
    f_D = (-1.8 * jnp.log10(term)) ** -2  # Darcy friction factor
    Cf_turbulent = f_D / 4.0  # Convert to Fanning friction factor
    
    # Smooth blending using tanh (transition centered at Re=2300)
    transition_width = 500.0  # Controls smoothness (smaller = sharper transition)
    Re_transition = 2300.0 + transition_width  # Typical transition Reynolds number
    blend = 0.5 * (1.0 + jnp.tanh((Re - Re_transition) / transition_width))
    
    # Blend between laminar and turbulent
    return Cf_laminar * (1.0 - blend) + Cf_turbulent * blend


def get_cf_diffusion(b, A, dA_dm, alpha_in, b_in, L_total):
    """Diffusion loss coefficient cf,D following Aungier (1993)."""
    D = (b / A) * dA_dm
    D_m = 0.4 * jnp.cos(alpha_in) * (b_in / L_total) ** 0.35
    def E_piecewise(D, D_m):
        return jnp.where(
            D <= 0,
            1.0,
            jnp.where(
                D < D_m,
                1.0 - 0.2 * (D / D_m) ** 2,
                0.8 * jnp.sqrt(D_m / D),
            ),
        )

    E = E_piecewise(D, D_m)
    return D * (1.0 - E), E


def get_cf_curvature(b, curvature, alpha):
    """Curvature loss coefficient cf,C following Aungier (1993)."""
    return (b * curvature * jnp.cos(alpha)) / 26.0


def get_friction_factor_haaland(Reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    Re_safe = jnp.maximum(Reynolds, 1.0)
    term = 6.9 / Re_safe + (roughness / diameter / 3.7) ** 1.11
    f = (-1.8 * jnp.log10(term)) ** -2
    return f


def compute_static_state(p0, T0, Ma, fluid):
    r"""
    Compute the static thermodynamic state corresponding to a given Mach number
    assuming an isentropic process from a known stagnation (total) state.

    The function determines the static pressure :math:`p`, static enthalpy
    :math:`h`, and flow velocity :math:`v` such that both the total energy and
    entropy are conserved between the stagnation and static states.

    The governing relations are:

    .. math::

        \begin{aligned}
        h_0 &= h + \tfrac{1}{2}v^2, \\
        s_0 &= s,
        \end{aligned}

    where :math:`h_0, s_0` are the stagnation (total) enthalpy and entropy,
    and :math:`v = a\,M_a` is the local flow velocity expressed in terms of the
    Mach number :math:`M_a` and the local speed of sound :math:`a`.

    By substituting :math:`v = a\,M_a`, the system becomes two nonlinear equations
    in :math:`(p, T)`:

    .. math::

        \begin{cases}
        f_1(p, T) = h_0 - \left[h(p, T) + \tfrac{1}{2}(a(p, T)\,M_a)^2\right] = 0, \\[0.5em]
        f_2(p, T) = s(p, T) - s_0 = 0,
        \end{cases}

    which are solved simultaneously using a Newton root-finding method.

    Parameters
    ----------
    p0 : float
        Stagnation (total) pressure [Pa].
    T0 : float
        Stagnation (total) temperature [K].
    Ma : float
        Local Mach number.
    fluid : object
        Thermodynamic fluid model providing the method
        ``get_state(INPUT_TYPE, p, T)`` returning properties such as
        speed of sound ``a``, enthalpy ``h``, and entropy ``s``.

    Returns
    -------
    p : float
        Static pressure [Pa].
    h : float
        Static enthalpy [J/kg].
    v : float
        Flow velocity [m/s].
    """
    # --- Reference stagnation state ---
    st0 = fluid.get_state(jxp.PT_INPUTS, p0, T0)
    h0, s0 = st0["h"], st0["s"]

    # --- Define residual system: f(p, h) = [f1, f2] ---
    def residual(x, _):
        p, T = x
        st = fluid.get_state(jxp.PT_INPUTS, p, T)
        a, s, h = st["a"], st["s"], st["h"]
        f1 = h0 - (h + 0.5 * (a * Ma) ** 2)  # energy balance
        f2 = s - s0  # isentropic condition
        return jnp.array([f1, f2])

    # --- Initial guess ---
    initial_guess = jnp.array([p0, T0])

    # --- Solve ---
    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    sol = optx.root_find(residual, solver, y0=initial_guess)
    p, T = sol.value

    # --- Evaluate final state ---
    st = fluid.get_state(jxp.PT_INPUTS, p, T)
    a = st["a"]
    h = st["h"]
    v = a * Ma

    return p, h, v


def get_analytical_Cp(alpha_in, geom_handle):
    r"""
    Compare the numerical diffuser solution against the analytical expression
    for the incompressible, inviscid pressure recovery coefficient.

    The analytical model assumes steady, axisymmetric, incompressible flow
    with negligible wall friction and no energy losses. Under these hypotheses,
    Bernoulli's equation applies along a streamline, and the pressure recovery
    coefficient between inlet (1) and outlet (2) can be written as

    $$
    C_p = 1
    - \cos^2(\alpha_1) \left( \frac{A_1}{A_2} \right)^2
    - \sin^2(\alpha_1) \left( \frac{r_1}{r_2} \right)^2
    $$

    where

    - $\alpha_1$ is the inlet flow angle (measured from the meridional direction),
    - $r_1$ and $r_2$ are the inlet and outlet radii,
    - $A_1$ and $A_2$ are the corresponding meridional flow areas.

    This expression represents the ideal pressure recovery in an annular vaneless diffuser.
    """
    geom_1 = geom_handle(0.0)
    geom_2 = geom_handle(geom_1["s_total"])
    AR = geom_1["A"] / geom_2["A"]
    RR = geom_1["r"] / geom_2["r"]
    Cp = 1.0 - (jnp.cos(alpha_in) ** 2) * AR**2 - (jnp.sin(alpha_in) ** 2) * RR**2
    return Cp


def evaluate_solution_dense(sol, n_points: int = 100):
    """
    Evaluate a Diffrax solution object at evenly spaced points between t0 and t1,
    using vmapped interpolation.

    Parameters
    ----------
    sol : diffrax.Solution
        Solution object returned by `diffrax.diffeqsolve`.
    n_points : int, optional
        Number of evaluation points (default: 200).

    Returns
    -------
    t_eval : jnp.ndarray
        1D array of evaluation times (shape: [n_points]).
    y_eval : jnp.ndarray
        2D array of interpolated state values (shape: [n_points, n_state]).
    """
    t_eval = jnp.linspace(sol.t0, sol.t1, n_points)
    y_eval = jax.vmap(sol.evaluate)(t_eval)
    return t_eval, y_eval
