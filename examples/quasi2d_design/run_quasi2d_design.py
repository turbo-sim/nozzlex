"""Exemplary quasi-2D inverse design: impose a centerline and a wall
pressure profile that share the SAME inlet and exit pressure (as is
physically typical -- pressure is uniform across the span at the inlet
plane and, for a well-designed nozzle, again at the exit), differing only
in the interior, and recover a curvature-consistent passage shape.

Method (see docs/inverse_design_2d.md in the nozzlex repo for the theory):

  1. Two streamlines only -- centerline and wall -- sharing a common,
     uniform inlet state, AND a common exit pressure `p_out`. Only the
     interior "bulge" of each profile differs (`center_bulge`/`wall_bulge`,
     a perturbation of the NURBS mid control point) -- so center and wall
     pressure match exactly at both ends *by construction*, not by
     correcting for it after the fact. (An earlier version of this script
     imposed independent p_center_out/p_wall_out targets and corrected the
     mismatch iteratively; that fights the physics right at the boundary
     and is numerically fragile -- this formulation avoids the problem
     instead of patching around it.)

  2. Each streamline is marched independently with the existing 1D
     closed-form isentropic solver from `nozzlex.inverse_design.functions`
     (frictionless/adiabatic, real-gas EOS via jaxprop) -- reused as-is,
     unmodified, as the per-streamline building block.

  3. The two streamlines' local mass flux (rho*u) is combined with a
     2-point (trapezoidal) quadrature across the span to recover the
     passage half-height h(x) from global mass conservation.

  4. ITERATIVE CORRECTION: even with matching endpoints, the interior
     pressure difference implied by the initial NURBS guess is generally
     NOT curvature-consistent -- the resulting shape's own curvature
     implies a different transverse pressure difference than the one that
     was imposed. So p_center(x) is kept fixed as the true design target,
     and p_wall(x) is corrected every iteration to match what the current
     shape's curvature actually implies via radial equilibrium
     (dp/dn = rho*V^2/R_c), until the two are self-consistent. This is the
     same DOF-counting argument as before, just now starting from a
     well-posed (matching-endpoint) initial guess instead of fighting an
     arbitrary one.


  5. A handful of interior streamlines are reconstructed for visualization
     by assuming a linear mass-flux profile between the (now converged)
     centerline and wall states, and finding the transverse position that
     carries each equal fraction of the total mass flow.

Everything here is illustrative/approximate (a nominal axial length scale,
a 2-point spanwise quadrature, a linear flux profile for the interior
streamlines) -- it is meant to make the theoretical discussion concrete,
not to be a production design tool.

"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jaxprop as jxp
import yaml
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq

from nozzlex.inverse_design.nurbs_creator import nurbs_curve_points
from nozzlex.inverse_design.functions import march_from_p_isentropic

jxp.set_plot_options(grid=False)
plt.rcParams.update({'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 12})

config_path = Path(__file__).with_name('settings.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

fluid_name = config['fluid_name']
backend = config.get('backend', 'HEOS')
fluid = jxp.Fluid(fluid_name, backend=backend)
inlet_opening = config['inlet_opening']   # full inlet height
width = config['width']
u_in = config['u_in']
L_nominal = config.get('L_nominal_factor', 20.0) * inlet_opening
compare_1d_default = config.get('compare_1d', False)
compare_moc_default = config.get('compare_moc', False)
correct_wall_pressure_default = config.get('correct_wall_pressure', False)

N_STREAMLINES = 10      # interior streamlines drawn for visualization (excl. center/wall)
MAX_ITER = 80
RELAX = 0.25
TOL = 1e-8              # relative pressure residual

results_dir = Path(__file__).parent / 'out'
results_dir.mkdir(parents=True, exist_ok=True)

case_labels = [c.get('label', f"Case {i+1}") for i, c in enumerate(config['cases'])]
num_cases = len(case_labels)
color_map = cm.get_cmap('plasma', num_cases)

x_nodes = np.linspace(0.0, 1.0, 100)


# =============================================================================
# PRESSURE PROFILE GENERATION -- edit x_sonic/conv_handle_frac/center_bend/
# wall_bend in settings.yaml to change the shape. dp/dx is kept smooth
# (continuous) everywhere, including at the convergent/divergent join --
# see the note in build_center_wall_profiles for how.
#
# 1. Convergent section (0 -> x_sonic): control points
#    [(0, p_in), (conv_handle_frac*x_sonic, p_in), (x_sonic, p_sonic)].
#    p_sonic is the PHYSICAL sonic pressure -- a pure thermodynamic
#    property of (fluid, s_in, h0) -- solved directly by a 1D root-find
#    (solve_sonic_pressure), no marching/iteration needed: the sonic
#    pressure doesn't depend on how p varies with x at all. x_sonic --
#    WHERE along the nozzle that pressure occurs -- is a free design
#    choice. The bifurcation point and the physical sonic condition
#    coincide by construction: whatever x_sonic you pick, the flow really
#    is sonic there, because we solved for the pressure that makes it so.
#    Used IDENTICALLY for both centerline and wall (no separate input --
#    they're the same curve because there's nothing to diverge yet).
#
# 2. Divergent section (x_sonic -> 1): TWO independent 3-control-point
#    NURBS curves (quadratic), one per streamline, built to match the
#    convergent curve's slope at the join (see build_center_wall_profiles)
#    -- center_bend/wall_bend control how far each branch travels along
#    that shared tangent before bending toward p_out. Equal values keep
#    both branches identical; different values make the wall diverge,
#    still smoothly.
#
# Every control point is given as (x, p_frac), with x in x/L and p_frac
# in [0,1] mapping to an actual pressure via p = p_in - p_frac*(p_in-p_out)
# (0 -> p_in, 1 -> p_out).
# =============================================================================

def build_profile_from_ctrl(ctrl_pts_frac, p_in, p_out, x_nodes, mask=None):
    """Smooth NURBS curve (any number of control points) from a list of
    (x, p_frac) control points, evaluated over x_nodes (or only where
    `mask` is True, NaN elsewhere)."""
    ctrl_pts = [(x, p_in - pf * (p_in - p_out)) for x, pf in ctrl_pts_frac]
    # A valid B-spline needs n_ctrl_pts >= degree+1 -- degree=4 with only 3
    # control points (the divergent branches) is degenerate and produces a
    # near-step-function curve. Cap degree to what the point count supports
    # (3 CPs -> degree 2, a quadratic, which is exactly the "single sign of
    # curvature" curve requested for the divergent branches).
    degree = min(4, len(ctrl_pts) - 1)
    nurbs_pts = nurbs_curve_points(ctrl_pts, [1.0] * len(ctrl_pts), degree=degree, num=400)
    if mask is None:
        return np.interp(x_nodes, nurbs_pts[:, 0], nurbs_pts[:, 1])
    prof = np.full_like(x_nodes, np.nan)
    prof[mask] = np.interp(x_nodes[mask], nurbs_pts[:, 0], nurbs_pts[:, 1])
    return prof


def find_sonic_x(iso, x_nodes):
    """Locate the PHYSICAL sonic point (M=1) of a marched profile, by
    finding where M=u/a crosses 1 and linearly interpolating to the
    crossing. This is a property of the actual flow solution, not of the
    pressure curve's geometry (its inflection point, p''=0, is a
    different thing and generally does NOT coincide with M=1)."""
    v_len = int(np.sum(~np.isnan(iso['u'])))
    M = iso['u'][:v_len] / iso['a'][:v_len]
    sign_change = np.where(np.diff(np.sign(M - 1.0)) != 0)[0]
    if len(sign_change) == 0:
        return 0.5 * (x_nodes[0] + x_nodes[-1])
    i = sign_change[0]
    t = (1.0 - M[i]) / (M[i + 1] - M[i])
    return x_nodes[i] + t * (x_nodes[i + 1] - x_nodes[i])


def solve_sonic_pressure(fluid, s_in, h0, p_low, p_high):
    """The sonic pressure p* (where u(p)=a(p) along the isentrope s_in) is
    a pure thermodynamic property of (fluid, s_in, h0) -- it does NOT
    depend on how p varies with x. So it's a plain 1D root-find on
    pressure alone, no marching/x-dependence/iteration-on-a-curve needed."""
    def residual(p):
        st = fluid.get_state(jxp.PSmass_INPUTS, p, s_in)
        u = np.sqrt(max(2.0 * (h0 - st.h), 0.0))
        return u - st.a
    return brentq(residual, p_low, p_high)


def solve_convergent_ctrl(fluid, p_in, p_out, s_in, h0, x_sonic, conv_handle_frac=0.6):
    """Convergent-section control points [(0, p_in), (x_handle, p_in),
    (x_sonic, p_sonic)]. p_sonic is the physical sonic pressure (solved
    once, directly -- see solve_sonic_pressure); x_sonic -- WHERE along
    the nozzle that pressure occurs -- is a free design choice.

    x_handle = conv_handle_frac * x_sonic, strictly LESS than x_sonic (not
    equal to it). Putting the handle at the same x as the endpoint (as an
    earlier version of this did) forces an almost-vertical incoming slope
    right at the throat -- physically unrealistic, and it also guarantees
    a slope mismatch with whatever the divergent branch starts with,
    which shows up as a visible kink in the recovered shape right at the
    throat. Separating the handle from x_sonic gives a moderate,
    controllable approach slope instead."""
    p_star = solve_sonic_pressure(fluid, s_in, h0, p_out, p_in)
    p_sonic_frac = (p_in - p_star) / (p_in - p_out)
    x_handle = conv_handle_frac * x_sonic
    return [(0.0, 0.0), (x_handle, 0.0), (x_sonic, p_sonic_frac)], x_sonic, p_sonic_frac


def build_center_wall_profiles(fluid, p_in, p_out, s_in, h0, x_sonic, conv_handle_frac,
                               center_x2, center_ctrl3_x, center_kink_frac,
                               wall_x2, wall_ctrl3, x_nodes):
    """Centerline and wall pressure profiles: identical through the
    convergent section (its endpoint IS the physical sonic pressure,
    solved directly -- see solve_convergent_ctrl).

    Divergent centerline: a smooth 4-CP cubic NURBS from the throat to a
    kink point, then FLAT (p = p_out) from the kink to the exit -- smooth
    (C1) at BOTH ends, not just the throat:
      CP1 (join)   -- fixed: (x_sonic, p_sonic).
      CP2          -- x is free (`center_x2`, must be > x_sonic); p_frac
                      is solved so CP2 lies on the convergent curve's own
                      incoming tangent line, guaranteeing dp/dx is
                      CONTINUOUS at the throat (same trick as the wall
                      branch below).
      CP3          -- x is free (`center_ctrl3_x`, between center_x2 and
                      the kink); p_frac fixed = 1.0 (same as CP4).
      CP4 (kink)   -- fixed p_frac=1.0; x is `center_kink_frac` in [0,1]
                      as a fraction of the divergent span (0 = kink at the
                      throat, i.e. fully flat; 1 = kink at the exit, i.e.
                      no flat part). Because CP3 and CP4 share the same
                      p_frac, the curve's incoming tangent AT the kink is
                      horizontal -- matching the flat tail's own (zero)
                      slope, so the whole centerline trace is smooth (C1)
                      all the way from the throat to the last node before
                      the exit, not just kinked into the flat part.

    Divergent wall: an independent 4-CP cubic NURBS branch --
      CP1 (join)   -- fixed: (x_sonic, p_sonic).
      CP2          -- x is free (`wall_x2`, must be > x_sonic); p_frac is
                      solved so CP2 lies on the convergent curve's own
                      incoming tangent line (extended past the join),
                      guaranteeing dp/dx is CONTINUOUS at the join.
      CP3          -- fully free, both x and p_frac (`wall_ctrl3` as
                      [x, p_frac]); p_frac=1.0 gives dp/dx=0 at the exit.
      CP4 (exit)   -- fixed: (1.0, 1.0)."""
    ref_ctrl, x_sonic, p_sonic_frac = solve_convergent_ctrl(
        fluid, p_in, p_out, s_in, h0, x_sonic, conv_handle_frac)

    conv_mask = x_nodes <= x_sonic
    div_mask = ~conv_mask
    p_ref = build_profile_from_ctrl(ref_ctrl, p_in, p_out, x_nodes, conv_mask)

    join = (x_sonic, p_sonic_frac)
    conv_handle_pt = ref_ctrl[1]
    tangent = (join[0] - conv_handle_pt[0], join[1] - conv_handle_pt[1])

    def handle_at_x(x2):
        # x2 = join[0] + t*tangent[0]  =>  t = (x2 - join[0]) / tangent[0]
        t = (x2 - join[0]) / tangent[0]
        return (x2, join[1] + t * tangent[1])

    center_handle = handle_at_x(center_x2)
    wall_handle = handle_at_x(wall_x2)
    wall_ctrl3_pt = tuple(wall_ctrl3)

    x_kink = x_sonic + center_kink_frac * (1.0 - x_sonic)
    center_ctrl = [join, center_handle, (center_ctrl3_x, 1.0), (x_kink, 1.0)]
    curve_mask = div_mask & (x_nodes <= x_kink)
    center_curve = build_profile_from_ctrl(center_ctrl, p_in, p_out, x_nodes, curve_mask)
    center_div = np.where(curve_mask, center_curve, p_out)

    wall_div = build_profile_from_ctrl(
        [join, wall_handle, wall_ctrl3_pt, (1.0, 1.0)], p_in, p_out, x_nodes, div_mask)

    p_center = np.where(conv_mask, p_ref, center_div)
    p_wall = np.where(conv_mask, p_ref, wall_div)
    return (p_center, p_wall, x_sonic, join, ref_ctrl,
            center_ctrl,
            [join, wall_handle, wall_ctrl3_pt, (1.0, 1.0)])


def march(p_profile, h_in, s_in, A_in, u_in):
    iso = march_from_p_isentropic(x_nodes, p_profile, h_in, s_in, A_in, u_in,
                                  fluid=fluid_name, backend=backend)
    return iso


def curvature_dp(x_phys, h_half, rho_avg, V_avg):
    """Signed transverse pressure difference implied by the shape's own
    curvature via radial equilibrium (rough, illustrative estimate)."""
    # A generous smoothing factor matters here: this is a GLOBAL fit, so a
    # local kink anywhere in h_half (e.g. from a taper blend zone near the
    # exit/throat) can otherwise distort the curvature estimate everywhere,
    # including far away near the inlet -- not just locally.
    spline = UnivariateSpline(x_phys, h_half, k=4, s=len(x_phys) * np.var(h_half) * 5e-2)
    h_p = spline.derivative(1)(x_phys)
    h_pp = spline.derivative(2)(x_phys)
    h_pp_safe = np.where(np.abs(h_pp) < 1e-9, np.sign(h_pp + 1e-30) * 1e-9, h_pp)
    R_c = (1.0 + h_p ** 2) ** 1.5 / h_pp_safe
    return rho_avg * V_avg ** 2 / R_c * h_half, spline(x_phys)


fig, axes = plt.subplots(2, 2, figsize=(9, 7))
ax_p, ax_shape, ax_massflux, ax_mach = axes.flat
for ax in axes.flat:
    ax.set_box_aspect(0.75)

fig2, axes2 = plt.subplots(num_cases, 2, figsize=(10, 4 * num_cases), squeeze=False)

legend_handles = []

for i, c in enumerate(config['cases']):
    label = case_labels[i]
    color = color_map(i)
    print(f"Processing {label}...")

    p0_total, Q0 = c['p0_total'], c['Q0']
    p_out = c['p_out']
    x_sonic_design = c['x_sonic']   # where along x/L the sonic point sits -- your choice
    conv_handle_frac = c.get('conv_handle_frac', 0.6)
    center_x2 = c.get('center_x2', x_sonic_design + 0.1)
    center_ctrl3_x = c.get('center_ctrl3_x', x_sonic_design + 0.2)
    center_kink_frac = c.get('center_kink_frac', 0.3)
    wall_x2 = c.get('wall_x2', x_sonic_design + 0.1)
    wall_ctrl3 = c.get('wall_ctrl3', [0.7, 1.0])
    compare_1d = c.get('compare_1d', compare_1d_default)
    compare_moc = c.get('compare_moc', compare_moc_default)
    correct_wall_pressure = c.get('correct_wall_pressure', correct_wall_pressure_default)

    # --- Common, uniform inlet state (shared by both streamlines) ---
    A_in = width * inlet_opening
    state_0 = fluid.get_state(jxp.PQ_INPUTS, p0_total, Q0)
    h_in = state_0.h - 0.5 * u_in ** 2
    state_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, state_0.s)
    mdot = A_in * u_in * state_in.rho
    p_in = state_in.p
    s_in = state_0.s

    # --- Fixed design target (centerline) and initial wall guess: shared
    #     convergent section ending exactly at the physical sonic pressure
    #     (solved directly, not iterated -- see solve_convergent_ctrl) at
    #     x_sonic_design (your chosen location). Centerline divergent branch
    #     is a smooth 4-CP cubic NURBS (center_x2/center_ctrl3_x/
    #     center_kink_frac) into a flat tail, smooth at BOTH the throat and
    #     the kink; wall divergent branch is a 4-CP cubic NURBS (wall_x2/
    #     wall_ctrl3), smooth at the throat and flat at the exit if
    #     wall_ctrl3's p_frac is left at 1.0. ---
    h0 = state_0.h
    p_center, p_wall_initial, x_sonic, join_pt, ref_ctrl, center_div_ctrl, wall_div_ctrl = \
        build_center_wall_profiles(fluid, p_in, p_out, s_in, h0, x_sonic_design,
                                   conv_handle_frac, center_x2, center_ctrl3_x,
                                   center_kink_frac, wall_x2, wall_ctrl3, x_nodes)
    p_wall = p_wall_initial.copy()
    print(f"  reference curve's sonic point (convergent/divergent split): x/L={x_sonic:.3f}")

    iso_c = march(p_center, h_in, s_in, A_in, u_in)
    v_len = int(np.sum(~np.isnan(iso_c['u'])))

    MIN_VALID_POINTS = 10   # need >k+1 points for the k=4 curvature spline, with margin
    DP_CAP = 0.5 * p_in     # cap curvature-driven correction to avoid feedback blow-up
    residual = None

    if correct_wall_pressure:
        # --- Iterative curvature correction: p_wall is a FREE variable,
        #     corrected each iteration to match what the shape's own
        #     curvature implies via radial equilibrium (see docstring §4). ---
        for it in range(MAX_ITER):
            iso_w = march(p_wall, h_in, s_in, A_in, u_in)
            v_len_iter = min(v_len, int(np.sum(~np.isnan(iso_w['u']))))

            if v_len_iter < MIN_VALID_POINTS:
                print(f"  ! iteration {it}: only {v_len_iter} valid points left (curvature "
                      f"correction likely diverging -- try a larger L_nominal_factor, "
                      f"a smaller RELAX, or a milder p_wall_out target). Stopping early, "
                      f"keeping the last stable wall pressure.")
                break
            v_len = v_len_iter

            x = x_nodes[:v_len]
            rho_c, u_c = iso_c['rho'][:v_len], iso_c['u'][:v_len]
            rho_w, u_w = iso_w['rho'][:v_len], iso_w['u'][:v_len]

            q_avg = 0.5 * (rho_c * u_c + rho_w * u_w)
            h_half = mdot / (2.0 * width * q_avg)

            x_phys = x * L_nominal
            rho_avg = 0.5 * (rho_c + rho_w)
            V_avg = 0.5 * (u_c + u_w)
            dp_curv, h_half_smooth = curvature_dp(x_phys, h_half, rho_avg, V_avg)
            dp_curv = np.clip(dp_curv, -DP_CAP, DP_CAP)

            # NOTE: parallel_exit/parallel_throat are deliberately NOT applied
            # here inside the feedback loop. Tapering the target introduces a
            # local kink in p_wall, which distorts the curvature spline's GLOBAL
            # fit (see curvature_dp) and was found to destabilize convergence
            # (oscillatory, non-convergent). They are instead applied once,
            # after this loop converges -- see below.
            p_wall_target = p_center[:v_len] + dp_curv
            p_wall_target[0] = p_in  # pin the shared inlet value
            p_wall_target = np.clip(p_wall_target, 0.05 * p_in, 3.0 * p_in)

            residual = np.max(np.abs(p_wall_target - p_wall[:v_len])) / p_in
            p_wall[:v_len] = p_wall[:v_len] + RELAX * (p_wall_target - p_wall[:v_len])

            if residual < TOL:
                break
        residual_str = f"{residual:.2e}" if residual is not None else "n/a"
        print(f"  stopped after {it+1} iterations, residual={residual_str}, "
              f"wall exit pressure: initial guess {p_wall_initial[v_len-1]/1e5:.3f} "
              f"-> converged {p_wall[v_len-1]/1e5:.3f} bar (target p_out={p_out/1e5:.3f} bar)")
    else:
        # --- p_wall FIXED: both streamlines' pressure is a genuine, fixed
        #     design input. Nothing left to correct -- each streamline's
        #     state is fully determined by its own imposed p(x), so h(x)
        #     just falls out of mass conservation directly (no iteration).
        #     The curvature-consistency diagnostic below is then purely
        #     informational: it shows how far this fixed pair is from
        #     radial-equilibrium-consistent, without enforcing it. ---
        iso_w = march(p_wall, h_in, s_in, A_in, u_in)
        v_len = min(v_len, int(np.sum(~np.isnan(iso_w['u']))))
        print(f"  p_wall fixed as imposed (correct_wall_pressure: false); "
              f"wall exit pressure = {p_wall[v_len-1]/1e5:.3f} bar "
              f"(target p_out={p_out/1e5:.3f} bar)")

    # --- Final consistent state ---
    iso_w = march(p_wall, h_in, s_in, A_in, u_in)
    v_len = min(v_len, int(np.sum(~np.isnan(iso_w['u']))))
    if v_len < MIN_VALID_POINTS:
        raise RuntimeError(
            f"'{label}': correction loop left only {v_len} valid points -- the "
            f"requested p_wall_out/L_nominal_factor combination is not achievable "
            f"with this shape parametrization. Try a milder p_wall_out (closer to "
            f"p_center_out), a larger L_nominal_factor, or a smaller RELAX.")
    x = x_nodes[:v_len]
    rho_c, u_c, a_c = iso_c['rho'][:v_len], iso_c['u'][:v_len], iso_c['a'][:v_len]
    rho_w, u_w, a_w = iso_w['rho'][:v_len], iso_w['u'][:v_len], iso_w['a'][:v_len]
    q_c, q_w = rho_c * u_c, rho_w * u_w
    q_avg = 0.5 * (q_c + q_w)
    h_half = mdot / (2.0 * width * q_avg)
    x_phys = x * L_nominal

    # --- Optional: equivalent 1D inverse design, using p_center(x) as the
    #     single imposed profile (real A_in/u_in this time, so iso_1d['A']
    #     is directly meaningful) ---
    if compare_1d:
        iso_1d = march(p_center, h_in, s_in, A_in, u_in)
        v_len_1d = int(np.sum(~np.isnan(iso_1d['A'])))
        x_1d_phys = x_nodes[:v_len_1d] * L_nominal
        h_half_1d = iso_1d['A'][:v_len_1d] / (2.0 * width)

    # --- Interior streamlines (equal mass-flux fractions), assuming a
    #     linear flux profile q(n) between the centerline and wall values ---
    stream_n = []
    for k in range(1, N_STREAMLINES):
        f = k / N_STREAMLINES
        a = (q_w - q_c) / (2.0 * h_half)
        b = q_c
        target = f * h_half * q_avg
        with np.errstate(divide='ignore', invalid='ignore'):
            n_k = np.where(np.abs(a) > 1e-6 * np.abs(b),
                          (-b + np.sqrt(np.maximum(b ** 2 + 4 * a * target, 0.0))) / (2 * a),
                          target / b)
        stream_n.append(n_k)

    # --- Plotting ---
    line, = ax_p.plot(x, p_center[:v_len] / 1e5, color=color, lw=2, label=label)
    legend_handles.append(line)
    ax_p.plot(x, p_wall_initial[:v_len] / 1e5, color=color, lw=1, ls=':')
    ax_p.plot(x, p_wall[:v_len] / 1e5, color=color, lw=2, ls='--')

    # Control points + control polygon: reference/convergent curve (thin
    # diamonds), center divergent branch (circles), wall divergent branch
    # (squares) -- all share the same join point at x_sonic.
    def cp_xy(ctrl_frac):
        cx = [pt[0] for pt in ctrl_frac]
        cp = [(p_in - pt[1] * (p_in - p_out)) / 1e5 for pt in ctrl_frac]
        return cx, cp

    ref_x, ref_p = cp_xy(ref_ctrl)
    center_div_x, center_div_p = cp_xy(center_div_ctrl)
    wall_div_x, wall_div_p = cp_xy(wall_div_ctrl)
    ax_p.plot(ref_x, ref_p, 'd--', color=color, lw=0.75, ms=6, mfc='none', mec=color, alpha=0.6)
    ax_p.plot(center_div_x, center_div_p, 'o--', color=color, lw=0.75, ms=6,
             mfc='none', mec=color, alpha=0.6)
    ax_p.plot(wall_div_x, wall_div_p, 's--', color=color, lw=0.75, ms=6,
             mfc='none', mec=color, alpha=0.6)
    ax_p.axvline(x_sonic, color=color, lw=0.5, ls=':', alpha=0.5)

    # Plot the RAW mass-conservation h_half here, not h_half_smooth (that's
    # a heavily-smoothed spline fit used only internally for the curvature
    # estimate in curvature_dp() -- plotting it was introducing spurious
    # wiggles/dips not present in the actual mass-conserving shape).
    ax_shape.plot(x_phys * 1e3, h_half * 1e3, color=color, lw=2)
    ax_shape.plot(x_phys * 1e3, -h_half * 1e3, color=color, lw=2)
    ax_shape.plot(x_phys * 1e3, np.zeros_like(x_phys), color=color, lw=0.75, ls='-.')
    for n_k in stream_n:
        ax_shape.plot(x_phys * 1e3, n_k * 1e3, color=color, lw=0.75, alpha=0.6)
        ax_shape.plot(x_phys * 1e3, -n_k * 1e3, color=color, lw=0.75, alpha=0.6)
    if compare_1d:
        oned_line, = ax_shape.plot(x_1d_phys * 1e3, h_half_1d * 1e3, color='k', lw=1.5,
                                   ls=(0, (5, 2, 1, 2)), label='1D inverse ($p_\\mathrm{center}$)' if i == 0 else None)
        ax_shape.plot(x_1d_phys * 1e3, -h_half_1d * 1e3, color='k', lw=1.5, ls=(0, (5, 2, 1, 2)))
        if i == 0:
            legend_handles.append(oned_line)

    # --- Optional: reference nozzle designed by the actual MOC package
    #     (moc.design_nozzle), for comparison against the inverse-design
    #     shapes above. Real-gas via jaxprop/CoolProp, same fluid and
    #     inlet/exit conditions as this case. Always uses backend="HEOS"
    #     (moc's own default) regardless of this case's own `backend` --
    #     REFPROP was found to make moc's Stage-3 kernel solver fail a
    #     property lookup for at least one fluid (MM) here. The design is
    #     also run at a nominal MOC_Y_T (not this case's actual, much
    #     smaller throat height) because moc's kernel solver was found to
    #     be numerically fragile at very small y_t -- MOC's inviscid
    #     solution is purely geometrically self-similar in y_t (verified:
    #     doubling y_t exactly doubles the returned wall coordinates), so
    #     the result is simply rescaled back to this case's real throat
    #     half-height afterward. The contour is then shifted in x so its
    #     own throat lines up with ours (MOC's x=0 convention isn't
    #     necessarily the same as x_phys's). ---
    MOC_Y_T = 0.01
    if compare_moc:
        try:
            idx_throat = int(np.argmin(h_half))
            scale = h_half[idx_throat] / MOC_Y_T
            valid = ~np.isnan(x_moc) & ~np.isnan(y_moc)
            x_moc, y_moc = x_moc[valid], y_moc[valid]
            x_moc_throat = x_moc[int(np.argmin(y_moc))]
            x_moc_shifted = (x_moc - x_moc_throat + x_phys[idx_throat]) * 1e3
            moc_line, = ax_shape.plot(x_moc_shifted, y_moc * 1e3, color='crimson', lw=1.5,
                                        ls=(0, (1, 1)), label='MOC' if i == 0 else None)
            ax_shape.plot(x_moc_shifted, -y_moc * 1e3, color='crimson', lw=1.5, ls=(0, (1, 1)))
            if i == 0:
                legend_handles.append(moc_line)
        except Exception as exc:
                print(f"  MOC design for '{label}' failed: {exc} -- skipping overlay")

    ax_massflux.plot(x, q_c, color=color, lw=2)
    ax_massflux.plot(x, q_w, color=color, lw=2, ls='--')
    idx_c = int(np.argmax(q_c))
    idx_w = int(np.argmax(q_w))
    ax_massflux.plot(x[idx_c], q_c[idx_c], 'o', color=color, ms=6, mec='black', zorder=3)
    ax_massflux.plot(x[idx_w], q_w[idx_w], 's', color=color, ms=6, mec='black', zorder=3)
    print(f"  actual M=1 location: centerline x/L={x[idx_c]:.3f}, wall x/L={x[idx_w]:.3f}")

    ax_mach.plot(x, u_c / a_c, color=color, lw=2)
    ax_mach.plot(x, u_w / a_w, color=color, lw=2, ls='--')

    # --- Contour maps of P and M across the passage. Only two streamlines
    #     are actually solved (center, wall); the transverse field between
    #     them is NOT physically computed -- pressure is linearly
    #     interpolated in n/h_half(x) (0 at the centerline, 1 at the wall,
    #     same simplification level as the linear-mass-flux assumption
    #     already used for the interior streamlines above), but M is then
    #     evaluated from REAL thermodynamic properties AT that interpolated
    #     local pressure (both streamlines share the same isentrope --
    #     same h_in, s_in -- so a single EOS lookup on P_field is valid),
    #     rather than separately linearly-interpolating the two 1D M(x)
    #     traces. The difference matters right at the wet-to-dry (two-phase
    #     -> superheated) crossing: a(p) genuinely jumps there (a REFPROP
    #     quirk, not a physical discontinuity), and interpolating the two
    #     precomputed M(x) traces smears that jump into a spurious,
    #     roughly axis-normal band (each trace's own jump fading out
    #     linearly toward the other side). Evaluating M from P_field
    #     instead makes the jump follow the actual local Q=1 pressure
    #     contour -- the same shape as any other iso-P/iso-M line, not an
    #     interpolation artifact. ---
    Ny = 41
    frac = np.linspace(-1.0, 1.0, Ny)   # -1 = bottom wall, 0 = centerline, 1 = top wall
    abs_frac = np.abs(frac)
    X = np.tile((x_phys * 1e3)[:, None], (1, Ny))
    Y = h_half[:, None] * 1e3 * frac[None, :]
    P_field_pa = p_center[:v_len, None] + abs_frac[None, :] * (p_wall[:v_len, None] - p_center[:v_len, None])
    P_field = P_field_pa / 1e5
    h0 = h_in + 0.5 * u_in ** 2
    field_state = fluid.get_state(jxp.PSmass_INPUTS, P_field_pa.ravel(),
                                  np.full(P_field_pa.size, s_in))
    h_field = np.asarray(field_state.h).reshape(P_field_pa.shape)
    a_field = np.asarray(field_state.a).reshape(P_field_pa.shape)
    u_field = np.sqrt(np.maximum(2.0 * (h0 - h_field), 0.0))
    M_field = u_field / a_field

    axP, axM = axes2[i]
    cf_p = axP.pcolormesh(X, Y, P_field, shading='gouraud', cmap='viridis')
    fig2.colorbar(cf_p, ax=axP, label='$P$ (bar)', shrink=0.6)
    # axP.set_title(f'{label}: pressure field (linear interp., illustrative)')

    cf_m = axM.pcolormesh(X, Y, M_field, shading='gouraud', cmap='magma')
    fig2.colorbar(cf_m, ax=axM, label='$M$ (-)', shrink=0.6)
    # axM.set_title(f'{label}: Mach field (linear interp., illustrative)')

    M_levels = np.arange(np.ceil(M_field.min() / 0.2) * 0.2, M_field.max(), 0.2)
    for ax, iso_color in ((axP, 'white'), (axM, 'white')):
        cs = ax.contour(X, Y, M_field, levels=M_levels, colors=iso_color,
                        linewidths=0.75, linestyles='-')
        ax.clabel(cs, fmt='%.1f', fontsize=7, colors=iso_color)

    for ax in (axP, axM):
        ax.plot(x_phys * 1e3, h_half * 1e3, color='k', lw=1)
        ax.plot(x_phys * 1e3, -h_half * 1e3, color='k', lw=1)
        ax.set_xlabel('$x$ (mm)')
        ax.set_ylabel('$y$ (mm)')
        ax.set_aspect('equal')

ax_p.set_xlabel('$x/L$ (-)'); ax_p.set_ylabel('$P$ (bar)')
# ax_p.set_title('Center (solid) / Wall: initial guess (dotted), converged (dashed)')
ax_p.set_xlim(0, 1)
axis_proxy = plt.Line2D([], [], color='black', lw=2, ls='-', label='Axis')
wall_proxy = plt.Line2D([], [], color='black', lw=2, ls='--', label='Wall')
ax_p.legend(handles=[axis_proxy, wall_proxy], loc='upper right', fontsize=9, frameon=True)

ax_shape.set_xlabel('$x$ (mm)'); ax_shape.set_ylabel('$y$ (mm)')
# ax_shape.set_title(f'Converged shape + streamlines ($L_\\mathrm{{nominal}}$ = {L_nominal*1e3:.0f} mm)')
ax_shape.set_aspect('equal')

ax_massflux.set_xlabel('$x/L$ (-)'); ax_massflux.set_ylabel(r'$\rho u$ (kg/m$^2$/s)')
# ax_massflux.set_title('Center (solid) / Wall (dashed) -- markers = each M=1 point')
ax_massflux.set_xlim(0, 1)

ax_mach.set_xlabel('$x/L$ (-)'); ax_mach.set_ylabel('$M$ (-)')
# ax_mach.set_title('Center (solid) / Wall (dashed)')
ax_mach.set_xlim(0, 1)

ax_shape.legend(handles=legend_handles, loc='upper right', fontsize=9, frameon=True, ncol=3)
fig.tight_layout()
fig.savefig(results_dir / 'quasi2d_overview.png', dpi=200, bbox_inches='tight')
print(f"Figure saved to {results_dir / 'quasi2d_overview.png'}")

fig2.tight_layout()
fig2.savefig(results_dir / 'quasi2d_fields.png', dpi=200, bbox_inches='tight')
print(f"Figure saved to {results_dir / 'quasi2d_fields.png'}")

plt.show()
