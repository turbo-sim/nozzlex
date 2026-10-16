
# the T of the nitrgogen is suspicious. it is growing higher than the inlet, higher than ambient. there must be an issue. this can be related to bot the thermal source term or the drag which also appears in the energy eq. 
# check the eq first and then move on
import sys
import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_transport_autonomous,
    nozzle_two_fluid_transport_core,
    get_nozzle_elliot,
    get_nozzle_elliot_fast,
    interfacial_area_dispersed_flows,
)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    nozzle_two_fluid_velocity_crit,
    replace_param,
    compute_critical_inlet_two_fluid,
    eval_end_of_domain_event,
)
# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Non-autonomous solver  (dy/dx = rhs,  x is the integration variable)
# -----------------------------------------------------------------------------
def transonic_nozzle_transport_nonautonomous(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Integrates the two-fluid transport equations in non-autonomous form:
        dy/dx = rhs(x, y)
    where x is the physical axial coordinate.  Cannot pass through a sonic
    point (det→0 stops the integration via an event).
    State y = [α1, α2, ρ1, ρ2, u1, u2, p, h1, h2, NB, ND]  (11 components).
    """
    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in    = state1_in["p"]
    rho1_in = state1_in["rho"];  h1_in = state1_in["h"]
    rho2_in = state2_in["rho"];  h2_in = state2_in["h"]

    alpha1_in = 0.6365890590354819
    alpha2_in = 1.0 - alpha1_in
    u1_in = params_model.u1_in
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1.0 / params_model.mixture_ratio)

    We_crit_b, We_crit_d, sigma_WN = params_model.we_in_bubble, params_model.we_in_droplet, 0.0728
    u_rel_abs   = jnp.maximum(jnp.abs(u1_in - u2_in), 1e-6)
    d_bubb_we   = (We_crit_b * sigma_WN) / (rho1_in * u_rel_abs**2)
    d_drop_we   = (We_crit_d * sigma_WN) / (rho2_in * u_rel_abs**2)
    ai_b, ai_d  = interfacial_area_dispersed_flows(alpha2_in, d_bubb_we, d_drop_we)
    nb_in = ai_b / (jnp.pi * d_bubb_we**2)
    nd_in = ai_d / (jnp.pi * d_drop_we**2)
    A_inlet, _, _, _ = params_model.geometry(1e-9)
    NB_in = A_inlet * u2_in * nb_in
    # ND_in = A_inlet * u1_in * nd_in
    ND_in = 1e3

    y_variable = jnp.array([
        alpha1_in, alpha2_in, rho1_in, rho2_in,
        u1_in, u2_in, p_in, h1_in, h2_in, NB_in, ND_in,
    ])
    # y indices: 0=α1, 1=α2, 2=ρ1, 3=ρ2, 4=u1, 5=u2, 6=p, 7=h1, 8=h2, 9=NB, 10=ND
    Y_ref = jnp.array([
        1.0, 1.0, rho1_in, rho2_in,
        u1_in, u2_in, p_in, h1_in, h2_in, NB_in, ND_in,
    ])

    # --- A-matrix scaling (11×11) ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(11))
    params_model = replace_param(params_model, "col_scales", jnp.ones(11))
    x_tmp   = jnp.array([1e-9])
    Y_tmp   = jnp.concatenate([x_tmp, y_variable])
    out_raw = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    col_scales = jnp.maximum(jnp.max(jnp.abs(out_raw["A_mat"]), axis=0), 1e-30)
    row_scales = jnp.maximum(
        jnp.max(jnp.abs(out_raw["A_mat"] / col_scales[None, :]), axis=1), 1e-30
    )
    params_model = replace_param(params_model, "col_scales", col_scales)
    params_model = replace_param(params_model, "row_scales", row_scales)
    out_scaled = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    params_model = replace_param(
        params_model, "det_ref", jnp.maximum(jnp.abs(out_scaled["D"]), 1e-10)
    )

    # --- Scaled ODE wrappers (t = x, y is 11-component) ---
    def ode_full(t, y, args):
        return nozzle_two_fluid_transport_core(t, y * Y_ref, args)

    def ode_rhs(t, y, args):
        return ode_full(t, y, args)["rhs"] / Y_ref

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        out    = nozzle_two_fluid_transport_core(t, y_phys, args)
        p, h1, h2 = y_phys[6], y_phys[7], y_phys[8]
        p_min = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        return jnp.min(jnp.stack([
            out["D"] + 10,
            p - p_min - 10,
            h1 - fluid1.h_min - 10,
            fluid1.h_max - h1 + 10,
            h2 - fluid2.h_min - 10,
            fluid2.h_max - h2 + 10,
        ]))

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl    = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol, dtmax=1e-4)
    event   = dfx.Event(cond_fn=event_fn, root_finder=optx.Bisection(rtol=1e-3, atol=1e-3))

    sol = dfx.diffeqsolve(
        dfx.ODETerm(ode_rhs),
        solver,
        t0=1e-9,
        t1=float(params_model.length),
        dt0=1e-6,
        y0=y_variable / Y_ref,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=event,
        max_steps=5_000,
        throw=False,
    )

    print(f"\n--- Non-autonomous solver stats ---")
    print(f"  result:          {sol.result}")
    print(f"  x_end          = {float(sol.ts[0]):.6f} m  (L = {params_model.length})")
    print(f"  num_steps      = {sol.stats['num_steps']}")
    print(f"  num_accepted   = {sol.stats['num_accepted_steps']}")
    print(f"  num_rejected   = {sol.stats['num_rejected_steps']}")

    return sol, Y_ref, params_model, ode_full


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
# @eqx.filter_jit
def transonic_nozzle_one_component_autonomus(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    # jax.debug.print("params_model={pm}", pm=params_model)
    # jax.debug.print("params_solver={ps}", ps=params_ivp)


    # [print(f"{k}: {float(v)}") for k, v in vars(params_model).items() if 'fluid' not in k and hasattr(v, '__float__')]
    # [print(f"{k}: {float(v) if hasattr(v, '__float__') else v}") for k, v in vars(params_ivp).items()]

    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in, rho1_in, rho2_in, h1_in, h2_in = (
        state1_in["p"],
        state1_in["rho"],
        state2_in["rho"],
        state1_in["h"],
        state2_in["h"]
    )

    alpha1_in = 0.6365890590354819
    alpha2_in = 1.0 - alpha1_in
    
    u1_in = params_model.u1_in        
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1 / params_model.mixture_ratio)

    # Interfacial area considering blending of two regimes
    We_crit_b = params_model.we_in_bubble
    We_crit_d = params_model.we_in_droplet
    sigma_WN = 0.0728 # N/m
    u_rel_abs = jnp.maximum(jnp.abs(u1_in - u2_in), 1e-6)

    d_bubb_we = (We_crit_b * sigma_WN) / (rho1_in * u_rel_abs**2)
    d_drop_we = (We_crit_d * sigma_WN) / (rho2_in * u_rel_abs**2)

    ai_b, ai_d = interfacial_area_dispersed_flows(alpha2_in, d_bubb_we, d_drop_we)

    nb_in = ai_b / (jnp.pi * d_bubb_we**2)
    nd_in = ai_d / (jnp.pi * d_drop_we**2)

    # jax.debug.print("bubble number:{nb}", nb=nb_in)
    # jax.debug.print("droplet number:{nd}", nd=nd_in)

    A, _, _, _ = params_model.geometry(1e-9)

    NB_in = A * u2_in * nb_in
    ND_in = A * u1_in * nd_in
    # ND_in = 1e3


    # # If no slip is assumed:
    # u2_in = u1_in # no slip at the inlet
    # alpha2_in = 1 / (1 + params_model.mixture_ratio * (rho2_in / rho1_in))
    # alpha1_in = 1 - alpha2_in

    jax.debug.print("alpha1:{a1} |  alpha2:{a2}",a2=alpha2_in, a1=alpha1_in)

    y_variable = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u1_in, u2_in, p_in, h1_in, h2_in, NB_in, ND_in])

    # --- Fixed row/column scales for A_mat (equilibration from inlet A_mat) ---
    # Evaluate the core once at inlet to get the actual A_mat entries
    # Use dummy row/col scales of 1.0 to get the unscaled matrix
    params_model = replace_param(params_model, "row_scales", jnp.ones(11))
    params_model = replace_param(params_model, "col_scales", jnp.ones(11))
    x_in_tmp = jnp.array([1e-9])
    y_inlet_tmp = jnp.concatenate([x_in_tmp, y_variable])
    out_inlet = nozzle_two_fluid_transport_autonomous(0.0, y_inlet_tmp, params_model)
    A_mat_inlet = out_inlet["A_mat"]
    # Column scaling: max absolute value in each column
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    # Row scaling: max absolute value in each row after column scaling
    A_col_scaled = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    jax.debug.print("col_scales: {c}", c=col_scales_in)
    jax.debug.print("row_scales: {r}", r=row_scales_in)
    params_model = replace_param(params_model, "row_scales", row_scales_in)
    params_model = replace_param(params_model, "col_scales", col_scales_in)
    # Compute |det(A_scaled)| at inlet to normalize the autonomous RHS to O(1)
    out_inlet_scaled = nozzle_two_fluid_transport_autonomous(0.0, y_inlet_tmp, params_model)
    det_ref_val = jnp.maximum(jnp.abs(out_inlet_scaled["D"]), 1e-10)
    params_model = replace_param(params_model, "det_ref", det_ref_val)

    x_in = jnp.array([1e-9])  # start slightly after inlet
    y_inlet = jnp.concatenate([x_in, y_variable])

    # --- State-variable scaling ---
    # Reference scales so that y_hat = Y / Y_ref is O(1) for all components.
    # Improves Jacobian conditioning for the implicit solver.
    #   Y = [x,    α1,  α2,  ρ1,      ρ2,      u1,    u2,    p,    h1,     h2    ]
    Y_ref = jnp.array([
        params_model.length,  # x ~ L
        1.0,                  # α1 already O(1)
        1.0,                  # α2 already O(1)
        rho1_in,              # ρ1 ~ initial liquid density
        rho2_in,              # ρ2 ~ initial gas density
        u1_in,                # u1 ~ initial liquid velocity
        u2_in,                # u2 ~ initial gas velocity
        p_in,                 # p  ~ initial pressure
        h1_in,                # h1 ~ initial liquid enthalpy
        h2_in,                # h2 ~ initial gas enthalpy
        NB_in,                # NB ~ initial bubble number flux
        ND_in,                # ND ~ initial droplet number flux
    ])

    # --- Scaled ODE wrappers ---
    def ode_full_scaled(t, y, args):
        """Evaluate full ODE output from scaled state."""
        y_phys = y * Y_ref
        return nozzle_two_fluid_transport_autonomous(0.0, y_phys, args)

    def ode_rhs_scaled(t, y, args):
        """RHS in scaled coordinates: dy_hat/dτ = f(y_hat * Y_ref) / Y_ref."""
        out = ode_full_scaled(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def eval_end_of_domain_event_scaled(t, y, args, **kwargs):
        """Event function: unscale before evaluating physical conditions."""
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_two_fluid_transport_autonomous(0.0, y_phys, args)
        det = jnp.min(out.get("Ma_mix"))
        M = 1.1 - det
        # M = 10

        p = y_phys[7]
        p_min = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_end, h2_end =  y_phys[8], y_phys[9]
        h1_min, h1_max = fluid1.h_min, fluid1.h_max
        h2_min, h2_max = fluid2.h_min, fluid2.h_max
 

        conditions = jnp.stack([
            x / L,                  # 0: x >= 0
            1 - x / L,              # 1: x <= L
            M,                      # 2: determinant condition
            p - p_min - 10,         # 3: p above table min
            h1_end - h1_min - 10,
            h1_max - h1_end + 10,
            h2_end - h2_min - 10,
            h2_max - h2_end + 10,

        ])
        min_val = jnp.min(conditions)
        min_idx = jnp.argmin(conditions)
        # jax.debug.print("event: min_idx={i} min_val={v} x={x}", i=min_idx, v=min_val, x=x)
        return min_val

    y_inlet_hat = y_inlet / Y_ref

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(ode_rhs_scaled)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,      # hard cap on pseudo-time step size
    )
    event = dfx.Event(
        cond_fn=eval_end_of_domain_event_scaled,
        root_finder=optx.Bisection(rtol=1e-3, atol=1e-3),
    )

    # --- Diagnostic: evaluate RHS at inlet to check magnitudes ---

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, dense=True)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=6_000,
        throw=False,
    )

    # --- Solver stats ---
    print(f"\n--- Solver stats ---")
    print(f"  result: {sol1.result}")
    print(f"  τ_end = {float(sol1.ts[0]):.6e}")
    print(f"  num_steps = {sol1.stats['num_steps']}")
    print(f"  num_accepted = {sol1.stats['num_accepted_steps']}")
    print(f"  num_rejected = {sol1.stats['num_rejected_steps']}")

    return sol1, Y_ref, params_model

# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------

table_dir = "fluid_tables"
fluid_name = ["water","nitrogen"]
backend = "HEOS"
h1_table = [0, 150e3]  # J/kg
h2_table = [90e3, 350e3]
p_table = [3e4, 3e6]   # Pa
N_h = 130
N_p = 130


@eqx.filter_jit
def _eval_model_batch(Y, params):
    return jax.lax.map(lambda y: nozzle_two_fluid_transport_autonomous(0.0, y, params), Y)

class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        # Explicitly set the encoding to utf-8 here
        self.file = open(filepath, "w", encoding="utf-8")
        
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        
    def flush(self):
        self.stream.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

if __name__ == "__main__":

    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)


    log_file = "transport_debug.txt"
    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err
    # -- 1. Find critical state with continuation --

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=20e5,  # Pa
        T01_in = 22 + 273.15, 
        T02_in = 22 + 273.15,
        length=0.3, # m
        roughness=0e-6, # m
        Ma_low=0.95,
        Ma_high=1.2,
        Ma_in=0.001,
        u1_in=1e-12,
        heat_transfer=1.0,
        wall_friction=1.0,
        fluid1=jxp.FluidBicubic(
            fluid_name=fluid_name[0],
            backend="HEOS",
            h_min=h1_table[0],
            h_max=h1_table[1],
            p_min=p_table[0],
            p_max=p_table[1],
            N_h=N_h,
            N_p=N_p,
            table_dir=table_dir,
        ),
        fluid2=jxp.FluidBicubic(
            fluid_name=fluid_name[1],
            backend="HEOS",
            h_min=h2_table[0],
            h_max=h2_table[1],
            p_min=p_table[0],
            p_max=p_table[1],
            N_h=N_h,
            N_p=N_p,
            table_dir=table_dir,
        ),
        # fluid1=jxp.FluidJAX(
        #     fluid_name[0],
        #     backend="HEOS",
        # ),
        # fluid2=jxp.FluidJAX(
        #     fluid_name[1],
        #     backend="HEOS",
        # ),
        geometry=get_nozzle_elliot,
        # geometry=symmetric_nozzle_geometry,
        mixture_ratio=68,
        bubble_number=1e8,
        droplet_number=1e8,
        we_in_bubble=8.0,
        we_in_droplet=8.0,
    )

    params_bvp = BVPSettings(
        solve_mode="mach_crit",
        num_points=60,
        rtol=1e-8,
        atol=1e-8,
        max_steps=500,
        jac_mode="bwd",
        verbose=False,
        method="Bisection",
        warmup_method="Dogleg",
        warmup_steps=0,
    )

    params_ivp = IVPSettings(
        solver_name="Kvaerno3", # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-7,
        atol=1e-7,
    )

    # L_elliot = 0.2 # m
    # params_model = replace_param(params_model, "length", L_elliot)


    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.linspace(0.001,0.005,5)
    # input_array = jnp.linspace(3.0, 3.5, 5)
    input_array = jnp.array([3.0]) # 3.447283238172531 2.8043222356472395 3.306200761348009, 3062005937099457
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "u1_in", p0_in)

        # --- Non-autonomous solver (t = x) ---
        # sol, Y_ref, params_model, ode_full = transonic_nozzle_transport_nonautonomous(params_model, params_bvp, params_ivp)

        # dt_ms = (time.perf_counter() - t0) * 1e3

        # # Post-processing: ts IS x (no tau->x mapping needed)
        # n_fine = 20000
        # ts = jnp.linspace(sol.t0, sol.ts[0], n_fine)          # uniform x grid
        # y_hat_uniform = jax.vmap(sol.evaluate)(ts)             # (n_fine, 11) scaled
        # out2 = y_hat_uniform * Y_ref[None, :]                  # (n_fine, 11) physical
        # # Prepend x column so _eval_model_batch gets [x, α1, ..., ND]
        # out_with_x = jnp.concatenate([ts[:, None], out2], axis=1)  # (n_fine, 12)
        # out = _eval_model_batch(out_with_x, params_model)

        # --- Autonomous solver (t = τ, x is state component 0) ---
        sol, Y_ref, params_model = transonic_nozzle_one_component_autonomus(params_model, params_bvp, params_ivp)
        dt_ms = (time.perf_counter() - t0) * 1e3
        n_fine = 20000
        ts = jnp.linspace(sol.t0, sol.ts[0], n_fine)          # uniform τ grid
        y_hat_uniform = jax.vmap(sol.evaluate)(ts)             # (n_fine, 12) scaled
        out2 = y_hat_uniform * Y_ref[None, :]                  # (n_fine, 12) physical, col0=x
        out = _eval_model_batch(out2, params_model)

        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h01_error  = (out["h1"].max() - out["h1"].min()) / out["h1"][0]
        h02_error  = (out["h2"].max() - out["h2"].min()) / out["h2"][0]
        s_error    = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

        print(
            f"Solution {i} | Solver status {sol.result} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h01 error {h01_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        solution_list.append(sol)
        out_list.append(out)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    # xg = solution_list[0].ys["x"]
    # rg = solution_list[0].ys["diameter"] / 2.0
    xg = out_list[0]["x"]
    rg = out_list[0]["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Ma_mix (-)")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        # axs[0].plot(x, out["p"] * 1e-5, linestyle="-", color=color,
        #             label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")
        axs[0].plot(x, out["Ma_mix"], linestyle="-", color=color,
                    label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")    
        # axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: velocity ---
    axs[1].set_ylabel("Velocity (m/s)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[1].plot(out["x"], out["v1"], color=color, label="water")
        axs[1].plot(out["x"], out["v2"], color=color, linestyle="--", label="nitrogen")
        axs[1].legend(loc="upper right", fontsize=7)

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Diameter [m]")
    ax2_right = axs[2].twinx()
    ax2_right.set_ylabel("Number density [1/m³]")
    ax2_right.set_yscale("log")
    for color, val, out in zip(colors, input_array, out_list):
        axs[2].plot(out["x"], out["d_bubble"], color="r", label="d_bubble")
        axs[2].plot(out["x"], out["d_droplet"], color="b", label="d_droplet")
        ax2_right.plot(out["x"], out["nb"], color="r", linestyle="--", label="nb")
        ax2_right.plot(out["x"], out["nd"], color="b", linestyle="--", label="nd")
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    axs[2].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    
    # Determinant
    axs[3].set_ylabel("Void Fraction")
    for color, val, out in zip(colors, input_array, out_list):
        axs[3].plot(out["x"], out["alpha2"], color=color)

    # --- row 3: nozzle geometry ---
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # plt.show()
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    axs = [ax0, ax1]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("NB")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        axs[0].plot(x, out["NB"], linestyle="-", color=color,
                    label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")    
        # axs[0].legend(loc="upper right", fontsize=7)
    
    axs[1].set_ylabel("ND")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        axs[1].plot(x, out["ND"], linestyle="-", color=color,
                    label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")    
    # plt.set_loglevel()

    # Show figures
    plt.show()
