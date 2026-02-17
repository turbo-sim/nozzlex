import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"



import time
import jax
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.functions import (
    nozzle_single_phase_autonomous_ph,
    symmetric_nozzle_geometry,
    NozzleParams,
    IVPSettings,
    replace_param,
    compute_static_state,
)


# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def nozzle_single_phase(
    params_model,
    params_solver,
):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [v, rho, p].
    """
    # Compute inlet conditions iteratively
    state_in = compute_static_state(
        params_model.p0_in,
        params_model.h0_in,
        params_model.Ma_in,
        params_model.fluid,
    )
    p_in, rho_in, a_in, h_in = state_in["p"], state_in["rho"], state_in["a"], state_in["h"]
    v_in = params_model.Ma_in * a_in
    x_in = 1e-9  # Start slightly after the nozzle inlet
    y0 = jnp.array([x_in, p_in, v_in, h_in])

    # Create and configure the solver
    t0 = 0.0  # Start at tau=0 (arbitrary)
    t1 = 1e9  # Large value that will not be reached
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)
    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Bisection(rtol=1e-10, atol=1e-10),
    )

    # Solve the ODE system without saving solution
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-12,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        event=event,
        max_steps=20_000,
    )

    # Solve the ODE system again saving the solution
    ts = jnp.linspace(t0, sol.ts[-1], params_solver.number_of_points)
    saveat = dfx.SaveAt(ts=ts, t1=True, fn=eval_ode_full)
    sol_dense = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=sol.ts[-1],
        dt0=1e-12,
        y0=y0,
        args=params_model,
        saveat=saveat,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        # event=event,  # No need to check event this time
        max_steps=20_000,

    )

    return sol_dense


# Define ODE RHS functions
def eval_ode_full(t, y, args):
    return nozzle_single_phase_autonomous_ph(t, y, args)


def eval_ode_rhs(t, y, args):
    return nozzle_single_phase_autonomous_ph(t, y, args)["rhs_autonomous"]


# Event: stop when position reaches either end of the domain [0, L]
# def eval_end_of_domain_event(t, y, args, **kwargs):
#     x = y[0]
#     L = params_model.length
#     return jnp.minimum(x, L - x)

def eval_end_of_domain_event(t, y, args, **kwargs):
    L = args.length

    # Unpack variables
    x, p, v, h = y
    fluid = args.fluid

    # Compute determinant D
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    a = state["a"]
    d = state["rho"]
    G = state["gruneisen"]

    dddp = (1 + G) / a**2
    dddh = - (d * G) / a**2

    A_mat = jnp.array([
        [v * dddp, d, v * dddh],
        [1.0, d * v, 0.0],
        [v, 0.0, -d * v],
    ])

    D = jnp.linalg.det(A_mat)

    # Two event conditions:
    # 1. Domain end: L - x
    # 2. Determinant-negative condition: 0.2 - x (active only if D < 0)
    cond_1 = L - x
    # cond_2 = jnp.where(D < 0, x - 0.3, 1e9)
    cond_2 = 1.2 -v/a

    # Trigger when either becomes zero
    val = jnp.minimum(cond_1, cond_2)

    return val

# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Define model parameters
    params_model = NozzleParams(
        Ma_in=0.25,
        p0_in=1.0e5,  # Pa
        d0_in=1.20,  # kg/m³
        h0_in=292e3,
        D_in=0.050,  # m
        length=5.00,  # m
        roughness=1e-6,  # m
        T_wall=300.0,  # K
        heat_transfer=0.0,
        wall_friction=0.0,
        # fluid=jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325),
        fluid=jxp.FluidJAX(name="air", backend="HEOS"),
        geometry=symmetric_nozzle_geometry,
    )

    params_solver = IVPSettings(solver_name="Dopri5", rtol=1e-8, atol=1e-8)

    # Inlet Mach number sensitivity analysis
    print("\n" + "-" * 60)
    print("Running inlet Mach number sensitivity analysis")
    print("-" * 60)
    input_array = jnp.linspace(0.1, 0.6, 50)

    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, Ma in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "Ma_in", Ma) 
        sol = nozzle_single_phase(params_model, params_solver)
        print(
            f"Ma_in = {Ma:0.2f} | Solution time: {(time.perf_counter() - t0) * 1e3:7.3f} ms"
        )
        solution_list.append(sol)

    # Create the figure
    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    axs = [ax0, ax1, ax2]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, sol in zip(colors, input_array, solution_list):
        x = sol.ys["x"]
        axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            sol.ys["p"] * 1e-5,
            linestyle="-",
            color=color,
            label=rf"$p_\mathrm{{in}}/p_0 = {val:0.3f}$",
        )
    # axs[0].legend(loc="lower right", fontsize=7)

    axs[1].set_ylabel("Mach number (-)")
    for i, (color, val, sol) in enumerate(zip(colors, input_array, solution_list)):
        axs[1].plot(sol.ys["x"], sol.ys["Ma"], color=color)

    # --- row 3: nozzle geometry ---
    axs[2].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[2].plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    axs[2].set_aspect("equal", adjustable="box")
    axs[2].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    def critical_mach_residual(Mach_in, params_model):
        params_model = replace_param(params_model, "Ma_in", Mach_in)
        sol = nozzle_single_phase(params_model, params_solver)
        max_mach = jnp.max(sol.ys["Ma"])
        jax.debug.print("1-mach = {}", 1-max_mach)
        return 1. - max_mach**2
        # return jnp.min(sol.ys["D"])

    solvers = {
        "Bisection": optx.Bisection(rtol=1e-4, atol=1e-4),
    }

    lower = float(0.2)
    upper = float(0.5)
    x0_initial = 0.4

    print(" ")
    print("-"*80)
    print("Bisection starts")
    print("-"*80)

    for name, solver in solvers.items():
        try:
            sol = optx.root_find(
                critical_mach_residual,
                solver,
                x0_initial,
                args=params_model,
                throw=False,
                options={"lower": lower, "upper": upper},
                # max_steps=20_000,
            )
            print(f"Success: {sol.result == optx.RESULTS.successful}")
            print(f"Root: {sol.value:0.6e}")
            print(f"Residual: {float(critical_mach_residual(sol.value, params_model)):.6e}")
            print(f"Steps: {sol.stats["num_steps"]}")
        except Exception as e:
            print(f"{name} solver failed: {e}")

    # Plotting the function R(Ma_in) = max(Mach) - 1 
    fig = plt.figure(figsize=(6, 5))
    x_vals = input_array
    y_vals = np.zeros(len(x_vals))

    for i, (color, val, sol) in enumerate(zip(colors, input_array, solution_list)):
        residual = critical_mach_residual(input_array[i], params_model)
        plt.plot(input_array[i], residual, marker="o", markerfacecolor=color, markeredgecolor=color, label=rf"$p_\mathrm{{in}}/p_0 = {val:0.3f}$",)   
        y_vals[i] = residual

    plt.plot(x_vals, y_vals, color = "k") 
    plt.xlabel("Inlet Mach number")
    plt.ylabel("1 - max(Mach)")
    plt.legend(loc="best", fontsize=7)
    fig.tight_layout(pad=1)

    # Show figures
    plt.show()
