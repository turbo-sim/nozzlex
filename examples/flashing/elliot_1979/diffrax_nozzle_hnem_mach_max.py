# import os
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"


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

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    nozzle_homogeneous_non_equilibrium_core,
    get_nozzle_elliot,
    compute_inlet_state_hne,
    get_speed_of_sound_mixture
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    nozzle_homogeneuous_non_equilibrium_mach_out,
    nozzle_homogeneuous_non_equilibrium,
    replace_param,
    compute_critical_inlet,
    # compute_static_state,
)
# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
# @eqx.filter_jit
def transonic_nozzle_one_component(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    
    max_mach = nozzle_homogeneuous_non_equilibrium_mach_out(params_model, params_ivp)
    # jax.debug.print("{M}", M=max_mach)

    # Test critical mach function
    # max_mach = compute_critical_inlet(Ma_lower=0.001, Ma_upper=0.005, params_model=params_model, params_solver=params_ivp)
        
    # Ma_in_cr = jnp.min(max_mach["Ma_mix"])
    # maximum_macxh_number = jnp.max(max_mach["Ma_mix"])
    # jax.debug.print("Max_mach={M}", M=max_mach)



    return max_mach

def transonic_nozzle_zero_function(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    
    # max_mach = nozzle_homogeneuous_non_equilibrium_mach_out(params_model, params_ivp)
    # jax.debug.print("{M}", M=max_mach)

    # Test critical mach function
    max_mach = compute_critical_inlet(Ma_lower=0.001, Ma_upper=0.05, params_model=params_model, params_solver=params_ivp)
        
    # Ma_in_cr = jnp.min(max_mach["Ma_mix"])
    # maximum_macxh_number = jnp.max(max_mach["Ma_mix"])
    # jax.debug.print("Max_mach={M}", M=max_mach)

    return max_mach


def _velocity_event_cond(t, y, args, **kwargs):
    """Event function: zero when M^2 - Ma_low^2 = 0."""
    # _, _, _, _, u, _, _, _ = y
    out = nozzle_homogeneous_non_equilibrium_core(t,y, args)
    # a = args.fluid.get_state(jxp.HmassP_INPUTS, h, p)["a"]
    # Ma_sqr = (v / a)**2
    # u_sqr = u ** 2
    # return u_sqr - args.u_low**2
    return out["Ma_mix"] - params_model.Ma_low


def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    rhs_fn = lambda xx, yy: nozzle_homogeneous_non_equilibrium_core(xx, yy, model)["rhs"]
    f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
    Jy = jax.jacrev(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
    Jt = jax.jacrev(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂t
    return f_star, Jy, Jt


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

    # return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------

table_dir = "fluid_tables"
fluid_name = ["water","nitrogen"]
backend = "HEOS"
h1_table = [60e3, 120e3]  # J/kg
h2_table = [200e3, 350e3]
p_table = [3e4, 3e6]   # Pa
N_h = 128
N_p = 128

if __name__ == "__main__":

    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    # -- 1. Find critical state with continuation --

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=20e5,  # Pa
        T01_in = 22 + 273.15, 
        T02_in = 22 + 273.15,
        length=0.250,  # m
        roughness=0e-6,  # m
        Ma_low=0.95,
        Ma_high=1.05,
        Ma_in=0.001,
        heat_transfer=0.0,
        wall_friction=0.0,
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
        mixture_ratio=68
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
        solver_name="Kvaerno3",  #"Kvaerno3",
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-9,
        atol=1e-9,
    )

    # L_elliot = 0.245 # m
    # params_model = replace_param(params_model, "length", L_elliot)


    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.asarray([0.001])
    input_array = jnp.linspace(0.03, 0.038, 20)
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    for i, Ma_in in enumerate(input_array):
        t0 = time.perf_counter()  
        params_model = replace_param(params_model, "Ma_in", Ma_in)
        max_mach = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)

        # # Relative error diagnostics
        # dt_ms = (time.perf_counter() - t0) * 1e3
        # out = sol.ys
        # mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        # h01_error = (out["h01"].max() - out["h01"].min()) / out["h01"][0]
        # h02_error = (out["h02"].max() - out["h02"].min()) / out["h02"][0]
        # s_error = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

        # print(
        #     f"Solution {i} | Solver status {sol.result._value:2d} | "
        #     f"steps {int(sol.stats['num_steps']):3d} | "
        #     f"mdot error {mdot_error:0.2e} | h01 error {h01_error:0.2e} | "
        #     f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        # )

        solution_list.append(1-max_mach)

    max_mach = transonic_nozzle_zero_function(params_model, params_bvp, params_ivp)



    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    plt.plot(input_array, solution_list, "-o", color="k")
    plt.plot(max_mach, 0, "^", color="r")

    # Show figures
    plt.show()
