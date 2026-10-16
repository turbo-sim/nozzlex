"""
Two-fluid pipe test: integrates the two-fluid autonomous ODE through a straight pipe
for the Elliot nitrogen+water case conditions.
Mirrors the structure of nozzlex/homogeneous_nonequilibrium/test_pipe.py
but uses the two-fluid model.
"""
import time
import sys
import jax
import jax.numpy as jnp
import diffrax as dfx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_core,
    nozzle_two_fluid_autonomous,
)
from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
)


def get_pipe(x, radius=0.01):
    """Straight circular pipe with constant cross-section."""
    area = jnp.pi * radius ** 2
    area_slope = jnp.zeros_like(x)
    perimeter = 2.0 * jnp.pi * radius
    return area, area_slope, perimeter, radius


def run_pipe(params_model, params_ivp):
    """Integrate the two-fluid ODE through a straight pipe."""
    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in   = state1_in["p"]
    rho1_in = state1_in["rho"]
    rho2_in = state2_in["rho"]
    h1_in  = state1_in["h"]
    h2_in  = state2_in["h"]

    # --- Inlet velocities and void fraction ---
    # Both u1 and u2 are prescribed; alpha1 is derived from the mixture ratio.
    # mixture_ratio = m_dot1/m_dot2 = (alpha1*rho1*u1) / (alpha2*rho2*u2)
    # → alpha1 = mixture_ratio * rho2 * u2 / (rho1 * u1 + mixture_ratio * rho2 * u2)
    u1_in = params_model.u1_in
    u2_in = params_model.u2_in
    alpha1_in = (params_model.mixture_ratio * rho2_in * u2_in) / (rho1_in * u1_in + params_model.mixture_ratio * rho2_in * u2_in)
    # alpha1_in = 0.6049
    alpha2_in = 1.0 - alpha1_in

    # # Wrong for testing only
    # alpha2_in = (params_model.mixture_ratio * rho2_in * u2_in) / (rho1_in * u1_in + params_model.mixture_ratio * rho2_in * u2_in)
    # # alpha1_in = 0.6049
    # alpha1_in = 1.0 - alpha2_in



    # u1_in = params_model.u1_in
    # u2_in = rho1_in * u1_in * alpha1_in / (rho2_in * alpha2_in * params_model.mixture_ratio)
        
    # u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1 / params_model.mixture_ratio)


    y_variable = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u1_in, u2_in, p_in, h1_in, h2_in])

    # --- Equilibrate scaling matrices from inlet A_mat ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    x_in_tmp = jnp.array([1e-9])
    y_inlet_tmp = jnp.concatenate([x_in_tmp, y_variable])
    out_inlet = nozzle_two_fluid_autonomous(0.0, y_inlet_tmp, params_model)
    A_mat_inlet = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model = replace_param(params_model, "row_scales", row_scales_in)
    params_model = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_scaled = nozzle_two_fluid_autonomous(0.0, y_inlet_tmp, params_model)
    det_ref_val = jnp.maximum(jnp.abs(out_inlet_scaled["D"]), 1e-10)
    params_model = replace_param(params_model, "det_ref", det_ref_val)

    x_in = jnp.array([1e-9])
    y_inlet = jnp.concatenate([x_in, y_variable])

    # --- State scaling ---
    Y_ref = jnp.array([
        params_model.length,
        1.0,
        1.0,
        rho1_in,
        rho2_in,
        u1_in,
        u2_in,
        p_in,
        h1_in,
        h2_in,
    ])
    y_inlet_hat = y_inlet / Y_ref

    def ode_full_scaled(t, y, args):
        y_phys = y * Y_ref
        return nozzle_two_fluid_autonomous(0.0, y_phys, args)

    def ode_rhs_scaled(t, y, args):
        out = ode_full_scaled(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        conditions = jnp.stack([
            x / L,
            1 - x / L,
        ])
        return jnp.min(conditions)

    # ----------------------------------------------------------------
    # INLET DIAGNOSTICS — evaluate physics before any integration
    # ----------------------------------------------------------------
    out_in = nozzle_two_fluid_autonomous(0.0, y_inlet, params_model)

    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2
    st1 = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    st2 = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    print("\n=== INLET DIAGNOSTICS ===")
    print(f"  alpha1        = {float(alpha1_in):.6f}  |  alpha2      = {float(alpha2_in):.6f}")
    print(f"  rho1          = {float(rho1_in):.4f} kg/m³  |  rho2   = {float(rho2_in):.4f} kg/m³")
    print(f"  u1            = {float(u1_in):.4f} m/s  |  u2          = {float(u2_in):.4f} m/s")
    print(f"  p             = {float(p_in):.2f} Pa")
    print(f"  h1            = {float(h1_in):.2f} J/kg  |  h2          = {float(h2_in):.2f} J/kg")
    print(f"  T1            = {float(st1['T']):.2f} K  |  T2          = {float(st2['T']):.2f} K")
    print(f"  a1            = {float(st1['a']):.2f} m/s  |  a2          = {float(st2['a']):.2f} m/s")
    print(f"  G1 (Gruneisen)= {float(st1['gruneisen']):.4f}  |  G2          = {float(st2['gruneisen']):.4f}")
    print(f"  mu1           = {float(st1['mu']):.4e} Pa·s  |  mu2      = {float(st2['mu']):.4e} Pa·s")
    print(f"  u_rel         = {float(u1_in - u2_in):.4f} m/s")
    print(f"  det(A)        = {float(out_in['D']):.6e}")
    print(f"  det_ref       = {float(params_model.det_ref):.6e}")
    print(f"  rhs_autonomous= {[float(v) for v in out_in['rhs_autonomous']]}")
    print(f"  b_vec         = {[float(v) for v in out_in['b_vec']]}")
    A_mat = out_in["A_mat"]
    print(f"  A_mat row norms = {[float(jnp.linalg.norm(A_mat[i])) for i in range(9)]}")
    rhs_scaled = ode_rhs_scaled(1e-9, y_inlet_hat, params_model)
    print(f"  rhs_scaled    = {[float(v) for v in rhs_scaled]}")
    print(f"  Y_ref         = {[float(v) for v in Y_ref]}")
    print("=========================\n")
    # ----------------------------------------------------------------

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(ode_rhs_scaled)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        # dtmax=1e-3,
    )
    event = dfx.Event(
        cond_fn=event_fn,
        root_finder=optx.Bisection(rtol=1e-3, atol=1e-3),
    )

    saveat = dfx.SaveAt(t1=True, dense=True)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-4,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=100,
        throw=False,
    )

    return sol, Y_ref, params_model


class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
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


def postprocess(sol, Y_ref, params_model, n_plot=1000):
    """Sample the dense solution on a uniform-τ grid and extract physics."""
    t0 = sol.t0
    ts = jnp.linspace(t0, sol.ts[0], n_plot)

    def eval_at_tau(t):
        y_phys = sol.evaluate(t) * Y_ref
        return nozzle_two_fluid_autonomous(0.0, y_phys, params_model)

    results = [eval_at_tau(t) for t in ts]
    scalar_keys = [k for k in results[0] if jnp.isscalar(results[0][k]) or results[0][k].shape == ()]
    return {k: jnp.stack([r[k] for r in results]) for k in scalar_keys}


if __name__ == "__main__":

    log_file = "test_pipe_tfm.txt"
    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    # -------------------------------------------------------------------------
    # Fluid tables
    # -------------------------------------------------------------------------
    table_dir = "fluid_tables"
    N_h, N_p = 130, 130

    # Case 1: water (fluid1) + nitrogen (fluid2) — Elliot 1979
    cases = {
        "water_nitrogen": dict(
            fluid1_name="water",
            fluid2_name="nitrogen",
            h1_range=[0.0, 150e3],
            h2_range=[90e3, 350e3],
            p_range=[3e4, 3e6],
            p0_in=20e5,
            T01_in=22 + 273.15,
            T02_in=22 + 273.15,
            u1_in=4.0,    # water velocity (m/s)
            u2_in=4.0,   # nitrogen velocity (m/s); alpha derived from mixture_ratio
            mixture_ratio=68,
            length=1.0,
            wall_friction = 0.0
        ),
    }

    for case_name, cfg in cases.items():
        print("\n" + "=" * 60)
        print(f"  Case: {case_name}")
        print("=" * 60)

        fluid1 = jxp.FluidBicubic(
            fluid_name=cfg["fluid1_name"],
            backend="HEOS",
            h_min=cfg["h1_range"][0], h_max=cfg["h1_range"][1],
            p_min=cfg["p_range"][0],  p_max=cfg["p_range"][1],
            N_h=N_h, N_p=N_p,
            table_dir=table_dir,
        )
        fluid2 = jxp.FluidBicubic(
            fluid_name=cfg["fluid2_name"],
            backend="HEOS",
            h_min=cfg["h2_range"][0], h_max=cfg["h2_range"][1],
            p_min=cfg["p_range"][0],  p_max=cfg["p_range"][1],
            N_h=N_h, N_p=N_p,
            table_dir=table_dir,
        )

        params_model = NozzleParams(
            p0_in=cfg["p0_in"],
            T01_in=cfg["T01_in"],
            T02_in=cfg["T02_in"],
            length=cfg["length"],
            roughness=0.0,
            Ma_low=0.95,
            Ma_high=1.05,
            Ma_in=0.001,
            u1_in=cfg["u1_in"],
            u2_in=cfg["u2_in"],
            heat_transfer=0.0,
            wall_friction=0.0,
            mixture_ratio=cfg["mixture_ratio"],
            bubble_number=cfg.get("bubble_number", 1e8),
            droplet_number=cfg.get("droplet_number", 1e8),
            fluid1=fluid1,
            fluid2=fluid2,
            geometry=get_pipe,
        )

        params_ivp = IVPSettings(
            solver_name="ImplicitEuler", # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
            adjoint_name="DirectAdjoint",
            number_of_points=60,
            rtol=1e-6,
            atol=1e-6,
        )

        t_start = time.perf_counter()
        sol, Y_ref, params_model = run_pipe(params_model, params_ivp)
        dt_ms = (time.perf_counter() - t_start) * 1e3

        print(f"Solver result: {sol.result} | steps: {int(sol.stats['num_steps'])} | time: {dt_ms:.0f} ms")

        out = postprocess(sol, Y_ref, params_model, n_plot=1000)

        # --- Plot ---
        fig = plt.figure(figsize=(7, 12))
        gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax2 = fig.add_subplot(gs[2], sharex=ax0)
        ax3 = fig.add_subplot(gs[3], sharex=ax0)
        ax4 = fig.add_subplot(gs[4], sharex=ax0)

        x = out["x"]
        ax0.plot(x, out["p"] * 1e-5)
        ax0.set_ylabel("Pressure (bar)")

        ax1.plot(x, out["v1"], label="v1 (water)")
        ax1.plot(x, out["v2"], label="v2 (nitrogen)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.legend(fontsize=7)

        ax2.plot(x, out["alpha1"]/out["alpha1"][0], label="α1")
        ax2.plot(x, out["alpha2"]/out["alpha2"][0], label="α2")
        ax2.set_ylabel("Void fraction")
        ax2.legend(fontsize=7)

        ax3.plot(x, out["T1"], label="T1")
        ax3.plot(x, out["T2"], label="T2")
        ax3.set_ylabel("Temperature (K)")
        ax3.legend(fontsize=7)

        ax4.plot(x, out["rho1"]/out["rho1"][0])
        ax4.plot(x, out["rho2"]/out["rho2"][0])
        ax4.set_ylabel("Density/Inlet Density")
        ax4.set_xlabel("Axial position (m)")

        fig.suptitle(f"Two-fluid pipe — {case_name}", fontsize=11)
        fig.tight_layout()

    plt.show()

    sys.stdout = tee_out.stream
    sys.stderr = tee_err.stream
    tee_out.close()
    tee_err.close()
    print(f"Log saved to {log_file}")
