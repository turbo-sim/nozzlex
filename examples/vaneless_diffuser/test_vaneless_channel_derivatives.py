import yaml
import jax
import jax.numpy as jnp
import jaxprop as jxp
import equinox as eqx

from nozzlex import vaneless_channel as vcm

from time import perf_counter


def get_geometry_vector(diffuser):
    """Extract a subset of geometry parameters as a flat vector."""
    g = diffuser.geometry
    return jnp.array([g.r_in, g.r_out, g.b_in, g.b_out])

def update_geometry(diffuser, x):
    """Return a new diffuser with updated geometry fields from vector x."""
    return eqx.tree_at(
        lambda d: (d.geometry.r_in, d.geometry.r_out, d.geometry.b_in, d.geometry.b_out),
        diffuser,
        tuple(x)
    )


# -------------------------------------------------------
# 1. Initialize problem
# -------------------------------------------------------
# Load configuration file
config_file = "case_radial_axial_bend.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Define the working fluid
fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

# Create base model
diffuser = vcm.VanelessChannel.from_dict(config, fluid)

# -------------------------------------------------------
# 2. Objective function (scalar Cp_out)
# -------------------------------------------------------
def objective(x, diffuser, fluid):
    """Update geometry, solve, and return outlet Cp."""
    diffuser = update_geometry(diffuser, x)
    out = diffuser.solve()
    return out["Cp"][-1]

# -------------------------------------------------------
# 3. JIT + gradient
# -------------------------------------------------------
objective_jit = jax.jit(objective)
grad_fwd = jax.jit(jax.jacfwd(objective))

# Initial parameter vector
x0 = get_geometry_vector(diffuser)

# Warmup to exclude compilation from timing
_ = objective_jit(x0, diffuser, fluid)
_ = grad_fwd(x0, diffuser, fluid)

# -------------------------------------------------------
# 4. Timing
# -------------------------------------------------------
print("\n--------------------------------------")
print("Forward-mode gradient (geometry subset)")
print("--------------------------------------")

t0 = perf_counter()
val = objective_jit(x0, diffuser, fluid)
t1 = perf_counter()
g_ad = grad_fwd(x0, diffuser, fluid)
t2 = perf_counter()

print(f"Objective eval   : {(t1 - t0)*1e3:.3f} ms")
print(f"Forward gradient : {(t2 - t1)*1e3:.3f} ms")

# -------------------------------------------------------
# 5. Finite-difference verification
# -------------------------------------------------------
rel_eps = 1e-6
g_fd = jnp.zeros_like(x0)

for i in range(len(x0)):
    eps = rel_eps * (jnp.abs(x0[i]) + 1.0)
    x_plus = x0.at[i].add(eps)
    x_minus = x0.at[i].add(-eps)
    f_plus = objective(x_plus, diffuser, fluid)
    f_minus = objective(x_minus, diffuser, fluid)
    g_fd = g_fd.at[i].set((f_plus - f_minus) / (2 * eps))

# -------------------------------------------------------
# 6. Compare AD vs FD
# -------------------------------------------------------
print()
print("--------------------------------------")
print("Gradient comparison (AD vs FD)")
print("--------------------------------------")

for name, g_ad_val, g_fd_val in zip(
    ["r_in", "r_out", "b_in", "b_out"], g_ad, g_fd
):
    abs_err = float(jnp.abs(g_ad_val - g_fd_val))
    rel_err = float(abs_err / (jnp.abs(g_ad_val) + 1e-14))
    print(
        f"{name:<8s}  "
        f"AD: {float(g_ad_val):+.6e}   "
        f"FD: {float(g_fd_val):+.6e}   "
        f"abs.err: {abs_err:.3e}   rel.err: {rel_err:.3e}"
    )

