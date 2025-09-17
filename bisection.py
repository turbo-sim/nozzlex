import jax.numpy as jnp
import optimistix as optx
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Define a discontinuous function
# ------------------------------
def f(x, args=None):
    # Negative for x < 1, positive for x >= 1
    return jnp.where(x < 1.0, -1.0, 1.0)

# Sample points finely for interpolation
x_vals = jnp.linspace(0, 2, 400)
y_vals = f(x_vals)

# ------------------------------
# Linear interpolation to detect jump
# ------------------------------
crossings = jnp.where((y_vals[:-1] < 0) & (y_vals[1:] > 0))[0]
x0, x1 = x_vals[crossings[0]], x_vals[crossings[0]+1]
y0, y1 = y_vals[crossings[0]], y_vals[crossings[0]+1]

# Linear interpolation formula for zero crossing
x_jump_linear = x0 - y0*(x1-x0)/(y1-y0)
print("Zero crossing (linear interpolation):", x_jump_linear)

# ------------------------------
# Define a JAX-compatible linear "spline" function
# ------------------------------
def linear_interp_jax(x, args=None):
    return jnp.interp(x, x_vals, y_vals)

# ------------------------------
# Optimistix solvers
# ------------------------------
# Optimistix solvers
solvers = {
    "Bisection": optx.Bisection(rtol=1e-6, atol=1e-6),
    "Newton": optx.Newton(rtol=1e-6, atol=1e-6),
    "Chord": optx.Chord(rtol=1e-6, atol=1e-6),
}

lower, upper = x0, x1
x0_initial = jnp.array(0.5)

for name, solver in solvers.items():
    print(f"\n=== {name} solver ===")
    try:
        sol = optx.root_find(
            linear_interp_jax,  # function now accepts x and args
            solver,
            x0_initial,
            args=None,
            throw=False,
            options={"lower": lower, "upper": upper},
        )
        print(f"Success: {sol.result == optx.RESULTS.successful}")
        print(f"Root: {sol.value:0.6e}")
        print(f"Residual: {float(linear_interp_jax(sol.value)):.6e}")
    except Exception as e:
        print(f"{name} solver failed: {e}")

# ------------------------------
# Plot results
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot(x_vals, np.array(y_vals), label='Discontinuous function', color='blue')
plt.axhline(0, color='black', linestyle='--', label='x-axis')

# Linear interpolation jump
plt.plot(float(x_jump_linear), 0, 'ro', label='Linear interp zero')

# Linear interpolation function for visualization
x_fine = np.linspace(float(lower), float(upper), 100)
plt.plot(x_fine, np.array(linear_interp_jax(x_fine)), 'g--', label='Linear interp approx')

plt.axvline(1.0, color='red', linestyle=':', label='Actual jump')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Discontinuous function zero-crossing detection (JAX-compatible)')
plt.legend()
plt.grid(True)
plt.show()

