import jax.numpy as jnp

a_H2O = jnp.pi * 0.25 * 0.08 * 0.1 * 0.0254 * 0.0254
a_N2 = jnp.pi * 0.125 * 0.01 * 0.0254 * 0.0254

N_tubes = 550
A_H2O = a_H2O * N_tubes
A_N2 = a_N2 * N_tubes

# Phase 1 is water
alpha_1 = A_H2O / (A_H2O + A_N2)
alpha_2 = 1.0 - alpha_1

print(f"alpha_1:    {alpha_1}")
print(f"alpha_2:    {alpha_2}")
