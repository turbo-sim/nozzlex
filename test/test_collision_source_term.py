import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import jax

def term(alpha, alpha_max=0.81, C=3.0):
    alpha_safe = jnp.minimum(alpha, 0.801)
    a13     = alpha_safe     ** (1.0 / 3.0)
    amax13  = alpha_max ** (1.0 / 3.0)
    denom   = amax13 * (amax13 - a13)
    arg     = C * amax13 * a13 / denom
    return 1 / denom * (1.0 - jnp.exp(-arg))

ALPHA = jnp.linspace(0.0, 0.81, 1000)
TERM = []

TERM = jax.vmap(term)(ALPHA)
# for alpha in ALPHA:
#     t = term(alpha)
#     TERM.append(t)

plt.plot(ALPHA, TERM)
plt.plot([0.801, 0.801], [0.0, 400])
print(TERM[-1])
plt.show()
