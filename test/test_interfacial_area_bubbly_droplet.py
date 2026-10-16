import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


@partial(jax.jit, static_argnames=())
def alpha_b_star(alpha_g, alpha_trans=0.15, alpha_max=0.80):
    """Stadtke's effective bubble void fraction (hyperbolic saturation form).

    Parameters
    ----------
    alpha_g : array-like or scalar
        Gas volume fraction, in [0, 1].
    alpha_trans : float
        End of the linear region (where alpha_b* = alpha_g exactly). Default 0.15.
    alpha_max : float
        Asymptotic upper bound (regime-transition ceiling). Default 0.35.

    Returns
    -------
    alpha_b_star : same shape as alpha_g
    """
    k = alpha_max - alpha_trans
    hyperbolic = alpha_max - (k * k) / (alpha_g - alpha_trans + k)
    # return jnp.where(alpha_g <= alpha_trans, alpha_g, hyperbolic)
    return alpha_g

@partial(jax.jit, static_argnames=())
def alpha_d_star(alpha_g, alpha_trans=0.8, alpha_max=0.35):
    """Stadtke effective droplet (liquid) fraction.

    Equal to alpha_b_star evaluated on the liquid fraction (1 - alpha_g).
    alpha_trans is on the liquid-fraction axis, so default 0.15 puts the
    linear region at alpha_g in [0.85, 1.0] — symmetric with the bubble side.
    """
    return alpha_b_star(1.0 - alpha_g, 1 - alpha_trans, alpha_max)


@partial(jax.jit, static_argnames=())
def smoothstep(alpha_g, alpha_start=0.2, alpha_end=0.8):
    """Smoothstep from 0 at alpha_b_transition to 1 at alpha_d_transition."""
    t = (alpha_g - alpha_start) / (alpha_end - alpha_start)
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@partial(jax.jit, static_argnames=())
def interfacial_area(
    alpha_g,
    d_bubble=4.0,
    d_droplet=0.8,
    alpha_b_transition=0.15,
    alpha_d_transition=0.85,
    alpha_max=0.80,
):
    """Interfacial area density as a function of gas void fraction.

    Blends the bubbly and droplet regimes using a smoothstep weight,
    then computes ai = ab * 6/d_bubble + ad * 6/d_droplet.
    """
    abs_ = alpha_b_star(alpha_g, alpha_b_transition, alpha_max)
    ads_ = alpha_d_star(alpha_g, alpha_d_transition, alpha_max)
    xd = smoothstep(alpha_g, alpha_b_transition, alpha_d_transition)
    xb = 1.0 - xd
    ab = abs_ * xb
    ad = ads_ * xd
    return ab * 6.0 / d_bubble + ad * 6.0 / d_droplet


alphag = jnp.linspace(0,1,100)

alpha_d_star_list = []
alpha_b_star_list = []

alpha_d = []
alpha_b = []

Xd = []
Xb = []

alpha_b_transition = 0.15
alpha_d_transition = 0.85
alpha_max = 0.8

# TODO: Estimate the dropelt diameter using the weber number
d_droplet = 0.8

# TODO: Estimate the bubble  diameter using the weber number
d_bubble = 4.0

Ai = []
Ai_function = []

for a in alphag:
    abs = alpha_b_star(a, alpha_b_transition, alpha_max)
    ads = alpha_d_star(a, alpha_d_transition, alpha_max)

    xd = smoothstep(a, alpha_b_transition, alpha_d_transition)
    xb = 1 - xd

    ab = abs * xb
    ad = ads * xd

    ai = ab * 6 / d_bubble + ad * 6 / d_droplet
    Ai.append(ai)

    aif = interfacial_area(a)
    Ai_function.append(aif)

    alpha_b.append(ab)
    alpha_d.append(ad)

    alpha_d_star_list.append(ads)
    alpha_b_star_list.append(abs)


u1 = 10
u2 = 10
u_rel_abs = jnp.abs(u1 - u2)
u_rel_abs = jnp.maximum(u_rel_abs, 1e-3)
print(u_rel_abs)

plt.plot(alphag, alpha_d_star_list, label="droplet star", color = "red", linestyle="--")
plt.plot(alphag, alpha_b_star_list, label="bubble star", color = "blue", linestyle="--")

plt.plot(alphag, alpha_d, label="droplet", color = "red")
plt.plot(alphag, alpha_b, label="bubble", color = "blue")
plt.ylim([0.0, 0.5])
plt.show()

plt.plot(alphag, Ai)
plt.plot(alphag, Ai_function, linestyle="--")
plt.show()
