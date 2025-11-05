
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import vaneless_channel_model_v5 as vdm

# 1. Wall friction factor plot
Re = jnp.logspace(jnp.log10(800), 6, 500)
diameter = 1.0
roughness_ratios = jnp.array([0.0, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 10e-3])
colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(roughness_ratios)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel(r"$Re$ $-$ Reynolds number ")
ax.set_ylabel(r"$C_{f,w}$ $-$ Fanning friction factor")
for eps, color in zip(roughness_ratios, colors):
    Cf_w = vdm.get_cf_wall(Re, eps * diameter, diameter)
    ax.plot(Re, Cf_w, label=fr"$\epsilon/D={eps:.3f}$", color=color)
    
ax.grid(True, which="both", ls="--", lw=0.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=jnp.arange(2, 10) * 0.1, numticks=15))
formatter = FuncFormatter(lambda x, p: f'{x:.3f}')
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)
ax.tick_params(axis='both', which='major', labelsize=11)
ax.tick_params(axis='both', which='minor', labelsize=11)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout(pad=1)

# 2. Diffusion loss coefficient
D_over_Dm = jnp.linspace(-0.25, 1.25, 300)
D_m = 1.0
D = D_over_Dm * D_m
b, A, alpha_in, b_in, L_total = 1.0, 1.0, 0.0, 1.0, (1.0/0.4)**(-1.0 / 0.35)
Cf_D, E = vdm.get_cf_diffusion(b, A, D, alpha_in, b_in, L_total)
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()
ax2.set_ylim([-0.1, 1.1])
ax1.plot(D_over_Dm, Cf_D, color="tab:blue", lw=1.5, label=r"$C_{f,D}$")
ax2.plot(D_over_Dm, E, color="tab:red", lw=1.5, label=r"$E$")
ax1.set_xlabel(r"$D/D_m$ $-$ Dimensionless divergence parameter")
ax1.set_ylabel(r"$C_{f,D}$ $-$ Diffusion loss coefficient ", color="tab:blue")
ax2.set_ylabel(r"$E$ $-$ Diffusion efficiency", color="tab:red")
ax1.grid(True, ls="--", lw=0.6)
fig.tight_layout(pad=1)

# 3. Curvature friction factor
alpha_deg = jnp.array([30, 40, 50, 60, 70, 80])  # degrees
kappa_b = jnp.linspace(0.0, 1.0, 300)           # product of curvature * b
b = 1.0                                           # channel width
colors = plt.cm.plasma(jnp.linspace(0.2, 0.8, len(alpha_deg)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel(r"$\kappa b$ $-$ Product of curvature and channel width")
ax.set_ylabel(r"$C_{f,C}$ $-$ Curvature friction factor")
for a_deg, color in zip(alpha_deg, colors):
    alpha = jnp.deg2rad(a_deg)
    curvature = kappa_b / b
    Cf_C = vdm.get_cf_curvature(b, curvature, alpha)
    ax.plot(kappa_b, Cf_C, color=color, lw=1.5, label=fr"$\alpha={a_deg:.0f}^\circ$")
ax.grid(True, ls="--", lw=0.6)
ax.legend(loc="best", fontsize=9)
fig.tight_layout(pad=1)

# Show the plots
plt.show()