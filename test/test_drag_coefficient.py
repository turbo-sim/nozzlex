"""
Plot the drag coefficient (CD) vs. Reynolds number (Re) for two formulations:
  - Schiller-Naumann (S-N): regime-dependent, lax.cond branch
  - Ishii-Zuber (I-Z): regime-dependent, smooth Hermite blend

Three void-fraction cases are shown for each:
  - Bubbly  (α₂ = 0.10, gas dispersed in liquid)
  - Droplet (α₂ = 0.90, liquid dispersed in gas)
  - Transition (α₂ = 0.50, midpoint of transition band)
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from nozzlex.two_fluid.nozzle_model_core import (
    calculate_drag_force,
    calculate_drag_force_ishii_zuber,
)

# ── Default fluid / geometry parameters ──────────────────────────────────────

RHO1  = 1000.0   # liquid density     [kg/m³]
RHO2  = 1.0      # gas density        [kg/m³]
MU1   = 1e-3     # liquid viscosity   [Pa·s]
MU2   = 1e-5     # gas viscosity      [Pa·s]
D     = 1e-3     # bubble/drop diam.  [m]
SIGMA = 0.072    # surface tension    [N/m]  (water/air @ 20 °C)

ALPHA2_VALUES = {
    "bubbly":     0.30,
    "transition": 0.50,
    "droplet":    0.90,
}


# ── Helper: back-calculate CD from a force ───────────────────────────────────
# Both functions share the same prefactor:
#   F = (3/4)/d · |u_rel| · u_rel · CD · alpha_disp · rho_cont
# => CD = F / [(3/4)/d · u_rel² · alpha_disp · rho_cont]

def _u_rel_and_cont(Re, alpha2, rho1, rho2, mu1, mu2, d):
    """Return (u_rel, rho_cont, mu_cont, alpha_disp) for the dominant regime."""
    if alpha2 <= 0.3:
        rho_c, mu_c, alpha_disp = rho1, mu1, alpha2
    else:
        rho_c, mu_c, alpha_disp = rho2, mu2, 1.0 - alpha2
    u_rel = Re * mu_c / (rho_c * d)
    return u_rel, rho_c, alpha_disp


def _recover_cd(F, u_rel, rho_c, alpha_disp, d):
    denom = (3.0 / 4.0) / d * u_rel**2 * alpha_disp * rho_c
    return abs(float(F) / denom) if abs(denom) > 1e-30 else float("nan")


def cd_from_sn(Re_values, alpha2, rho1=RHO1, rho2=RHO2,
               mu1=MU1, mu2=MU2, d=D):
    alpha1 = 1.0 - alpha2
    Cd_list = []
    for Re in Re_values:
        u_rel, rho_c, alpha_disp = _u_rel_and_cont(Re, alpha2, rho1, rho2, mu1, mu2, d)
        F = calculate_drag_force(
            jnp.float32(alpha1), jnp.float32(alpha2),
            jnp.float32(rho1),   jnp.float32(rho2),
            jnp.float32(mu1),    jnp.float32(mu2),
            jnp.float32(u_rel),  jnp.float32(0.0),
            jnp.float32(d),
        )
        Cd_list.append(_recover_cd(F, u_rel, rho_c, alpha_disp, d))
    return np.array(Cd_list)


def cd_from_iz(Re_values, alpha2, rho1=RHO1, rho2=RHO2,
               mu1=MU1, mu2=MU2, d=D, sigma=SIGMA):
    alpha1 = 1.0 - alpha2
    Cd_list = []
    for Re in Re_values:
        u_rel, rho_c, alpha_disp = _u_rel_and_cont(Re, alpha2, rho1, rho2, mu1, mu2, d)
        F = calculate_drag_force_ishii_zuber(
            jnp.float32(alpha1), jnp.float32(alpha2),
            jnp.float32(rho1),   jnp.float32(rho2),
            jnp.float32(mu1),    jnp.float32(mu2),
            jnp.float32(u_rel),  jnp.float32(0.0),
            jnp.float32(d),
            jnp.float32(sigma),
        )
        Cd_list.append(_recover_cd(F, u_rel, rho_c, alpha_disp, d))
    return np.array(Cd_list)


# ── Standalone Schiller-Naumann reference (no volume fraction) ───────────────

def cd_schiller_naumann(Re):
    Re = np.maximum(Re, 1e-12)
    return (24.0 / Re) * (1.0 + 0.15 * Re**0.687)


# ── Sweep ─────────────────────────────────────────────────────────────────────

Re = np.logspace(-1, 4, 300)

Cd_SN = cd_schiller_naumann(Re)

results_sn = {label: cd_from_sn(Re, a2) for label, a2 in ALPHA2_VALUES.items()}
results_iz = {label: cd_from_iz(Re, a2) for label, a2 in ALPHA2_VALUES.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────

colors = {"bubbly": "tab:blue", "transition": "tab:green", "droplet": "tab:red"}
alpha2_labels = {k: f"α₂={v}" for k, v in ALPHA2_VALUES.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, (results, title) in zip(axes, [
    (results_sn, "Schiller–Naumann"),
    (results_iz, "Ishii–Zuber"),
]):
    ax.loglog(Re, Cd_SN, "k--", lw=1.2, label="S-N single-sphere")
    ax.loglog(Re, 24.0 / Re, "k:", lw=0.9, label="Stokes  24/Re")
    for label, Cd in results.items():
        ax.loglog(Re, Cd, color=colors[label], lw=2,
                  label=f"{label.capitalize()}  ({alpha2_labels[label]})")
    ax.set_xlabel("Reynolds number  Re  [–]")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xlim([Re[0], Re[-1]])
    ax.set_ylim([1e-2, 1e3])

axes[0].set_ylabel("Drag coefficient  $C_D$  [–]")
fig.suptitle("Drag coefficient vs. Reynolds number – formulation comparison", fontsize=12)
plt.tight_layout()
plt.show()
