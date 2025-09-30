from nozzlex.functions_old import real_gas_prop as rg
import barotropy as bpy
import numpy as np

# the void fractions do not need to be computed if working with specific volumes instead of densities  
def void_fractions(x, rho_V, rho_L):
    eps_V = (x * rho_L) / (x * rho_L + (1 - x) * rho_V)
    eps_L = 1 - eps_V
    return eps_V, eps_L

# Homogeneous Equilibrium Model (HEM)
fluid = rg.Fluid(name="CO2")
p = 6e6
rho = 240

state = fluid.set_state(rg.DmassP_INPUTS, rho, p)
x = state["Q"]
rho_mix = state["rhomass"]
h_mix = state["h"]
rho_L = state["rho_L"]
rho_V = state["rho_V"]
drhodp_L = state["drhodp_L"]
drhodp_V = state["drhodp_V"]
drhodh_L = state["drhodh_L"]
drhodh_V = state["drhodh_V"]

# Without the quality perturbation, the derivs are different with respect the one obtained from coolprop
drho_dp = rho_mix**2 *((x/rho_V**2)*drhodp_V + (1-x)/(rho_L**2)*drhodp_L)
drho_dh = rho_mix**2 *((x/rho_V**2)*drhodh_V + (1-x)/(rho_L**2)*drhodh_L)

print("==============================================================")
print("")
print("============================================")
print("Derivatives for Homogeneous Equilibrium Model")
print("============================================")
print("")
print("Derivatives without considering quality variation")
print("Calculated drhodp:  ", drho_dp)
print("CoolProp drhodp:", state["drho_dP"])
print("")
print("Calculated drhodh:  ", drho_dh)
print("CoolProp drhodh:", state["drho_dh"])
print("")

# Considering the variation of quality when perturbing
dh = max(1e-6 * abs(h_mix), 1e-3 * 5000)
dp = max(1e-6 * abs(p), 1e-3 * (np.exp(0.028782) - 1.0) * p)

state_dp = fluid.set_state(rg.HmassP_INPUTS, h_mix, p+dp)
x_dp = state_dp["Q"]
rho_V_dp = state_dp["rho_V"]
rho_L_dp = state_dp["rho_L"]
vol_V_dp = 1 / rho_V_dp
vol_L_dp = 1 / rho_L_dp
# eps_V_dp, eps_L_dp = void_fractions(x_dp, rho_V_dp, rho_L_dp)

state_dh = fluid.set_state(rg.HmassP_INPUTS, h_mix + dh, p)
x_dh = state_dh["Q"]
rho_V_dh = state_dh["rho_V"]
rho_L_dh = state_dh["rho_L"]
vol_V_dh = 1 / rho_V_dh
vol_L_dh = 1 / rho_L_dh
# eps_V_dh, eps_L_dh = void_fractions(x_dh, rho_V_dh, rho_L_dh)

# rho_dp = eps_V_dp * rho_V_dp + eps_L_dp * rho_L_dp
vol_dp = x_dp * vol_V_dp + (1-x_dp) * vol_L_dp
rho_dp = 1 / vol_dp
# rho_dh = eps_V_dh * rho_V_dh + eps_L_dh * rho_L_dh
vol_dh = x_dh * vol_V_dh + (1-x_dh) * vol_L_dh
rho_dh = 1 / vol_dh

drho_dp = (rho_dp- rho_mix)/dp
drho_dh = (rho_dh- rho_mix)/dh

print("Derivatives considering quality variation")
print("Calculated drhodp:  ", drho_dp)
print("CoolProp drhodp:", state["drho_dP"])
print("")
print("Calculated drhodh:  ", drho_dh)
print("CoolProp drhodh:", state["drho_dh"])
print("")
print("==============================================================")
