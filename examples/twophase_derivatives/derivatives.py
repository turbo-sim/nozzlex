from nozzlex.functions_old import real_gas_prop as rg
    
def void_fractions(x, rho_V, rho_L):
    eps_V = (x * rho_L) / (x * rho_L + (1 - x) * rho_V)
    eps_L = 1 - eps_V
    return eps_V, eps_L


fluid = rg.Fluid(name="CO2")
h = 300e3
p = 1e6
state = fluid.set_state(rg.HmassP_INPUTS, h, p)
x = state["Q"]
rho_mix = state["rhomass"]
h_mix = h
rho_L = state["rho_L"]
rho_V = state["rho_V"]
drhodp_L = state["drhodp_L"]
drhodp_V = state["drhodp_V"]
drhodh_L = state["drhodh_L"]
drhodh_V = state["drhodh_V"]

# Without the quality perturbation, the derivs are different with respect the one obtained from coolprop
drho_dp = rho_mix**2 *((x/rho_V**2)*drhodp_V + (1-x)/(rho_L**2)*drhodp_L)
drho_dh = rho_mix**2 *((x/rho_V**2)*drhodh_V + (1-x)/(rho_L**2)*drhodh_L)

print("=====================================================")
print("Derivatives without considering quality variation")
print("")
print("Calculated drhodp:  ", drho_dp)
print("CoolProp drhodp:", state["drho_dP"])
print("")
print("Calculated drhodh:  ", drho_dh)
print("CoolProp drhodh:", state["drho_dh"])
print("")

# Considering the variation of quality when perturbing
dp = 1e-1 # Pressure Perturbation
dh = 1e-1 # Enthalpy perturbation

state_dp = fluid.set_state(rg.HmassP_INPUTS, h_mix, p+dp)
x_dp = state_dp["Q"]
rho_V_dp = state_dp["rho_V"]
rho_L_dp = state_dp["rho_L"]
eps_V_dp, eps_L_dp = void_fractions(x_dp, rho_V_dp, rho_L_dp)

state_dh = fluid.set_state(rg.HmassP_INPUTS, h_mix + dh, p)
x_dh = state_dh["Q"]
rho_V_dh = state_dh["rho_V"]
rho_L_dh = state_dh["rho_L"]
eps_V_dh, eps_L_dh = void_fractions(x_dh, rho_V_dh, rho_L_dh)

rho_dp = eps_V_dp * rho_V_dp + eps_L_dp * rho_L_dp
rho_dh = eps_V_dh * rho_V_dh + eps_L_dh * rho_L_dh

drho_dp = (rho_dp- rho_mix)/dp
drho_dh = (rho_dh- rho_mix)/dh

print("============================================")
print("Derivatives considering quality variation")
print("")
print("Calculated drhodp:  ", drho_dp)
print("CoolProp drhodp:", state["drho_dP"])
print("")
print("Calculated drhodh:  ", drho_dh)
print("CoolProp drhodh:", state["drho_dh"])
print("")

