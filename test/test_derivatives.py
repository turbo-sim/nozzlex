import jaxprop as jxp
import jax.numpy as jnp
import jax
# Create and set the fluid
from CoolProp import AbstractState
import CoolProp.CoolProp as CP

n = 100
P = jnp.linspace(3e4, 3e6, 100)
H1 = jnp.linspace(80e3, 100e3, 100)
H2 = jnp.linspace(290e3, 310e3, 100)

fluid1 = jxp.FluidJAX("water", "HEOS")
AS1 = AbstractState("HEOS", "Water")

fluid2 = jxp.FluidJAX("nitrogen", "HEOS")
AS2 = AbstractState("HEOS", "Nitrogen")

i=0

for p in P:

    state1 = fluid1.get_state(jxp.HmassP_INPUTS, H1[i], p)
    a1 = state1["a"]
    rho1 = state1["rho"]
    G1 = state1["gruneisen"]



    # --- Thermodynamic state fluid 2 ---
    state2 = fluid2.get_state(jxp.HmassP_INPUTS, H2[2], p)
    a2 = state2["a"]
    rho2 = state2["rho"]
    G2 = state2["gruneisen"]


        # Update the state
    AS1.update(CP.HmassP_INPUTS, H1[i], p)


    # Now get the derivative (∂ρ/∂p)_h
    drho_dP_1 = AS1.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    drho_dh_1 = AS1.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

            # Update the state
    AS2.update(CP.HmassP_INPUTS, H2[i], p)

    # Now get the derivative (∂ρ/∂p)_h
    drho_dP_2 = AS2.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    drho_dh_2 = AS2.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)


    deltap1 = (drho_dP_1 - (1 + G1) / a1**2)/drho_dP_1
    deltap2 = (drho_dP_2 - ((1 + G2) / a2**2))/drho_dP_2
    deltah1 = (drho_dh_1 - (-(rho1 * G1) / a1**2))/drho_dh_1
    deltah2 = (drho_dh_2 - (-(rho2 * G2) / a2**2))/drho_dh_2

    i=i+1

    print(f"Water: dp2={deltap1} | dh1={deltah1}")
    print(f"Nitrogen: dp2={deltap2} | dh2={deltah2}")
