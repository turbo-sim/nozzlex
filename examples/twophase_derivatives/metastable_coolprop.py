import CoolProp.CoolProp as CP

# 1. Initialize the AbstractState with REFPROP backend
#    Format: AbstractState("Backend", "Fluid")
backend = "REFPROP"
fluid = "CO2"
AS = CP.AbstractState(backend, fluid)
# AS.build_spinodal()
# 2. Define a state that is physically metastable
#    Example: Liquid water at 150°C (423.15 K) and 1 atm (101325 Pa).
#    Equilibrium state: This should be steam. 
#    Metastable state: We want superheated liquid properties.
T_input = 280
P_input = 1013250

# 3. CRITICAL STEP: Manually specify the phase
#    This tells REFPROP to look at the liquid root of the EOS,
#    ignoring the vapor dome check.
AS.specify_phase(CP.iphase_liquid)



try:
    # 4. Update the state using Pressure and Temperature
    AS.update(CP.PT_INPUTS, P_input, T_input)

    # 5. Retrieve properties
    rho = AS.rhomass()      # Density [kg/m^3]
    h = AS.hmass()          # Enthalpy [J/kg]
    cp = AS.cpmass()        # Specific Heat [J/kg/K]

    print(f"--- Metastable Liquid {fluid} (REFPROP) ---")
    print(f"Temperature: {T_input:.2f} K")
    print(f"Pressure:    {P_input:.2f} Pa")
    print(f"Density:     {rho:.4f} kg/m^3")
    print(f"Enthalpy:    {h:.2f} J/kg")
    
    # 6. Unspecify phase to return to normal behavior for future calculations
    AS.unspecify_phase()

except ValueError as e:
    print("Calculation failed. The state might be too far into the metastable region.")
    print(f"Error: {e}")
