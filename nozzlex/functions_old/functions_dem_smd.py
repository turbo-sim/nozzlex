import scipy.linalg
import scipy.integrate
import numpy as np
import matplotlib as mpl
from cycler import cycler
import numpy as np
from scipy.linalg import det
import CoolProp.CoolProp as CP
import jaxprop as jxp
import pandas as pd
import os
import barotropy as bpy
from . import real_gas_prop as rg
from .SmoothWallNozzleGeometry import SmoothWallNozzleGeometry


def print_dict(data, indent=0):
    """
    Recursively prints nested dictionaries with indentation.
 
    Parameters
    ----------
    data : dict
        The dictionary to print. It can contain nested dictionaries as values.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
        It controls the number of spaces before each key.
 
    Returns
    -------
    None
 
    Examples
    --------
    >>> data = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    >>> print_dict(data)
    a: 1
    b:
        c: 2
        d:
            e: 3
    """
 
    for key, value in data.items():
        print("    " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print("")
            print_dict(value, indent + 1)
        else:
            print(value)
            

def save_selected_to_csv(out_dict, keys_to_save, filename):
    """
    Save selected entries from a simulation output dictionary to output/filename.

    Parameters:
    - out_dict: dict containing simulation data
    - keys_to_save: list of keys to include in the CSV
    - filename: name of the CSV file only (e.g., "run1.csv")
    """
    # Ensure 'output/' directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Full path to the output file
    full_path = os.path.join(output_dir, filename)

    # Expand scalars to match the length of the first non-scalar entry
    first_key = next(key for key in keys_to_save if not isinstance(out_dict[key], (int, float)))
    length = len(out_dict[first_key])

    filtered = {}
    for key in keys_to_save:
        val = out_dict[key]
        filtered[key] = [val] * length if isinstance(val, (int, float)) else val

    # Save to CSV
    df = pd.DataFrame(filtered)
    df.to_csv(full_path, index=False)


def postprocess_ode(t, y, ode_handle):
    """
    Post-processes the output of an ordinary differential equation (ODE) solver.

    This function takes the time points and corresponding ODE solution matrix,
    and for each time point, it calls a user-defined ODE handling function to
    process the state of the ODE system. It collects the results into a
    dictionary where each key corresponds to a state variable and the values
    are numpy arrays of that state variable at each integration step.

    Parameters
    ----------
    t : array_like
        Integration points at which the ODE was solved, as a 1D numpy array.
    y : array_like
        The solution of the ODE system, as a 2D numpy array with shape (n,m) where
        n is the number of points and m is the number of state variables.
    ode_handle : callable
        A function that takes in a integration point and state vector and returns a tuple,
        where the first element is ignored, the second is a dictionary representing the processed state,
        and the third is a flag (0 or 1) indicating singularity.

    Returns
    -------
    ode_out : dict
        A dictionary where each key corresponds to a state variable and each value
        is a numpy array containing the values of that state variable at each integration step.
        Also includes "singularity_detected" key with a boolean flag.
    """
    # Initialize ode_out as a dictionary
    ode_out = {}
    singularity_detected = False

    for t_i, y_i in zip(t, y.T):
        _, out, flag = ode_handle(t_i, y_i)

        if flag == 1:
            singularity_detected = True
        
        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    # Add singularity info
    ode_out["singularity_detected"] = singularity_detected

    return ode_out

# Post-process when an autonomous system is solved: 
# the output of the solver is only y and not t,y
def postprocess_ode_autonomous(t, y, ode_handle):
    """
    """
    # Initialize ode_out as a dictionary
    ode_out = {}

    for t_i, y_i in zip(t, y.T):
        _, out = ode_handle(y_i)

        for key, value in out.items():
            # Initialize with an empty list
            if key not in ode_out:
                ode_out[key] = []
            # Append the value to list of current key
            ode_out[key].append(value)

    # Convert lists to numpy arrays
    for key in ode_out:
        ode_out[key] = np.array(ode_out[key])

    return ode_out


def get_linear_convergent_divergent(z_coordinate, convergent_length, divergent_length, radius_in, radius_throat, radius_out, width, type):

    if type == "Planar":
        area_in = 2*radius_in*width
        area_throat = 2*radius_throat*width
        area_out = 2*radius_out*width

        dAdz_div = (area_out - area_throat)/divergent_length
        dAdz_conv = (area_throat - area_in)/convergent_length

        if z_coordinate <= convergent_length:
            area_slope = (area_throat-area_in)/convergent_length
            area = area_in + area_slope*z_coordinate
        else:
            area_slope = (area_out-area_throat)/divergent_length
            area = area_throat + area_slope*(z_coordinate-convergent_length)

        radius = area/(2*width)
        perimeter = 2 * (width + 2 * radius)

    return area, area_slope, perimeter, radius, dAdz_div, dAdz_conv


def get_wall_friction(velocity, density, viscosity, roughness, diameter):
    """
    Computes the frictional stress at the wall of a pipe due to viscous effects.

    The function first calculates the Reynolds number to characterize the flow.
    It then uses the Haaland equation to find the Darcy-Weisbach friction factor.
    Finally, it calculates the wall shear stress using the Darcy-Weisbach equation.

    Parameters
    ----------
    velocity : float
        The flow velocity of the fluid in the pipe (m/s).
    density : float
        The density of the fluid (kg/m^3).
    viscosity : float
        The dynamic viscosity of the fluid (Pa·s or N·s/m^2).
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    stress_wall : float
        The shear stress at the wall due to friction (Pa or N/m^2).
    friction_factor : float
        The Darcy-Weisbach friction factor, dimensionless.
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    """
    reynolds = velocity * density * diameter / viscosity
    friction_factor = get_friction_factor_haaland(reynolds, roughness, diameter)
    stress_wall = (1 / 8) * friction_factor * density * velocity**2
    return stress_wall, friction_factor, reynolds


def get_friction_factor_haaland(reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    f = (-1.8 * np.log10(6.9 / reynolds + (roughness / diameter / 3.7) ** 1.11)) ** -2
    return f


# The method pipeline_steady_state_1D computes the critical mass flow rate with the Possible-Impossible flow (PIF) algorithm (if specified, else one can directly impose the mass flow/velocity at the inlet)
# and integrate in space to find the adapted supersonic solution. The PIF algorithm is the simplest among the ones available in the 
# iterature and the ODE is a conventional ODE system and not an autonomous system of equations.

def dem_term_angielczyk_1(perimeter, p, gamma, p_sat_T_LM, p_cr, area):

    ## Angielczyk tuned the correlation for water with this coefficients for CO2
    # C_1 = 5.17
    # C_2 = 0.87
    # C_3 = 0.25

    ## Accoding to Angielczyk, correlation originally developed for water!!
    C_1 = 0.008390
    C_2 = 0.633691
    C_3 = 0.228127

    base = (p_sat_T_LM - p) / (p_cr - p_sat_T_LM)

    f = (C_1 * perimeter / area + C_2) * (1 - gamma) * base ** C_3

    return f

def dem_term_angielczyk_2(p, gamma, p_sat_T_LM, p_cr, dAdz_div, dAdz_conv):

    ## New correlation developed by Angielczyk for CO2
    C_1 = 38
    C_2 = 1.291e-31
    C_3 = 75.28
    C_4 = -0.22

    dAdz_ref = 0.0801424e-4

    f = (C_1 + C_2*np.exp(C_3*(dAdz_div - dAdz_conv)/(dAdz_ref - dAdz_conv)))*(1-gamma)*((p_sat_T_LM - p) / (p_cr - p_sat_T_LM))**C_4 
    
    return f


def richardson(vol_frac_v):

    return (1-vol_frac_v)**(-1.75)


def speed_sound_dem(gamma, quality_hem, state_L, state_V, rho_mix_dem, rho_meta, c_meta): 

    speed_sound_L = state_L["speed_sound"]
    cp_L = state_L["cpmass"]
    rho_L = state_L["rhomass"]
    dsdp_L = state_L["dsdp_L"]

    speed_sound_V = state_V["speed_sound"]
    cp_V = state_V["cpmass"]
    rho_V = state_V["rhomass"]
    dsdp_V = state_V["dsdp_V"]

    T_mix = state_V["T"]

    vol_frac_V = (quality_hem/rho_V) / (quality_hem/rho_V + (1 - quality_hem)/rho_L)
    vol_frac_L = ((1 - quality_hem)/rho_L) / (quality_hem/rho_V + (1 - quality_hem)/rho_L)

    rho_mix_hem = vol_frac_L * rho_L + vol_frac_V * rho_V

    # Speed of sound of the two-phase mixture
    mechanical_equilibrium = vol_frac_L / (
        rho_L * speed_sound_L**2
    ) + vol_frac_V / (rho_V * speed_sound_V**2)
    thermal_equilibrium = T_mix * (
        vol_frac_L * rho_L / cp_L * dsdp_L**2
        + vol_frac_V * rho_V / cp_V * dsdp_V**2
    )
    compressibility_HEM = mechanical_equilibrium + thermal_equilibrium
    c_hem = (1 / rho_mix_hem / compressibility_HEM) ** 0.5

    c_dem = np.sqrt(1 / (rho_mix_dem**2 * ((gamma / (rho_mix_hem**2 * c_hem**2)) + (1 - gamma) / (rho_meta**2 * c_meta**2))))

    return c_dem

def pipeline_steady_state_1D_autonomous(
    fluid_name,
    pressure_in,
    temperature_in,
    properties_in,
    convergent_length,
    divergent_length,
    roughness,
    radius_in,
    radius_throat,
    radius_out,
    nozzle_type,
    width,
    mass_flow=None,
    mach_in=None,
    critical_flow = False,
    include_friction=True,
    include_heat_transfer=False,
    temperature_external=None,
    number_of_points=None,
):
    # Check for correct inputs
    if (
        (mass_flow is None and mach_in is None and critical_flow is False) or 
        (mass_flow is not None and mach_in is not None) 
    ):
        raise ValueError(
            "Check input settins for the velocity."
        )
    
    fluid_bpy = bpy.Fluid(fluid_name, backend="HEOS")
    fluid_rg = rg.Fluid(fluid_name, backend="HEOS")
    p_cr = fluid_bpy.critical_point.p
    
    # Define inlet area and length of the nozzle
    total_length = 0.4631
    if nozzle_type == "Planar":
        area_in = 2*radius_in*width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi*radius_in**2

    # Calculate inlet density
    enthalpy_in = properties_in.h
    density_in = properties_in.rho
    s_frozen = properties_in.s

    # Calculate velocity based on specified parameter
    if mass_flow is not None:
        velocity_in = mass_flow / (area_in * density_in)
    elif mach_in is not None:
        velocity_in = mach_in * properties_in["a"]
    elif critical_flow is True:
        
        mach_impossible = 0.25
        mach_possible = 0.0000001
        u_impossible = mach_impossible*properties_in.a
        u_possible = mach_possible*properties_in.a
        m_impossible = density_in*u_impossible*area_in
        m_possible = density_in*u_possible*area_in
        m_possible = 15
        m_impossible = 17
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []

    SuperMobiDick = SmoothWallNozzleGeometry(file_path=r"C:\Users\ancio\OneDrive - Danmarks Tekniske Universitet\Documents\python_scripts\nozzlex\nozzlex\functions_old\nozzle_coordinates_smd.csv")
    SuperMobiDick.discretize_geometry()
    SuperMobiDick.calculate_inclination_angles()
    SuperMobiDick.compute_circumferential_areas()
    # SuperMobiDick.visualize_geometry()
    sectional_positions = SuperMobiDick.get_sectional_positions()
    sectional_radii = SuperMobiDick.get_sectional_heights()
    pressure_gradient_areas = SuperMobiDick.get_pressure_gradient_areas()
    shear_stress_areas = SuperMobiDick.get_shear_stress_areas() 
    cross_sectional_areas = SuperMobiDick.get_sectional_areas()

    init_gamma = 0

    # Define the ODE system
    def odefun(y):
        nonlocal init_gamma

        z, p, v, h, gamma = y  # position, velocity, density, pressure
        # print(z)

        # Interpolate geometry-dependent parameters
        area = np.interp(
            z, sectional_positions,
            cross_sectional_areas
        )  # Cross-sectional area
        area_slope = (np.interp(
            z + 1e-3, sectional_positions,
            cross_sectional_areas
        ) - area) / 1e-3  # Area derivative
        radius = np.interp(
            z, sectional_positions,
            sectional_radii
        ) 
        diameter = 2 * radius
        perimeter = np.pi * diameter  # Wetted perimeter for circular cross-section

        try:

            state = fluid_rg.set_state(rg.HmassP_INPUTS, h, p) # I use rg instead bpy as I need derivatives for HEM and single phase
            quality = state["Q"]

            # Liquid and vapor saturated conditions
            state_L = fluid_rg.set_state(rg.PQ_INPUTS, p, 0.00)
            state_V = fluid_rg.set_state(rg.PQ_INPUTS, p, 1.00)
            h_L = state_L["h"]
            rho_L = state_L["rho"]
            vol_L = 1 / rho_L
            visc_L = state_L["viscosity"]
            h_V = state_V["h"]
            rho_V = state_V["rho"]
            vol_V = 1 / rho_V
            visc_V = state_V["viscosity"]

            if 0 < quality < 1 and 0 < gamma < 1:

                if init_gamma == 0:
                    gamma = 1e-6
                    init_gamma = 1

                # Steps for forward differences
                dh = max(1e-6 * abs(h), 1e-3 * 5000)
                dp = max(1e-6 * abs(p), 1e-3 * (np.exp(0.028782) - 1.0) * p)
                dgamma = 1e-6 * gamma

                # Get metastable state
                guess_state = fluid_bpy.get_state(bpy.PQ_INPUTS, p, 0.00)
                rhoT_guess = [guess_state["rho"], guess_state["T"]]
                meta = fluid_bpy.get_state_metastable(
                    prop_1 = "s",
                    prop_1_value = s_frozen,
                    prop_2 = "p",
                    prop_2_value = p,
                    rhoT_guess = rhoT_guess,
                    print_convergence=False,
                    solver_algorithm="lm"
                )
                rho_meta = meta["rho"]
                vol_meta = 1 / rho_meta
                h_meta = meta["h"]
                visc_meta = meta["mu"]
                T_meta = meta["T"]
                T_meta = state_L["T"] if T_meta < state_L["T"] else T_meta

                state_sat_T_LM = fluid_bpy.get_state(bpy.fluid_properties.QT_INPUTS, 0.00, T_meta)
                p_sat_T_LM = state_sat_T_LM["p"]
                p_sat_T_LM = p if p_sat_T_LM < p else p_sat_T_LM

                X = (h - (1 - gamma) * h_meta - gamma * h_L) / (h_V - h_L)
                X = 0 if X < 0 else X

                vol = X * vol_V + (gamma-X) * vol_L + (1 - gamma) * vol_meta
                rho = 1 / vol
                x_eq = X / gamma if gamma > X else X
                h_eq = x_eq * h_V + (1 - x_eq) * h_L 

                # Perturbing pressure
                state_V_dp = fluid_bpy.get_state(bpy.PQ_INPUTS, p + dp, 1.00)
                state_L_dp = fluid_bpy.get_state(bpy.PQ_INPUTS, p + dp, 0.00)
                rho_V_dp = state_V_dp["rho"]
                rho_L_dp = state_L_dp["rho"]
                vol_V_dp = 1 / rho_V_dp
                vol_L_dp = 1 / rho_L_dp

                state_dp = fluid_bpy.get_state(bpy.HmassP_INPUTS, h_eq, p + dp)
                x_eq_dp = state_dp["Q"] if gamma > X else x_eq
                x_eq_dp = 0 if x_eq_dp < 0 else x_eq_dp
                X_dp = x_eq_dp * gamma if gamma > X else X

                guess_state_dp = fluid_bpy.get_state(bpy.PQ_INPUTS, p + dp, 0.00)
                rhoT_guess_dp = [guess_state_dp["rho"], guess_state_dp["T"]]
                meta_dp = fluid_bpy.get_state_metastable(
                    prop_1 = "s",
                    prop_1_value = s_frozen,
                    prop_2 = "p",
                    prop_2_value = p + dp,
                    rhoT_guess = rhoT_guess_dp,
                    print_convergence=False,
                    solver_algorithm="lm"
                )
                rho_meta_dp = meta_dp["rho"]
                vol_meta_dp = 1 / rho_meta_dp

                vol_dp = X_dp * vol_V_dp + (gamma-X_dp) * vol_L_dp + (1 - gamma) * vol_meta_dp
                rho_dp = 1 / vol_dp

                # Perturbing enthalpy
                state_dh = fluid_bpy.get_state(bpy.HmassP_INPUTS, h_eq + dh, p)
                x_eq_dh = state_dh["Q"] if gamma > X else x_eq
                X_dh = x_eq_dh * gamma if gamma > X else X

                vol_dh = X_dh * vol_V + (gamma-X_dh) * vol_L + (1 - gamma) * vol_meta
                rho_dh = 1 / vol_dh

                # Perturbing gamma
                gamma_dgamma = gamma + dgamma
                X_dgamma = (h - (1 - gamma_dgamma) * h_meta - gamma_dgamma * h_L) / (h_V - h_L)
                X_dgamma = 0 if X_dgamma < 0 else X_dgamma
                vol_dgamma = X_dgamma * vol_V + (gamma_dgamma-X_dgamma) * vol_L + (1 - gamma_dgamma) * vol_meta
                rho_dgamma = 1 / vol_dgamma

                # Final derivatives
                drho_dP = (rho_dp - rho)/dp
                drho_dh = (rho_dh - rho)/dh
                drhodgamma = (rho_dgamma - rho) / dgamma

                # Calculate viscosity for friction correlation
                vol_frac_V = (X/rho_V) / (X/rho_V + (gamma - X)/rho_L + (1 - gamma)/rho_meta)
                vol_frac_L = ((gamma - X)/rho_L) / (X/rho_V + (gamma - X)/rho_L + (1 - gamma)/rho_meta)
                vol_frac_meta = ((1-gamma)/rho_meta) / (X/rho_V + (gamma - X)/rho_L + (1 - gamma)/rho_meta)
                viscosity_liq_meta = vol_frac_L * visc_L + vol_frac_meta * visc_meta
                density_liq_meta = vol_frac_L * rho_L + vol_frac_meta * rho_meta

                # friction factor
                state_L = fluid_rg.set_state(rg.PQ_INPUTS, p, 0.00)
                state_V = fluid_rg.set_state(rg.PQ_INPUTS, p, 1.00)
                stress_wall, _, _ = get_wall_friction(
                    velocity=v,
                    density=density_liq_meta,
                    viscosity=viscosity_liq_meta,
                    roughness=roughness,
                    diameter=diameter,
                )
                LM_param = richardson(vol_frac_v=vol_frac_V)
                stress_wall = stress_wall * LM_param

                # dem_term_angielczyk_2(p=p, y=y, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, dAdz_div=dAdz_div, dAdz_conv=dAdz_conv), 
                dem_source_term = dem_term_angielczyk_1(perimeter=perimeter, p=p, gamma=gamma, p_sat_T_LM=p_sat_T_LM, p_cr=p_cr, area=area)
                dem_source_term = 1e-3 if dem_source_term < 1e-3 else dem_source_term

                sound_speed = speed_sound_dem(gamma=gamma, quality_hem=x_eq, state_L=state_L, state_V=state_V, rho_mix_dem=rho, rho_meta=rho_meta, c_meta=meta["a"])
            
            else:  # single phase and two-phase HEM
                rho = state["rho"]
                x = state["Q"]
                h = state["h"]
                sound_speed = state["a"]
                drho_dP = state["drho_dP"]
                drho_dh = state["drho_dh"]
                viscosity = state["mu"]
                dem_source_term = 1e-6
                drhodgamma = 1e-6

                if 0 < x < 1 and x is not np.nan:
                    # friction factor
                    state_L = fluid_bpy.get_state(bpy.PQ_INPUTS, p, 0.00)
                    state_V = fluid_bpy.get_state(bpy.PQ_INPUTS, p, 1.00)
                    stress_wall, _, _ = get_wall_friction(
                        velocity=v,
                        density=state_L["rho"],
                        viscosity=state_L["viscosity"],
                        roughness=roughness,
                        diameter=diameter,
                    )
                    vol_frac_v = (x/rho_V) / (x/rho_V + (1 - x)/rho_L)
                    LM_param = richardson(vol_frac_v=vol_frac_v)
                    stress_wall = stress_wall * LM_param

                else:
                    # friction factor
                    stress_wall, _, _ = get_wall_friction(
                        velocity=v,
                        density=rho,
                        viscosity=viscosity,
                        roughness=roughness,
                        diameter=diameter,
                    )


            # Coefficient matrix M for ODE system
            M = np.asarray(
                [
                    [v * drho_dP,    rho,          v * drho_dh,   v * drhodgamma],
                    [1,              rho * v,      0.00,          0             ],
                    [-v,              0.00,        rho * v,       0             ],
                    [0,               0,           0,             1             ],
                ]
            )

            # Right-hand side of the system
            b = np.asarray(
                [
                    -rho * v / area * area_slope,
                    -perimeter / area * stress_wall,
                    v * stress_wall * perimeter / area,
                    dem_source_term,
                ]
            )

            M1 = M.copy()
            M1[:, 0] = b
            M2 = M.copy()
            M2[:, 1] = b
            M3 = M.copy()
            M3[:, 2] = b
            M4 = M.copy()
            M4[:, 3] = b
            
            # Compute determinants
            det_D = det(M)
            det_N1 = det(M1)
            det_N2 = det(M2)
            det_N3 = det(M3)
            det_N4 = det(M4)
            # print("det 1", det_D)
            # print("det 2", det_N1)
            # print("det 3", det_N2)
            # print("det 4", det_N3)
            # print("det 5", det_N4)
            # print(" ")
        
            dy = [det_D, det_N1, det_N2, det_N3, det_N4]

            # Save the output at each step in the dictionary
            out = {
                "distance": z,
                "velocity": v,
                "density": rho,
                "pressure": p,
                "gamma": gamma,
                "quality": quality,
                "speed_of_sound": sound_speed,
                "enthalpy": h,
                # "entropy": state["s"],
                "total_enthalpy": h + 0.5 * v**2,
                "mach_number": v / sound_speed,
                "mass_flow": v * rho * area,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "determinant_D": det_D,
                "determinant_N1": det_N1,
                "determinant_N2": det_N2,
                "determinant_N3": det_N3,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out
    
        except Exception as e:
            # print(f"[ODEFUN ERROR @ z={z:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan, np.nan], None  # forces integrator to stop

    # This function avoid to keep solving for the Impossible Flow when the matrix
    # is singular (backward flow)    
    def stop_at_zero_det(t, y):
        _, out = odefun(y)
        det_M = out["determinant_D"]
        return det_M
    stop_at_zero_det.terminal = True  
    stop_at_zero_det.direction = 0   
    
    # As in this method we are not integrating in the space variable x but for the dummy
    # variable t, it is necessary to stop when the length of the nozzle is achieved
    def stop_at_length(t, y):
        z = y[0]               
        return z - total_length      
    stop_at_length.terminal = True     
    stop_at_length.direction = 1  

    def stop_at_zero(t, y):
        z = y[0]  # assuming y[0] is the nozzle length variable (x)
        return z  
    stop_at_zero.terminal = True
    stop_at_zero.direction = 1 

    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-3
        while error > tol:
            pif_iterations += 1  
            raw_solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(y)[0],
                [1, 0],
                [0.00, pressure_in, u_guess, enthalpy_in, 0.00],
                # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-6,
                atol=1e-6,
                events=[stop_at_length, stop_at_zero_det]
            )
            solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)
            
            if raw_solution.t_events[1].size > 0:
                m_impossible = m_guess
            else:
                m_possible = m_guess

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible
            print(m_impossible)
            print(m_possible)

        flow_rate = u_guess*area_in*density_in


        # Calculate the solution with the last possible flow rate calculated
        u_possible = m_possible/(density_in * area_in)
        possible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(y)[0],
            [1, 0],
            [0.00, pressure_in, u_possible, enthalpy_in, 0.00],
            # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
            events=[stop_at_length, stop_at_zero_det]
        )     
        possible_solution = postprocess_ode_autonomous(possible_solution.t, possible_solution.y, odefun)
        
    # TODO: fix this is case you do not want to run PIF
    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="Radau",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in
 

    return mass_flow, pif_iterations, possible_solution

