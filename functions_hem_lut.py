import scipy.linalg
import scipy.integrate
import numpy as np
import pandas as pd
import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
# import perfect_gas_prop.perfect_gas_prop as perfect_gas_prop 
import real_gas_prop.real_gas_prop as rg
from cycler import cycler
import numpy as np
from scipy.linalg import det

COLORS_PYTHON = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

COLORS_MATLAB = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


try:
    import IPython.core.ultratb
except ImportError:
    # No IPython. Use default exception printing.
    pass
else:
    import sys

    sys.excepthook = IPython.core.ultratb.ColorTB()

def load_LUT(filename):
    folder = "LuT"  # folder name
    filepath = os.path.join(folder, filename)  # build full path
    
    df = pd.read_csv(filepath)
    
    h_vals = np.sort(df["h [J/kg]"].unique())
    P_vals = np.sort(df["P [Pa]"].unique())
    
    def reshape_quantity(col):
        return df.pivot(index="h [J/kg]", columns="P [Pa]", values=col).values
    
    lut = {
        "h_vals": h_vals,
        "P_vals": P_vals,
        "rho": reshape_quantity("rho [kg/m³]"),
        "drho_dP": reshape_quantity("drho_dP [kg/m³/Pa]"),
        "drho_dh": reshape_quantity("drho_dh [kg²/(m³·J)]"),
        # Add more quantities as needed
    }
    return lut

def build_interpolators(lut):
    interpolators = {}
    h_vals, P_vals = lut["h_vals"], lut["P_vals"]
    
    for key in lut:
        if key in ["h_vals", "P_vals"]:
            continue
        interpolators[key] = RegularGridInterpolator(
            (h_vals, P_vals), lut[key], bounds_error=False, fill_value=None
        )
    return interpolators

def query_properties(interpolators, h, P):
    point = np.array([[h, P]])
    results = {}
    for key, interp in interpolators.items():
        results[key] = interp(point)[0]
    return results


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


def get_linear_convergent_divergent(length, convergent_length, divergent_length, radius_in, radius_throat, radius_out, width, type):
    
    # if type == "Axisymmetric":
    #     area_in = np.pi*radius_in**2
    #     area_throat = np.pi*radius_throat**2
    #     area_out = np.pi*radius_out**2
        
    #     if length <= convergent_length:
    #         radius_slope = (radius_throat-radius_in)/convergent_length
    #         radius = radius_in + radius_slope*length
    #         area = np.pi*radius**2
    #     elif length > convergent_length:
    #         radius_slope = (radius_out-radius_in)/divergent_length
    #         radius = radius_in + radius_slope*(length-convergent_length)
    #         area = np.pi*radius**2

    #     radius = np.sqrt(area/np.pi)
    #     perimeter = np.pi*radius*2

    if type == "Planar":
        area_in = 2*radius_in*width
        area_throat = 2*radius_throat*width
        area_out = 2*radius_out*width

        if length <= convergent_length:
            area_slope = (area_throat-area_in)/convergent_length
            area = area_in + area_slope*length

        elif length > convergent_length:
            area_slope = (area_out-area_throat)/divergent_length
            area = area_throat + area_slope*(length-convergent_length)

        radius = area/(2*width)
        perimeter = 2*width+2*2*radius

    return area, area_slope, perimeter, radius


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

def pipeline_steady_state_1D(
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
    
    fluid = rg.Fluid(fluid_name, backend="HEOS", exceptions=True)
    
    # Define inlet area and length of the nozzle
    total_length = convergent_length+divergent_length
    if nozzle_type == "Planar":
        area_in = 2*radius_in*width
    elif nozzle_type == "Axisymmetric":
        area_in = np.pi*radius_in**2

    # Calculate inlet density
    density_in = properties_in.rho

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
        m_guess = (m_impossible+m_possible) / 2
        u_guess = m_guess / (density_in * area_in)

    # Initialize out_list to store all the state outputs
    out_list = []
    lut = load_LUT("LUT.csv")
    interpolators = build_interpolators(lut)
    
    # Define the ODE system
    def odefun(t, y):
        x = t  # distance
        p, v, h = y  # velocity, density, pressure
        
        try:

            point = np.array([[h, p]])  # Note: must be 2D array for interpolator input
            rho = interpolators["rho"](point)[0]
            drho_dP = interpolators["drho_dP"](point)[0]
            drho_dh = interpolators["drho_dh"](point)[0]
            viscosity = interpolators["viscosity"](point)[0]


            # # Calculate area and geometry properties for convergent nozzles only
            # area, area_slope, perimeter, diameter = get_geometry(
            #     length=x, total_length=length, area_in=area_in, area_ratio=area_ratio
            # )

            # Calculate area and geometry properties for convergent-divergent nozzles
            area, area_slope, perimeter, radius = get_linear_convergent_divergent(
                length=x, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
                radius_out=radius_out, width=width, type=nozzle_type)
            diameter = radius*2

            # Wall friction calculations
            stress_wall, friction_factor, reynolds = get_wall_friction(
                velocity=v,
                density=rho,
                viscosity=viscosity,
                roughness=roughness,
                diameter=diameter,
            )
            if not include_friction:
                stress_wall = 0.0
                friction_factor = 0.0

            # Heat transfer (if applicable)
            if include_heat_transfer:
                U = 10 # To include correlations 
            else:
                U = 0.0
                heat_in = 0

            # Coefficient matrix M for ODE system
            M = np.asarray(
                [
                    [v*drho_dP, rho, v*drho_dh],
                    [1, rho * v, 0.00],
                    [-v, 0.00, rho*v],
                ]
            )

            determinant = det(M)

            # If singularity detected
            flag = 0
            # if (1-1e-3) <= v/state["a"] <= (1+1e-3):
            #     flag = 1
            if abs(determinant) < 2:
                flag = 1

            # Right-hand side of the system
            # G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
            G = 0
            b = np.asarray(
                [
                    -rho * v / area * area_slope,
                    -perimeter / area * stress_wall,
                    perimeter / area * G / v * (stress_wall * v + heat_in),
                ]
            )

            # Solve the system to get the change in state
            dy = scipy.linalg.solve(M, b)

            # Save the output at each step in the dictionary
            out = {
                "distance": x,
                "velocity": v,
                "density": rho,
                "pressure": p,
                "temperature": interpolators["temperature"](point)[0],
                # "speed_of_sound": state["a"],
                "viscosity": interpolators["viscosity"](point)[0],
                "enthalpy": interpolators["enthalpy"](point)[0],
                "entropy": interpolators["entropy"](point)[0],
                "total_enthalpy": interpolators["enthalpy"](point)[0] + 0.5 * v**2,
                # "mach_number": v / state["a"],
                "mass_flow": v * rho * area,
                "area": area,
                "area_slope": area_slope,
                "perimeter": perimeter,
                "diameter": diameter,
                "stress_wall": stress_wall,
                "friction_factor": friction_factor,
                "reynolds": reynolds,
                "source_1": b[0],
                "source_2": b[1],
                "source_3": b[2],
                "determinant": determinant,
            }

            # Append the output dictionary to out_list
            out_list.append(out)

            return dy, out, flag
        
        except Exception as e:
            print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
            return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop
        
    
    # Possible-impossible flow (PIF) algorithm to find critical flow rate
    if critical_flow is True: 
        pif_iterations = 0
        print("Possible-Impossible Flow (PIF) algorithm starts...")
        error = abs(m_impossible-m_possible)/m_possible
        tol = 1e-3
        while error > tol:
            pif_iterations += 1  
            solution = scipy.integrate.solve_ivp(
                lambda t, y: odefun(t, y)[0],
                [0.0, total_length],
                [u_guess, density_in, pressure_in],
                t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
                method="RK45",
                rtol=1e-9,
                atol=1e-9,
            )
            solution = postprocess_ode(solution.t, solution.y, odefun)

            if solution["singularity_detected"] == True:
                m_impossible = m_guess
            else:
                m_possible = m_guess

            m_guess = (m_impossible+m_possible) / 2
            u_guess = m_guess / (density_in * area_in)  
            error = abs(m_impossible-m_possible)/m_possible

        flow_rate = u_guess*area_in*density_in

        # Calculate the solution with the last possible flow rate calculated
        u_possible = m_possible/(density_in * area_in)
        possible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, (total_length)],
            [u_possible, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        )     
        possible_solution = postprocess_ode(possible_solution.t, possible_solution.y, odefun)

        # Calculate the solution with the last impossible flow rate calculated
        u_impossible = m_impossible/(density_in * area_in)
        impossible_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, (total_length)],
            [u_impossible, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        impossible_solution = postprocess_ode(impossible_solution.t, impossible_solution.y, odefun) 

        # Calculate the solution with u_guess
        u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
        solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [0.0, (total_length)],
            [u_avg, density_in, pressure_in],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        solution = postprocess_ode(solution.t, solution.y, odefun) 
        mass_flow = impossible_solution["mass_flow"][0]
        density = impossible_solution["density"][-1]
        x = convergent_length*1.00001
        area, area_slope, perimeter, radius = get_linear_convergent_divergent(x, 
                                                                              convergent_length=convergent_length,
                                                                              divergent_length=divergent_length,
                                                                              radius_in=radius_in,
                                                                              radius_out=radius_out,
                                                                              radius_throat=radius_throat,
                                                                              width=width,
                                                                              type=nozzle_type)
        velocity = mass_flow/(density*area)

        # Calculate the supersonic branch
        supersonic_solution = scipy.integrate.solve_ivp(
            lambda t, y: odefun(t, y)[0],
            [x, (total_length)],
            [velocity, 
             impossible_solution["density"][-1], 
             impossible_solution["pressure"][-1]],
            t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        ) 
        supersonic_solution = postprocess_ode(supersonic_solution.t, supersonic_solution.y, odefun) 


    else:
        pif_iterations = None
        solution = scipy.integrate.solve_ivp(
        lambda t, y: odefun(t, y)[0],
        [0.0, total_length],
        [velocity_in, density_in, pressure_in],
        t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
        method="RK45",
        rtol=1e-9,
        atol=1e-9,
        )
        solution = postprocess_ode(solution.t, solution.y, odefun)

        flow_rate = velocity_in*density_in*area_in
 

    return supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations



# # Autonomous solver 
# def pipeline_steady_state_1D_autonomous(
#     fluid_name,
#     pressure_in,
#     temperature_in,
#     properties_in,
#     convergent_length,
#     divergent_length,
#     roughness,
#     radius_in,
#     radius_throat,
#     radius_out,
#     nozzle_type,
#     width,
#     mass_flow=None,
#     mach_in=None,
#     critical_flow = False,
#     include_friction=True,
#     include_heat_transfer=False,
#     temperature_external=None,
#     number_of_points=None,
# ):
#     # Check for correct inputs
#     if (
#         (mass_flow is None and mach_in is None and critical_flow is False) or 
#         (mass_flow is not None and mach_in is not None) 
#     ):
#         raise ValueError(
#             "Check input settins for the velocity."
#         )
    
#     fluid = rg.Fluid(fluid_name, backend="HEOS", exceptions=True)
    
#     # Define inlet area and length of the nozzle
#     total_length = convergent_length+divergent_length
#     if nozzle_type == "Planar":
#         area_in = 2*radius_in*width
#     elif nozzle_type == "Axisymmetric":
#         area_in = np.pi*radius_in**2

#     # Calculate inlet density
#     density_in = properties_in.rho

#     # Calculate velocity based on specified parameter
#     if mass_flow is not None:
#         velocity_in = mass_flow / (area_in * density_in)
#     elif mach_in is not None:
#         velocity_in = mach_in * properties_in["a"]
#     elif critical_flow is True:
#         mach_impossible = 0.02
#         mach_possible = 0.001
#         u_impossible = mach_impossible*properties_in.a
#         u_possible = mach_possible*properties_in.a
#         m_impossible = density_in*u_impossible*area_in
#         m_possible = density_in*u_possible*area_in
#         # m_possible = 0.02535
#         # m_impossible = 0.02535
#         m_guess = (m_impossible+m_possible) / 2
#         u_guess = m_guess / (density_in * area_in)

#     # Initialize out_list to store all the state outputs
#     out_list = []
    
#     # Define the ODE system
#     def odefun(y):
#         x, v, rho, p = y  # position, velocity, density, pressure
        
#         try:
#             # Thermodynamic state from perfect gas properties
#             state = fluid.set_state(DmassP_INPUTS, rho, p)

#             # Calculate area and geometry properties for convergent-divergent nozzles
#             area, area_slope, perimeter, radius = get_linear_convergent_divergent(
#                 length=x, convergent_length=convergent_length, divergent_length=divergent_length, radius_in=radius_in, radius_throat=radius_throat,
#                 radius_out=radius_out, width=width, type=nozzle_type)
#             diameter = radius*2

#             # Wall friction calculations
#             stress_wall, friction_factor, reynolds = get_wall_friction(
#                 velocity=v,
#                 density=rho,
#                 viscosity=state["mu"],
#                 roughness=roughness,
#                 diameter=diameter,
#             )
#             if not include_friction:
#                 stress_wall = 0.0
#                 friction_factor = 0.0

#             # Heat transfer (if applicable)
#             if include_heat_transfer:
#                 U = 10 # To include correlations 
#             else:
#                 U = 0.0
#                 heat_in = 0

#             # Coefficient matrix M for ODE system
#             M = np.asarray(
#                 [
#                     [rho, v, 0.0],
#                     [rho * v, 0.0, 1.0],
#                     [0.0, -state["a"]**2, 1.0],
#                 ]
#             )

#             # Right-hand side of the system
#             # G = state.isobaric_expansion_coefficient * state.a**2 / state.cp
#             G = 0
#             b = np.asarray(
#                 [
#                     -rho * v / area * area_slope,
#                     -perimeter / area * stress_wall,
#                     perimeter / area * G / v * (stress_wall * v + heat_in),
#                 ]
#             )

#             M1 = M.copy()
#             M1[:, 0] = b
#             M2 = M.copy()
#             M2[:, 1] = b
#             M3 = M.copy()
#             M3[:, 2] = b
            
#             # Compute determinants
#             det_D = det(M)
#             det_N1 = det(M1)
#             det_N2 = det(M2)
#             det_N3 = det(M3)
#             # print("det 1", det_M)
#             # print("det 2", det_M1)
#             # print("det 3", det_M2)
#             # print("det 4", det_M3)
#             # print(" ")
        
#             dy = [det_D, det_N1, det_N2, det_N3]

#             # Save the output at each step in the dictionary
#             out = {
#                 "distance": x,
#                 "velocity": v,
#                 "density": rho,
#                 "pressure": p,
#                 "temperature": state["T"],
#                 "speed_of_sound": state["a"],
#                 "viscosity": state["mu"],
#                 "enthalpy": state["h"],
#                 "entropy": state["s"],
#                 "total_enthalpy": state["h"] + 0.5 * v**2,
#                 "mach_number": v / state["a"],
#                 "mass_flow": v * rho * area,
#                 "area": area,
#                 "area_slope": area_slope,
#                 "perimeter": perimeter,
#                 "diameter": diameter,
#                 "stress_wall": stress_wall,
#                 "friction_factor": friction_factor,
#                 "reynolds": reynolds,
#                 "source_1": b[0],
#                 "source_2": b[1],
#                 "determinant_D": det_D,
#                 "determinant_N1": det_N1,
#                 "determinant_N2": det_N2,
#                 "determinant_N3": det_N3,
#             }

#             # Append the output dictionary to out_list
#             out_list.append(out)

#             return dy, out
        
#         except Exception as e:
#             # print(f"[ODEFUN ERROR @ x={x:.4f}] {e}")
#             return [np.nan, np.nan, np.nan, np.nan]  # forces integrator to stop

#     # This function avoid to keep solving for the Impossible Flow when the matrix
#     # is singular (backward flow)    
#     def stop_at_zero_det(t, y):
#         _, out = odefun(y)
#         det_M = out["determinant_D"]
#         return det_M

#     stop_at_zero_det.terminal = True  # Stop when det(M) = 0
#     stop_at_zero_det.direction = 0   # Trigger event when det(M) == 0, no direction preference
    
#     # As in this method we are not integrating in the space variable x but for the dummy
#     # variable t, it is necessary to stop when the length of the nozzle is achieved
#     def stop_at_length(t, y):
#         x = y[0]               
#         return x - total_length      
#     stop_at_length.terminal = True     
#     stop_at_length.direction = 1  

#     def stop_at_zero(t, y):
#         x = y[0]  # assuming y[0] is the nozzle length variable (x)
#         return x  

#     stop_at_zero.terminal = True
#     stop_at_zero.direction = -1 

#     # Possible-impossible flow (PIF) algorithm to find critical flow rate
#     if critical_flow is True: 
#         pif_iterations = 0
#         print("Possible-Impossible Flow (PIF) algorithm starts...")
#         error = abs(m_impossible-m_possible)/m_possible
#         tol = 1e-2
#         while error > tol:
#             pif_iterations += 1  
#             raw_solution = scipy.integrate.solve_ivp(
#                 lambda t, y: odefun(y)[0],
#                 [0, 1],
#                 [0, u_guess, density_in, pressure_in],
#                 t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
#                 method="Radau",
#                 rtol=1e-9,
#                 atol=1e-9,
#                 events=[stop_at_length, stop_at_zero_det]
#             )
#             solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)
            
#             if raw_solution.t_events[1].size > 0:
#                 m_impossible = m_guess
#             else:
#                 m_possible = m_guess

#             m_guess = (m_impossible+m_possible) / 2
#             u_guess = m_guess / (density_in * area_in)  
#             error = abs(m_impossible-m_possible)/m_possible
#             print(m_impossible)
#             print(m_possible)

#         flow_rate = u_guess*area_in*density_in


#         # Calculate the solution with the last possible flow rate calculated
#         u_possible = m_possible/(density_in * area_in)
#         possible_solution = scipy.integrate.solve_ivp(
#             lambda t, y: odefun(y)[0],
#             [0, 1],
#             [0, u_possible, density_in, pressure_in],
#             t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
#             method="Radau",
#             rtol=1e-9,
#             atol=1e-9,
#             events=[stop_at_length, stop_at_zero_det]
#         )     
#         possible_solution = postprocess_ode_autonomous(possible_solution.t, possible_solution.y, odefun)
        
#         # Calculate the solution with the last impossible flow rate calculated
#         u_impossible = m_impossible/(density_in * area_in)
#         impossible_solution = scipy.integrate.solve_ivp(
#             lambda t, y: odefun(y)[0],
#             [0, 1],
#             [0, u_impossible, density_in, pressure_in],
#             t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
#             method="RK45",
#             rtol=1e-9,
#             atol=1e-9,
#             events=[stop_at_length, stop_at_zero_det]
#         ) 
#         impossible_solution = postprocess_ode_autonomous(impossible_solution.t, impossible_solution.y, odefun) 

#         # Calculate the solution with u_guess
#         u_avg = (m_possible+m_impossible)/(2*density_in * area_in)
#         solution = scipy.integrate.solve_ivp(
#             lambda t, y: odefun(y)[0],
#             [0, 1],
#             [0, u_avg, density_in, pressure_in],
#             t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
#             method="Radau",
#             rtol=1e-9,
#             atol=1e-9,
#             events=[stop_at_length, stop_at_zero_det]
#         ) 
#         solution = postprocess_ode_autonomous(solution.t, solution.y, odefun) 

#         mass_flow = solution["mass_flow"][0]
#         density = solution["density"][-1]
#         x = convergent_length*1.00001
#         area, _, _, _ = get_linear_convergent_divergent(x, 
#                                                         convergent_length=convergent_length,
#                                                         divergent_length=divergent_length,
#                                                         radius_in=radius_in,
#                                                         radius_out=radius_out,
#                                                         radius_throat=radius_throat,
#                                                         width=width,
#                                                         type=nozzle_type)
#         velocity = mass_flow/(density*area)

#         supersonic_solution = scipy.integrate.solve_ivp(
#             lambda t, y: odefun(y)[0],
#             [1, 0], # Inverting dummy variable limits so you so not go backward at the singularity (all the determinants<0)
#             [x,
#              velocity, 
#              solution["density"][-1], 
#              solution["pressure"][-1]],
#             t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
#             method="RK45",
#             rtol=1e-9,
#             atol=1e-9,
#             events=[stop_at_length, stop_at_zero]
#         ) 
#         supersonic_solution = postprocess_ode_autonomous(supersonic_solution.t, supersonic_solution.y, odefun) 
#         print("rate out", supersonic_solution["mass_flow"][-1])
#     # To fix:
#     else:
#         pif_iterations = None
#         solution = scipy.integrate.solve_ivp(
#         lambda t, y: odefun(t, y)[0],
#         [0.0, total_length],
#         [velocity_in, density_in, pressure_in],
#         t_eval=np.linspace(0, total_length, number_of_points) if number_of_points else None,
#         method="Radau",
#         rtol=1e-9,
#         atol=1e-9,
#         )
#         solution = postprocess_ode(solution.t, solution.y, odefun)

#         flow_rate = velocity_in*density_in*area_in
 

#     return supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations

