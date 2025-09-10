import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.interpolate import NearestNDInterpolator
import barotropy as bpy
import pandas as pd

def fill_nan_nearest_2d(array):
    x, y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
    xys = np.vstack((x[~np.isnan(array)], y[~np.isnan(array)])).T
    values = array[~np.isnan(array)]
    interp = NearestNDInterpolator(xys, values)
    filled = interp(x, y)
    return filled

# === SETUP ===
fluid_name = 'CO2'
fluid = bpy.Fluid(name=fluid_name, backend="HEOS")

# Grid definition
P_min, P_max = 1e6, 10e6  # Pa
h_min, h_max = 1.5e5, 6e5  # J/kg
nP, nh = 500, 500
P_vals = np.linspace(P_min, P_max, nP)
h_vals = np.linspace(h_min, h_max, nh)

# Arrays 
rho = np.full((nh, nP), np.nan)
Q = np.full((nh, nP), np.nan)
T = np.full((nh, nP), np.nan)          # Temperature [K]
viscosity = np.full((nh, nP), np.nan)  # Viscosity [Pa.s]
entropy = np.full((nh, nP), np.nan)    # Entropy [J/kg-K]

# === FILL LUT ===
for i, h in enumerate(h_vals):
    for j, P in enumerate(P_vals):
        try:
            state = bpy.compute_properties_coolprop(fluid._AS, bpy.HmassP_INPUTS, h, P)
            Q[i, j] = state["Q"]
            rho[i, j] = state["rho"]
            T[i, j] = state["T"]                   # Save temperature
            viscosity[i, j] = state["viscosity"]   # Save viscosity
            entropy[i, j] = state["s"]             # Save entropy
        except Exception:
            continue

# === FINITE DIFFERENCE DERIVATIVES ===
drho_dP = np.full_like(rho, np.nan)
drho_dh = np.full_like(rho, np.nan)
deltaP = 1e4   # Pa
deltah = 1e2   # J/kg

rho_interp_linear = RegularGridInterpolator((h_vals, P_vals), rho, bounds_error=False, fill_value=np.nan)
rho_filled = fill_nan_nearest_2d(rho)
spline_rho = RectBivariateSpline(h_vals, P_vals, rho_filled)

for i, h in enumerate(h_vals):
    for j, P in enumerate(P_vals):
        q = Q[i, j]
        if not np.isfinite(q) or (0 < q < 1):  # Skip two-phase
            continue
        try:
            rho_pP = spline_rho.ev(h, P + deltaP)
            rho_mP = spline_rho.ev(h, P - deltaP)
            rho_ph = spline_rho.ev(h + deltah, P)
            rho_mh = spline_rho.ev(h - deltah, P)

            if np.isfinite(rho_pP) and np.isfinite(rho_mP):
                drho_dP[i, j] = (rho_pP - rho_mP) / (2 * deltaP)

            if np.isfinite(rho_ph) and np.isfinite(rho_mh):
                drho_dh[i, j] = (rho_ph - rho_mh) / (2 * deltah)

        except:
            continue

drho_dP_filled = fill_nan_nearest_2d(drho_dP)
drho_dh_filled = fill_nan_nearest_2d(drho_dh)

interp_drho_dP = RegularGridInterpolator((h_vals, P_vals), drho_dP_filled, bounds_error=False, fill_value=np.nan)
interp_drho_dh = RegularGridInterpolator((h_vals, P_vals), drho_dh_filled, bounds_error=False, fill_value=np.nan)
spline_drho_dP = RectBivariateSpline(h_vals, P_vals, drho_dP_filled)
spline_drho_dh = RectBivariateSpline(h_vals, P_vals, drho_dh_filled)

# === CHECK 5 SINGLE-PHASE POINTS ===
print("CHECKING 5 SINGLE-PHASE POINTS")
print("=" * 40)
np.random.seed(0)
count = 0

while count < 5:
    h_rand = np.random.uniform(h_min + 5000, h_max - 5000)
    P_rand = np.random.uniform(P_min + 1e5, P_max - 1e5)
    try:
        Q_val = PropsSI("Q", "H", h_rand, "P", P_rand, fluid_name)
        if 0 < Q_val < 1:
            continue  # Skip two-phase

        # Get CoolProp values
        rho_cp = PropsSI("D", "H", h_rand, "P", P_rand, fluid_name)
        drho_dP_cp = PropsSI("d(D)/d(P)|H", "H", h_rand, "P", P_rand, fluid_name)
        drho_dh_cp = PropsSI("d(D)/d(H)|P", "H", h_rand, "P", P_rand, fluid_name)

        # Finite difference linear interpolation (RegularGridInterpolator)
        rho_fd = rho_interp_linear([[h_rand, P_rand]])[0]
        drho_dP_fd = interp_drho_dP([[h_rand, P_rand]])[0]
        drho_dh_fd = interp_drho_dh([[h_rand, P_rand]])[0]

        # Bicubic spline interpolation for density and derivatives
        rho_spline = spline_rho.ev(h_rand, P_rand)
        drho_dP_spline = spline_drho_dP.ev(h_rand, P_rand)
        drho_dh_spline = spline_drho_dh.ev(h_rand, P_rand)

        print(f"\nPoint {count+1}: h = {h_rand:.2e} J/kg, P = {P_rand:.2e} Pa")
        print(f"Density (CoolProp)         : {rho_cp:.3f} kg/m³")
        print(f"Density (Linear Interp)    : {rho_fd:.3f} kg/m³")
        print(f"Density (Bicubic)          : {rho_spline:.3f} kg/m³")

        print(f"d(rho)/dP|h (CoolProp)     : {drho_dP_cp:.4e} kg/(m³·Pa)")
        print(f"d(rho)/dP|h (Linear Interp): {drho_dP_fd:.4e} → Err = {100*abs(drho_dP_fd - drho_dP_cp)/abs(drho_dP_cp):.2f}%")
        print(f"d(rho)/dP|h (Bicubic)      : {drho_dP_spline:.4e} → Err = {100*abs(drho_dP_spline - drho_dP_cp)/abs(drho_dP_cp):.2f}%")

        print(f"d(rho)/dh|P (CoolProp)     : {drho_dh_cp:.4e} kg²/(m³·J)")
        print(f"d(rho)/dh|P (Linear Interp): {drho_dh_fd:.4e} → Err = {100*abs(drho_dh_fd - drho_dh_cp)/abs(drho_dh_cp):.2f}%")
        print(f"d(rho)/dh|P (Bicubic)      : {drho_dh_spline:.4e} → Err = {100*abs(drho_dh_spline - drho_dh_cp)/abs(drho_dh_cp):.2f}%")

        count += 1

    except Exception as e:
        # Optionally print(e) for debugging
        continue

# === OPTIONAL PLOT (Density Field & Saturation Dome) ===
T_vals = np.linspace(PropsSI("Ttriple", fluid_name) + 1, PropsSI("Tcrit", fluid_name) - 1, 500)
P_sat = [PropsSI("P", "T", T, "Q", 0, fluid_name) for T in T_vals]
h_f = [PropsSI("H", "T", T, "Q", 0, fluid_name) for T in T_vals]
h_g = [PropsSI("H", "T", T, "Q", 1, fluid_name) for T in T_vals]

plt.figure(figsize=(6, 5))
cp = plt.contourf(h_vals / 1e6, P_vals / 1e6, rho.T / 1000, levels=50, cmap='viridis')
plt.plot(np.array(h_f) / 1e6, np.array(P_sat) / 1e6, 'w--', lw=2, label='Sat. Liquid')
plt.plot(np.array(h_g) / 1e6, np.array(P_sat) / 1e6, 'w--', lw=2, label='Sat. Vapor')
plt.colorbar(label='Density [kg/m³]')
plt.xlabel('Enthalpy [MJ/kg]')
plt.ylabel('Pressure [MPa]')
plt.title('CO₂ Density Map with Saturation Dome')
plt.legend()
plt.tight_layout()
plt.show()

## Save table
data = {
    "enthalpy": [],
    "pressure": [],
    "density": [],
    "drho_dP": [],
    "drho_dh": [],
    "rho_linear_interp": [],
    "rho_bicubic_interp": [],
    "temperature": [],     
    "quality": [],              
    "viscosity": [],
    "entropy": [],             # Add entropy here
}

for i, h in enumerate(h_vals):
    for j, P in enumerate(P_vals):
        if not np.isfinite(Q[i, j]) or (0 < Q[i, j] < 1):  # Skip two-phase
            continue

        data["enthalpy"].append(h)
        data["pressure"].append(P)
        data["density"].append(rho[i, j])
        data["drho_dP"].append(drho_dP_filled[i, j])
        data["drho_dh"].append(drho_dh_filled[i, j])
        data["rho_linear_interp"].append(rho_interp_linear([[h, P]])[0])
        data["rho_bicubic_interp"].append(spline_rho.ev(h, P))
        data["temperature"].append(T[i, j])
        data["quality"].append(Q[i, j])
        data["viscosity"].append(viscosity[i, j])
        data["entropy"].append(entropy[i, j])  # Append entropy

# Save to CSV
df_all = pd.DataFrame(data)
df_all.to_csv("LUT.csv", index=False, float_format="%.6f")
print("=" * 40)
print("Saved full combined table to 'LUT.csv'")