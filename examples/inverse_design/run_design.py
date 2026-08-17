from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import jaxprop as jxp
import pandas as pd
import yaml
from pathlib import Path
from nozzlex.inverse_design.nurbs_creator import nurbs_curve_points
from nozzlex.inverse_design.functions import march_from_p_ode, march_from_p_isentropic
from nozzlex.inverse_design.plotting import plot_design_overview

jxp.set_plot_options(grid=False)

config_path = Path(__file__).with_name('settings.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

fluid_name = config['fluid_name']
backend = config.get('backend', 'HEOS')
fluid = jxp.Fluid(fluid_name, backend=backend)
u_in_default = config.get('u_in')
inlet_opening_default = config.get('inlet_opening')
width_default = config.get('width')
isentropic_default = config.get('isentropic', True)

cases_params = [
    (c['p0_total'], c['Q0'], c['p_out'],
     c.get('inlet_opening', inlet_opening_default),
     c.get('width', width_default),
     c.get('u_in', u_in_default),
     c.get('isentropic', isentropic_default))
    for c in config['cases']
]
case_labels = [c.get('label', f"Case {i+1}") for i, c in enumerate(config['cases'])]

results_dir = Path(__file__).parent / 'out'
results_dir.mkdir(parents=True, exist_ok=True)

all_geometry_data = []
plot_results = []

for i, (p0_total, Q0, p_out, inlet_opening, width, u_in, isentropic) in enumerate(cases_params):
    case_label = case_labels[i]
    print(f"Processing {case_label}...")

    # --- A. Thermodynamics & Inlet Setup ---
    A_in = width * inlet_opening
    state_0 = fluid.get_state(jxp.PQ_INPUTS, p0_total, Q0)
    h_in = state_0.h - 0.5 * u_in**2
    state_in = fluid.get_state(jxp.HmassSmass_INPUTS, h_in, state_0.s)
    mdot = A_in * u_in * state_in.rho
    inlet_params = {'u_in': u_in, 'h_in': h_in, 'A_in': A_in, 'h0': state_0.h, 'mdot': mdot}

    # --- B. Pressure Profile Generation (NURBS) ---
    x_nodes = np.linspace(0.0, 1.0, 100)
    p_mid = 0.5 * (p0_total + p_out)
    ctrl_pts = [(0.0, state_in.p), (0.5, state_in.p), (0.5, p_mid), (0.5, p_out), (1.0, p_out)]
    nurbs_pts = nurbs_curve_points(ctrl_pts, [1.0] * len(ctrl_pts), degree=4, num=400)
    p_profile = np.interp(x_nodes, nurbs_pts[:, 0], nurbs_pts[:, 1])

    # --- C. Run 1D Solver ---
    out = march_from_p_ode(x_nodes, p_profile, inlet_params, fluid_name=fluid_name, backend=backend)
    res_x = out['x']
    res_p = np.interp(res_x, x_nodes, p_profile)

    # Extract Properties
    rho, a, Q_plot, T, s, mach = [], [], [], [], [], []
    for h_val, p_val, u_val in zip(out['h'], res_p, out['u']):
        try:
            st = fluid.get_state(jxp.HmassP_INPUTS, h_val, p_val)
            rho.append(st.rho); a.append(st.a); T.append(st.T); s.append(st.s); mach.append(u_val / st.a)
            Q_plot.append(st.Q if (0.0 < st.Q <= 1.0) else np.nan)
        except:
            break

    v_len = len(T)
    curr_x_norm = res_x[:v_len] # Normalized axial coordinate (0 to 1)

    # --- 3. Area Ratio (A/A*), dimensionless ---
    A_throat_actual = out['A'][:v_len].min()
    area_ratio = out['A'][:v_len] / A_throat_actual

    # --- Isentropic cross-check (closed-form, frictionless), optional per case ---
    if isentropic:
        iso = march_from_p_isentropic(x_nodes, p_profile, h_in, state_0.s, A_in, u_in,
                                      fluid=fluid_name, backend=backend)
        iso_v_len = np.sum(~np.isnan(iso['A']))
        iso_x_norm = iso['x'][:iso_v_len]
        iso_area_ratio = iso['A'][:iso_v_len] / iso['A'][:iso_v_len].min()

    # --- 4. Store Numerical Results ---
    df_case = pd.DataFrame({
        'case': i+1,
        'x_norm': curr_x_norm,      # Non-dimensional length (0 to 1)
        'area_ratio': area_ratio,    # A/A*
        'mach': mach,
        'p_pa': res_p[:v_len],
        'T_K': T,
        's_JkgK': s,
        'Q': Q_plot
    })
    all_geometry_data.append(df_case)

    # --- 5. Collect Results for the Overview Plot ---
    case_result = {
        'x_norm': curr_x_norm,
        'p': res_p[:v_len],
        'Q': Q_plot,
        'area_ratio': area_ratio,
        'u': out['u'][:v_len],
        'mach': mach,
        'rho': rho,
        's': s,
        'T': T,
    }
    if isentropic:
        case_result.update({
            'iso_x_norm': iso_x_norm,
            'iso_p': iso['p'][:iso_v_len],
            'iso_Q': iso['Q'][:iso_v_len],
            'iso_area_ratio': iso_area_ratio,
            'iso_u': iso['u'][:iso_v_len],
            'iso_mach': iso['M'][:iso_v_len],
            'iso_rho': iso['rho'][:iso_v_len],
        })
    plot_results.append(case_result)

fig, axes, fig_ts, ax_ts = plot_design_overview(fluid, plot_results, case_labels)

# Export Combined Data
pd.concat(all_geometry_data).to_csv(results_dir / 'nozzle_data.csv', index=False)
print(f"Success! Data saved with A/A* ratios to {results_dir / 'nozzle_data.csv'}")

# Save Figures
fig.savefig(results_dir / 'nozzle_overview.png', dpi=200, bbox_inches='tight')
fig_ts.savefig(results_dir / 'nozzle_Ts_diagram.png', dpi=200, bbox_inches='tight')
print(f"Figures saved to {results_dir}")

plt.show()
