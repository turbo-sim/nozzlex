import numpy as np
import jaxprop as jxp
import matplotlib.pyplot as plt

jxp.set_plot_options(grid=False)

# def speed_sound_dem(c_hem): 

#     c_dem = np.sqrt(1 / (rho_mix_dem**2 * ((gamma / (rho_mix_hem**2 * c_hem**2)) + (1 - gamma) / (rho_meta**2 * c_meta**2))))

#     return c_dem


fluid = jxp.Fluid("water")
p = 10e6

Q = np.linspace(0.01, 0.99, 100)
gamma = 0.1

a_hem = []
void_frac = []
for i, q in enumerate(Q):

    state = fluid.get_state(jxp.PQ_INPUTS, p, q)

    rhoT_guess = [state["rho"], state["T"]]
    # meta = fluid.get_state_metastable(
    #     prop_1 = "q",
    #     prop_1_value = q,
    #     prop_2 = "p",
    #     prop_2_value = p,
    #     rhoT_guess = rhoT_guess,
    #     print_convergence=False,
    #     # solver_max_iterations=1000,
    #     solver_algorithm="lm"
    # )

    a_hem.append(state["a"])
    void_frac.append(state.quality_volume)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(void_frac, a_hem, "-o", label="HEM")
ax.set_xlabel("Void fraction (-)")
ax.set_ylabel("Speed of sound (m/s)")
plt.tight_layout()
plt.show()
