import pickle
import matplotlib.pyplot as plt
import nozzlex

nozzlex.set_plot_options()

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


path = "solution_22.45.pkl"
sol_1 = load_pickle(path)

path = "solution_22.45_2ht.pkl"
sol_2 = load_pickle(path)

path = "solution_22.45_4ht.pkl"
sol_4 = load_pickle(path)

path = "solution_22.45_e3.pkl"
sol_e3 = load_pickle(path)

plt.plot(sol_1["x"], sol_1["v"], label="1htc")
plt.plot(sol_2["x"], sol_2["v"], label="2htc")
plt.plot(sol_4["x"], sol_4["v"], label="4htc")
plt.plot(sol_e3["x"], sol_e3["v"], label="htcw e-3")

print(sol_4["b_vec"][-1])

plt.ylabel("velocity [m/s]")
plt.xlabel("x [m]")
plt.grid(True)
plt.legend()
# plt.show()
