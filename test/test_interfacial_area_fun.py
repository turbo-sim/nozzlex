import numpy as np
import matplotlib.pyplot as plt
from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import interfacial_area, interfacial_area_old

X = np.logspace(-8, 0, 2000)
X = np.clip(X, 0, 1)

AREA_NEW, DIAM_NEW = [], []
AREA_OLD, DIAM_OLD = [], []

for x in X:
    area, diameter = interfacial_area(x, Nb=1e12, Nd=1e8)
    AREA_NEW.append(area)
    DIAM_NEW.append(diameter)

    area_old, diam_old = interfacial_area_old(x, Nb=1e12, Nd=1e8)
    AREA_OLD.append(area_old)
    DIAM_OLD.append(diam_old)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# --- Top-left: Ai, linear ---
axs[0, 0].plot(X, AREA_OLD, label="old")
axs[0, 0].plot(X, AREA_NEW, label="new", linestyle="--")
axs[0, 0].set_ylabel("Ai [m²/m³]")
axs[0, 0].set_title("Interfacial area — linear")
axs[0, 0].legend()
axs[0, 0].grid()

# --- Top-right: Ai, log-log ---
axs[0, 1].plot(X, AREA_OLD, label="old")
axs[0, 1].plot(X, AREA_NEW, label="new", linestyle="--")
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
axs[0, 1].set_title("Interfacial area — log-log")
axs[0, 1].legend()
axs[0, 1].grid()

# --- Bottom-left: d, linear ---
axs[1, 0].plot(X, DIAM_OLD, label="old")
axs[1, 0].plot(X, DIAM_NEW, label="new", linestyle="--")
axs[1, 0].set_xlabel("alpha [-]")
axs[1, 0].set_ylabel("d [m]")
axs[1, 0].set_title("Diameter — linear")
axs[1, 0].legend()
axs[1, 0].grid()

# --- Bottom-right: d, log-log ---
axs[1, 1].plot(X, DIAM_OLD, label="old")
axs[1, 1].plot(X, DIAM_NEW, label="new", linestyle="--")
axs[1, 1].set_xscale("log")
axs[1, 1].set_yscale("log")
axs[1, 1].set_xlabel("alpha [-]")
axs[1, 1].set_title("Diameter — log-log")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()
