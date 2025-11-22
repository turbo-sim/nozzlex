import numpy as np
import matplotlib.pyplot as plt
# from nozzle_model_core import get_nozzle_elliot, interfacial_area, symmetric_nozzle_geometry
from test_elliot_bspline import get_nozzle_elliot

X = np.linspace(0, 0.320, 200)
AREA = []
RADIUS = []
PERIMETER = []
dAdx = []

for x in X:
    area, area_slope, perimeter, radius = get_nozzle_elliot(x,)
    AREA.append(area)
    RADIUS.append(radius)
    PERIMETER.append(perimeter)
    dAdx.append(area_slope)


# plt.plot(X, AREA, label="area")
# plt.plot(X, dAdx, label="dAdx")
# plt.plot([0.1192587, 0.1192587], [-0.05, 0.05])
plt.plot(X, RADIUS, label="radius")
# plt.plot(X, PERIMETER, label="perimeter")
# plt.plot(X, PERIMETER, label="perimeter")
plt.legend()
# plt.ylim([0.0, 0.026])
plt.grid()

plt.show()


# alpha = np.linspace(0,1,100)
# Ai = []
# for a in alpha:
#     _, inter_area = interfacial_area(a, Nd=1e9)
#     Ai.append(inter_area)

# plt.plot(alpha, Ai)
# plt.show()
