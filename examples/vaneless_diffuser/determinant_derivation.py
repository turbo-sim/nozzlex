import sympy as sp

# -------------------------------------------------------------------------
# Symbol definitions
# -------------------------------------------------------------------------
rho, vm, vtheta, h, p, c, G, Ma = sp.symbols("rho vm vtheta h p c G Ma")

# -------------------------------------------------------------------------
# Define matrix A
# -------------------------------------------------------------------------
A = sp.Matrix(
    [
        [rho / vm, 0, -rho * G / c**2, (1 + G) / c**2],
        [rho * vm, 0, 0, 1],
        [0, rho * vm, 0, 0],
        [vm, vtheta, 1, 0],
    ]
)

# -------------------------------------------------------------------------
# Determinant
# -------------------------------------------------------------------------
detA = sp.simplify(A.det())

# Substitute vm = Ma * c
detA_Ma = sp.simplify(detA.subs(vm, Ma * c))

# -------------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------------
print("Det(A):")
print(detA)
print("\nDet(A) in terms of Mach number:")
print(detA_Ma)

print("\nLaTeX form (in terms of Ma):")
print(sp.latex(detA_Ma))
