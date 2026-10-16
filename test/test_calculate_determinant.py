import sympy as sp
import jax.numpy as jnp

# =============================================================================
# 9-equation matrix (original two-velocity formulation)
# Variables: [dα1, dα2, dρ1, dρ2, du1, du2, dp, dh1, dh2]
# =============================================================================

# 1. Define all scalar symbols
rho1, rho2, alpha1, alpha2 = sp.symbols('rho1 rho2 alpha1 alpha2')
u1, u2, rho_mix, const, u_inv = sp.symbols('u1 u2 rho_mix const u_inv')
G1, G2, a1, a2 = sp.symbols('G1 G2 a1 a2')

# 2. Define complex terms to keep the matrix construction clean
diff_u = u2 - u1
sum_rho = rho1 + rho2

# Terms from Row 4 & 5
term_A = alpha1 * alpha2 * sum_rho * (diff_u**2)
term_B = alpha1**2 * alpha2 * (sum_rho / rho1) * diff_u * u1
term_C = alpha1 * alpha2**2 * (sum_rho / rho2) * diff_u * u2
term_D1 = (alpha2 * rho1 - alpha1 * rho2) * (u1 - u2) - rho_mix * const * u2
term_D2 = (alpha1 * rho2 - alpha2 * rho1) * (u2 - u1) - rho_mix * const * u1

# 3. Construct the 9x9 Matrix
A_mat_9 = sp.Matrix([
        # Row 1
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 2
        [rho1 * u1, 0.0, alpha1 * u1, 0.0, alpha1 * rho1, 0.0, 0.0, 0.0, 0.0],

        # Row 3
        [0.0, rho2 * u2, 0.0, alpha2 * u2, 0.0, alpha2 * rho2, 0.0, 0.0, 0.0],

        # Row 4
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1, 0.0, alpha1, 0.0, 0.0],

        # Row 5
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2, alpha2, 0.0, 0.0],

        # Row 6
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1**2, 0.0, 0.0, alpha1 * rho1 * u1, 0.0],

        # Row 7
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2**2, 0.0, 0.0, alpha2 * rho2 * u2],

        # Row 8
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, (1 + G1) / a1**2, -(rho1 * G1) / a1**2, 0.0],

        # Row 9
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, (1 + G2) / a2**2, 0.0, -(rho2 * G2) / a2**2],
    ])

# 4. Calculate Determinant (9x9)
print("=" * 60)
print("9-equation matrix (9x9)")
print("=" * 60)
det_A_9 = A_mat_9.det()
simplified_det_9 = sp.factor(det_A_9)

print("Simplified Determinant:")
print(sp.latex(simplified_det_9))

# Eigenvalues (9x9)
lam = sp.symbols('lambda')
char_poly_9 = A_mat_9.charpoly(lam)
eigenvalues_9 = sp.solve(char_poly_9, lam)
print(f"\nEigenvalues ({len(eigenvalues_9)}):")
for ev in eigenvalues_9:
    print(f"  {ev}")


# =============================================================================
# 8-equation matrix (homogeneous, single velocity u1 = u2 = u)
# Variables: [dαv, dρv, dρl, du, dp, dhv, dhl]  (7x7)
# αl = 1 - αv is eliminated
# =============================================================================

print("\n" + "=" * 60)
print("8-equation matrix (7x7, homogeneous u1=u2=u)")
print("=" * 60)

# Symbols for the homogeneous formulation
rhov, rhol = sp.symbols('rhov rhol')
alphav, alphal = sp.symbols('alpha_v alpha_l')
u, rho_mix_h = sp.symbols('u rho_mix')
Gv, Gl, av, al = sp.symbols('Gv Gl av al')

# 7x7 A_mat matching nozzle_model_core.py _lax_eigth
# Columns: [dαv, dρv, dρl, du, dp, dhv, dhl]
# Rows: [MassV, MassL, MixMomentum, V-Energy, L-Energy, V-EOS, L-EOS]
A_mat_8 = sp.Matrix([
    # Row 1: Vapor mass
    [rhov * u, alphav * u, 0, alphav * rhov, 0, 0, 0],

    # Row 2: Liquid mass (αl = 1 - αv, so dαl = -dαv)
    [-rhol * u, 0, alphal * u, alphal * rhol, 0, 0, 0],

    # Row 3: Mixture momentum
    [0, 0, 0, rho_mix_h * u, 1, 0, 0],

    # Row 4: Vapor energy
    [0, 0, 0, alphav * rhov * u**2, 0, alphav * rhov * u, 0],

    # Row 5: Liquid energy
    [0, 0, 0, alphal * rhol * u**2, 0, 0, alphal * rhol * u],

    # Row 6: Vapor EOS
    [0, -1, 0, 0, (1 + Gv) / av**2, -(rhov * Gv) / av**2, 0],

    # Row 7: Liquid EOS
    [0, 0, -1, 0, (1 + Gl) / al**2, 0, -(rhol * Gl) / al**2],
])

# Calculate Determinant (7x7)
det_A_8 = A_mat_8.det()
simplified_det_8 = sp.factor(det_A_8)

print("Simplified Determinant:")
print(sp.latex(simplified_det_8))

# Eigenvalues (7x7)
char_poly_8 = A_mat_8.charpoly(lam)
eigenvalues_8 = sp.solve(char_poly_8, lam)
print(f"\nEigenvalues ({len(eigenvalues_8)}):")
for ev in eigenvalues_8:
    print(f"  {ev}")


# =============================================================================
# Numerical eigenvalue computation (JAX)
# =============================================================================
def compute_eigenvalues(A_mat_np):
    """
    Given a system matrix A_mat, returns all eigenvalues
    sorted by real part.
    """
    eigs = jnp.linalg.eigvals(A_mat_np)
    idx  = jnp.argsort(eigs.real)
    return eigs[idx]