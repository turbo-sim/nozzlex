import optimistix as optx

solver = optx.Bisection(rtol=1e-3, atol=1e-3)


print(solver.atol)
