import numpy as np
import matplotlib.pyplot as plt

# Lorenz equations
def lorenz(x, y, z, sigma, rho, beta):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return dxdt, dydt, dzdt

# Euler method for solving ODEs
def euler_method(func, initial_conditions, sigma, rho, beta, dt, num_steps):
    x, y, z = initial_conditions
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0] = [x, y, z]

    for step in range(num_steps):
        dx, dy, dz = func(x, y, z, sigma, rho, beta)
        x += dt * dx
        y += dt * dy
        z += dt * dz
        trajectory[step + 1] = [x, y, z]

    return trajectory

# Set parameters
sigma = 10.0
rho = 28.0
beta_values = np.linspace(0.2, 1.0, 100)
initial_conditions = [1.0, 1.0, 1.0]
dt = 0.001
num_steps = 1000000

# Iterate over beta values and store intersection points
beta_list = []
z_list = []


for beta in beta_values:
    # Use Euler method to compute solutions
    trajectory = euler_method(lorenz, initial_conditions, sigma, rho, beta, dt, num_steps)
    trajectory = trajectory[400000:] # discard the transient stage

    # Find intersection points with x=0 plane moving in positive direction abd beta values
    intersection_points = []
    for i in range(1, len(trajectory)):
        if trajectory[i-1, 0] < 0 and trajectory[i, 0] >= 0:
            t_interp = np.interp(0, [trajectory[i-1, 0], trajectory[i, 0]], [i-1, i])
            z_interp = np.interp(t_interp, [i-1, i], [trajectory[i-1, 2], trajectory[i, 2]])
            intersection_points.append(z_interp)

    beta_list.extend([beta] * len(intersection_points))
    z_list.extend(intersection_points)

# Plot the bifurcation diagram
plt.figure(figsize=(10, 6))
plt.scatter(beta_list, z_list, s=0.1, c="black", label="Intersection Points")
plt.title("Lorenz System Bifurcation Diagram (Changing Beta)")
plt.xlabel("Beta")
plt.ylabel("Z")
plt.legend()
plt.show()