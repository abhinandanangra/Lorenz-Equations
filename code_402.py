#%%
# Importing the necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
'''Definitions'''
# Defining the lorenz system
def lorenz(t, XYZ, sigma, rho, beta):
    x, y, z = XYZ
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return [xdot, ydot, zdot]

#---------------------------------------------------------------------
#%%
'''space trajectories'''
# Solve the system using solve_ivp
sol_trajectories = solve_ivp(lorenz, (0, 100), [1, 1, 1], args=(10, 28, 8/3), dense_output=True)

# Plotting x, y, and z in three separate plots
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(sol_trajectories.t,sol_trajectories.y[0], 'r', label='x')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('x vs t')

plt.subplot(3, 1, 2)
plt.plot(sol_trajectories.t, sol_trajectories.y[1], 'g', label='y')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('y vs t')

plt.subplot(3, 1, 3)
plt.plot(sol_trajectories.t, sol_trajectories.y[2], 'b', label='z')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('z vs t')

plt.tight_layout()
plt.show()

#---------------------------------------------------------------
#%%
''' Phase portraits'''
# Solve the system using solve_ivp
sol_pp = solve_ivp(lorenz, (0, 100), [1, 1, 1], args=(10,28,8/3), dense_output=True)

# Plotting the solution (3D)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(3,2,1, projection='3d')
ax.plot(sol_pp.sol(np.linspace(0,25,1000))[0], sol_pp.sol(np.linspace(0,25,1000))[1], sol_pp.sol(np.linspace(0,25,1000))[2], label='Lorenz System')
ax.set_title('Lorenz System - 3D Phase Portrait')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# Plotting 2D phase portraits
plt.figure(figsize=(12,6))

plt.subplot(131)
plt.plot(sol_pp.sol(np.linspace(0,25,1000))[0],sol_pp.sol(np.linspace(0,25,1000))[1])
plt.xlabel('X(t)')
plt.ylabel('Y(t)')
plt.grid(True)

plt.subplot(132)
plt.plot(sol_pp.sol(np.linspace(0,25,1000))[1],sol_pp.sol(np.linspace(0,25,1000))[2])
plt.xlabel('Y(t)')
plt.ylabel('Z(t)')
plt.grid(True)

plt.subplot(133)
plt.plot(sol_pp.sol(np.linspace(0,25,1000))[0],sol_pp.sol(np.linspace(0,25,1000))[2])
plt.xlabel('X(t)')
plt.ylabel('Z(t)')
plt.grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------
#%%
'''Sensitivity to initial conditions'''
# Solve the system for the original and perturbed initial conditions
epsilon = 1e-3
initial_conditions = [1,1,1]
perturbed_initial_conditions_1 = [initial_conditions[0] + epsilon, initial_conditions[1], initial_conditions[2]]
perturbed_initial_conditions_2 = [initial_conditions[0] - epsilon, initial_conditions[1], initial_conditions[2]]

# Solve the system using solve_ivp
solution_original = solve_ivp(lorenz, (0, 100), initial_conditions, args=(10, 28, 8/3), dense_output=True)
solution_perturbed_1 = solve_ivp(lorenz, (0, 100), perturbed_initial_conditions_1, args=(10, 28, 8/3), dense_output=True)
solution_perturbed_2 = solve_ivp(lorenz, (0, 100), perturbed_initial_conditions_2, args=(10, 28, 8/3), dense_output=True)


# Plotting sensitivity to initial conditions
fig = plt.figure(figsize=(12, 6))

# x vs t
plt.subplot(3, 1, 1)
plt.plot(solution_original.t, solution_original.y[0], 'b', label='Original Trajectory')
plt.plot(solution_perturbed_1.t, solution_perturbed_1.y[0], 'r', label='Perturbed Trajectory 1')
plt.plot(solution_perturbed_2.t, solution_perturbed_2.y[0], 'g', label='Perturbed Trajectory 2')
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Lorenz Equations - Sensitivity to Initial Conditions (x)')
plt.legend()
plt.grid(True)

# y vs t
plt.subplot(3, 1, 2)
plt.plot(solution_original.t, solution_original.y[1], 'b', label='Original Trajectory')
plt.plot(solution_perturbed_1.t, solution_perturbed_1.y[1], 'r', label='Perturbed Trajectory 1')
plt.plot(solution_perturbed_2.t, solution_perturbed_2.y[1], 'g', label='Perturbed Trajectory 2')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Lorenz Equations - Sensitivity to Initial Conditions (y)')
plt.legend()
plt.grid(True)

# z vs t
plt.subplot(3, 1, 3)
plt.plot(solution_original.t, solution_original.y[2], 'b', label='Original Trajectory')
plt.plot(solution_perturbed_1.t, solution_perturbed_1.y[2], 'r', label='Perturbed Trajectory 1')
plt.plot(solution_perturbed_2.t, solution_perturbed_2.y[2], 'g', label='Perturbed Trajectory 2')
plt.xlabel('Time')
plt.ylabel('z')
plt.title('Lorenz Equations - Sensitivity to Initial Conditions (z)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------------------------------------
# %%
'''Observing the system after transient stage for Beta = 0.5'''
# Solve the system using solve_ivp
sol_pt = solve_ivp(lorenz, [0, 140], [1, 1, 1], args=(10, 28, 0.5), dense_output=True,t_eval = np.linspace(40,140,200))

# Plotting x, y, and z in three separate plots
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(np.linspace(40,140,200), sol_pt.y[0], 'r', label='x')
plt.legend()
plt.xlabel('Time')
plt.ylabel('x')
plt.title('Lorenz Equations - Time Evolution after transient time')

plt.subplot(3, 1, 2)
plt.plot(np.linspace(40,140,200), sol_pt.y[1], 'g', label='y')
plt.legend()
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Lorenz Equations - Time Evolution after transient time')

plt.subplot(3, 1, 3)
plt.plot(np.linspace(40,140,200), sol_pt.y[2], 'b', label='z')
plt.legend()
plt.xlabel('Time')
plt.ylabel('z')
plt.title('Lorenz Equations - Time Evolution after transient time')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------------
# %%
'''Depicting the intersection points'''
solution = solve_ivp(lorenz, (0, 100), [0.1,0.1,0.1], args=(10,28,0.4), dense_output=True)

# Extract the solution
t_values = np.linspace(40, 100, 10000)
xyz_solution = solution.sol(t_values)

# Find intersection points with x=0 plane moving in positive direction
intersection_points = []

for i in range(1, len(t_values)):
    if xyz_solution[0, i-1] < 0 and xyz_solution[0, i] >= 0:
        t_interp = np.interp(0, [xyz_solution[0, i-1], xyz_solution[0, i]], [t_values[i-1], t_values[i]])
        z_intersect = solution.sol(t_interp)[2]
        #z_interp = np.interp(t_interp, t_values, xyz_solution[2])
        intersection_points.append(z_intersect)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz_solution[0], xyz_solution[1], xyz_solution[2], label='Lorenz System')
ax.scatter([], [], [], label='Intersection Points', color='red')  # Empty scatter for legend
ax.scatter([0] * len(intersection_points), [0] * len(intersection_points), intersection_points, color='red')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Lorenz System - 3D Trajectory with Intersection Points (Beta = 0.4)')
ax.legend()
plt.show()

#-----------------------------------------------------------------------------------------
# %%
'''Bifurcation diagram method 1'''
beta_range = np.linspace(0.2, 1.0, 100)
t_values = np.linspace(40, 1000, 10000)

# Store intersection points for each beta
all_intersection_points = []

for beta in beta_range:
    # Solve the system using solve_ivp
    solution = solve_ivp(lorenz, (0, 1000), [1,1,1], args=(10,28, beta), dense_output=True)

    # Extract the solution
    xyz_solution = solution.sol(t_values)

    # Find intersection points and append to the list
    for i in range(1, len(t_values)):
        if xyz_solution[0, i-1] < 0 and xyz_solution[0, i] >= 0:
            t_interp = np.interp(0, [xyz_solution[0, i-1], xyz_solution[0, i]], [t_values[i-1], t_values[i]])
            z_intersect = solution.sol(t_interp)[2]
            #z_interp = np.interp(t_interp, t_values, xyz_solution[2])
            all_intersection_points.append((beta, z_intersect))

# Unpack the list of tuples into separate lists for plotting
betas, intersection_points = zip(*all_intersection_points)

# Plot bifurcation diagram
plt.scatter(betas, intersection_points, s=0.1, c='b', marker='.')
plt.xlabel('Beta')
plt.ylabel('Z-axis Intersection Points')
plt.title('Lorenz System - Bifurcation Diagram')
plt.show()

# %%
''' Bifurcation diagram method 2'''
beta_range = np.linspace(0.2, 1.0, 100)
t_values = np.linspace(40, 1000, 10000)

# Lists to store bifurcation diagram points
beta_max_points = []
beta_min_points = []
z_max_points = []
z_min_points = []

for beta in beta_range:
    # Solve the system using solve_ivp
    solution = solve_ivp(lorenz, (0, 1000), [1,1,1], args=(10,28, beta), dense_output=True)

    # Extract the solution
    xyz_solution = solution.sol(t_values)

    # Find local maxima and minima within the loop
    for i in range(1, len(t_values) - 1):
        if xyz_solution[2, i - 1] < xyz_solution[2, i] and xyz_solution[2, i] > xyz_solution[2, i + 1]:
            beta_max_points.append(beta)
            z_max_points.append(xyz_solution[2, i])
        elif xyz_solution[2, i - 1] > xyz_solution[2, i] and xyz_solution[2, i] < xyz_solution[2, i + 1]:
            beta_min_points.append(beta)
            z_min_points.append(xyz_solution[2, i])


# Plot bifurcation diagram
plt.scatter(beta_max_points, z_max_points, s=0.1, color = 'black', marker='.')
plt.scatter(beta_min_points, z_min_points, s=0.1, color = 'red', marker='.')
plt.xlabel('Beta')
plt.ylabel('Z-axis Extrema (maxima)')
plt.title('Lorenz System - Bifurcation Diagram (Method 2)')
plt.ylim(5,45)
plt.show()

#-----------------------------------------------------------------------------------
# %%
'''Depicting the intersection points - Changing Rho'''
solution = solve_ivp(lorenz, (0, 100), [0.1,0.1,0.1], args=(10,24,8/3), dense_output=True)

# Extract the solution
t_values = np.linspace(40, 100, 10000)
xyz_solution = solution.sol(t_values)

# Find intersection points with x=0 plane moving in positive direction
intersection_points = []

for i in range(1, len(t_values)):
    if xyz_solution[0, i-1] < 0 and xyz_solution[0, i] >= 0:
        t_interp = np.interp(0, [xyz_solution[0, i-1], xyz_solution[0, i]], [t_values[i-1], t_values[i]])
        z_intersect = solution.sol(t_interp)[2]
        #z_interp = np.interp(t_interp, t_values, xyz_solution[2])
        intersection_points.append(z_intersect)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz_solution[0], xyz_solution[1], xyz_solution[2], label='Lorenz System')
ax.scatter([], [], [], label='Intersection Points', color='red')  # Empty scatter for legend
ax.scatter([0] * len(intersection_points), [0] * len(intersection_points), intersection_points, color='red')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Lorenz System - 3D Trajectory with Intersection Points (rho = 24)')
ax.legend()
plt.show()

#----------------------------------------------------------------------------------------
#%%
''' Bifurcation diagram method 2 - Changing rho'''
rho_range = np.linspace(20, 100, 1000)
t_values = np.linspace(40, 1000, 10000)

# Lists to store bifurcation diagram points
rho_max_points = []
rho_min_points = []
z_max_points = []
z_min_points = []

for rho in rho_range:
    # Solve the system using solve_ivp
    solution = solve_ivp(lorenz, (0, 1000), [1,1,1], args=(10,rho, 8/3), dense_output=True)

    # Extract the solution
    xyz_solution = solution.sol(t_values)

    # Find local maxima and minima within the loop
    for i in range(1, len(t_values) - 1):
        if xyz_solution[2, i - 1] < xyz_solution[2, i] and xyz_solution[2, i] > xyz_solution[2, i + 1]:
            rho_max_points.append(rho)
            z_max_points.append(xyz_solution[2, i])
        elif xyz_solution[2, i - 1] > xyz_solution[2, i] and xyz_solution[2, i] < xyz_solution[2, i + 1]:
            rho_min_points.append(rho)
            z_min_points.append(xyz_solution[2, i])


# Plot bifurcation diagram
plt.scatter(rho_max_points, z_max_points, s=0.1, color = 'black', marker='.')
plt.scatter(rho_min_points, z_min_points, s=0.1, color = 'red', marker='.')
plt.xlabel('Rho')
plt.ylabel('Z-axis Extrema (maxima)')
plt.title('Lorenz System - Bifurcation Diagram (Method 2) for changing rho')
plt.show()

#----------------------------------------------------------------------------------
#
# %%
'''Depicting the intersection points - Changing Sigma'''
solution = solve_ivp(lorenz, (0, 100), [0.1,0.1,0.1], args=(5,28,8/3), dense_output=True)

# Extract the solution
t_values = np.linspace(40, 100, 10000)
xyz_solution = solution.sol(t_values)

# Find intersection points with x=0 plane moving in positive direction
intersection_points = []

for i in range(1, len(t_values)):
    if xyz_solution[0, i-1] < 0 and xyz_solution[0, i] >= 0:
        t_interp = np.interp(0, [xyz_solution[0, i-1], xyz_solution[0, i]], [t_values[i-1], t_values[i]])
        z_intersect = solution.sol(t_interp)[2]
        #z_interp = np.interp(t_interp, t_values, xyz_solution[2])
        intersection_points.append(z_intersect)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz_solution[0], xyz_solution[1], xyz_solution[2], label='Lorenz System')
ax.scatter([], [], [], label='Intersection Points', color='red')  # Empty scatter for legend
ax.scatter([0] * len(intersection_points), [0] * len(intersection_points), intersection_points, color='red')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Lorenz System - 3D Trajectory with Intersection Points (Sigma = 24)')
ax.legend()
plt.show()
# %%
