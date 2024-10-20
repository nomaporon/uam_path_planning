import opengen as og
import casadi.casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
N = 40  # Number of waypoints
dim = 2  # 2D problem
obstacle_center_point = np.array([31.034679, -9.07367])
r_max = 1.1  # Maximum ratio between consecutive segments
theta_max = np.pi / 6  # Maximum angle between consecutive segments
obstacle_radius = 2.0  # Radius of the obstacle
w_dist, w_obs = 1.0, 500.0  # Weights for distance cost, obstacle penalty, and deviation from initial path
z0 = np.array([35.590685, -27.711422])  # Start point
# z0 = np.array([-25, 35])  # Start point
zN = np.array([26.478673, 9.564082])  # End point
# zN = np.array([35, -25])  # Start point

# Define the optimization variables
z = cs.SX.sym('z', N * dim)
# print(z)
z_end_points = cs.SX.sym('z_end_points', 4)

z_start = z_end_points[:2]
z_goal = z_end_points[2:]

# Define the cost function
def get_cost(z):
    dist_cost = 0
    for i in range(N + 1):
        if i == 0:
            dz = z[:dim] - z_start # dz = z_1 - z_0
        elif i == N:
            dz = z_goal - z[-dim:] # dz = z_N - z_N-1
        else:
            dz = z[i*dim:(i+1)*dim] - z[(i-1)*dim:i*dim] # dz = z_i+1 - z_i
        dist_cost += cs.dot(dz, dz) # add |Δz|^2 to dist_cost

    # Penalty terms
    penalty_from_obs = 0
    for i in range(N):
        zi = z[i*dim:(i+1)*dim]
        dist_to_obs = cs.dot(zi - obstacle_center_point, zi - obstacle_center_point)
        penalty_from_obs += cs.fmax(0, obstacle_radius - dist_to_obs) ** 2 # if z_i is in circle of obs, add penalty
        # if i == N - 1:
        #     print("{} \n".format(dist_to_obs))
        #     print("{} \n".format(penalty_from_obs))
            
    total_cost = w_dist * dist_cost + w_obs * penalty_from_obs
    return total_cost

# Define the constraints function
def constraint_function(z):
    constraints = []
    for i in range(N - 1):
        dz1 = z[i*dim:(i+1)*dim] - (z_start if i == 0 else z[(i-1)*dim:i*dim])
        dz2 = (z_goal if i == N-1 else z[(i+1)*dim:(i+2)*dim]) - z[i*dim:(i+1)*dim]
        
        constraints.append(cs.fmax(0.0, cs.norm_2(dz2) - r_max * cs.norm_2(dz1)))
        constraints.append(cs.fmax(0.0, cs.norm_2(dz1) / r_max - cs.norm_2(dz2)))
        
        cos_theta = cs.dot(dz1, dz2) / (cs.norm_2(dz1) * cs.norm_2(dz2))
        constraints.append(cs.fmax(0.0, cs.cos(theta_max) - cos_theta))

    return cs.vertcat(*constraints)

def create_x_init(displacement=0):
    a = np.linalg.norm(zN - z0) / 2
    if abs(displacement) > 1:
        raise ValueError(f'abs(displacement) = {abs(displacement)} must be smaller than 1')
    out = np.zeros(2*N)
    if displacement == 0:
        t = np.linspace(z0[0], zN[0], N+2)[1:-1]
        out[0::2] = t
        t = np.linspace(z0[1], zN[1], N+2)[1:-1]
        out[1::2] = t
        return out
    # Create circle arc from x0 to xf
    b = displacement * a  # distance of chord from midpoint (x0 + xf) / 2
    v = z0 - zN
    alpha = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    beta = 2 * np.arctan(2*a*b / (a**2-b**2))
    radius = (a**2+b**2) / (2*b)
    t = np.linspace((np.pi-beta)/2, (np.pi+beta)/2, N+2)[1:-1]
    ell = R @ np.vstack((radius*np.cos(t), (b**2-a**2)/(2*b) + radius*np.sin(t)))
    C = (zN + z0) / 2
    ell[0, :] += C[0]
    ell[1, :] += C[1]
    out[0::2] = ell[0, :]
    out[1::2] = ell[1, :]
    return out

# Create an OpenGen problem
problem = og.builder.Problem(z, z_end_points, get_cost(z)) \
            .with_penalty_constraints(constraint_function(z))

# Set up solver options
build_config = og.config.BuildConfiguration()  \
    .with_build_directory("python_build")      \
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("flying_car_optimizer")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_initial_tolerance(1e-4)\
    .with_max_inner_iterations(1000)

# Build the solver
builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
builder.build()

# Use the solver
solver = og.tcp.OptimizerTcpManager('python_build/flying_car_optimizer')
solver.start()

# Generate initial trajectory
end_points = np.array([z0, zN]).flatten()
init_trajectory = (np.linspace(z0, zN, N + 2)[1:-1]).flatten()
# displacements = np.arange(-1, 2) / 4
# print(displacements)
# init_trajectory = create_x_init(displacements[2])

solver.ping()

try:
    server_response = solver.call(end_points, initial_guess=init_trajectory)
    if server_response.is_ok():
        solution = server_response.get()
        u_star = solution.solution
        status = solution.exit_status
        cost = solution.cost
        print("Solution: ", u_star)
        print("Status: ", status)
        print("Cost: ", cost)
        print("Penalty: ", solution.penalty)
        
        # Plot the result
        plt.figure(figsize=(10, 8))
        # plt.figure()
        plt.plot([z0[0], *init_trajectory[::2], zN[0]], [z0[1], *init_trajectory[1::2], zN[1]], 'g--', label='Initial path')
        plt.plot([z0[0], *u_star[::2], zN[0]], [z0[1], *u_star[1::2], zN[1]], 'bo-', label='Optimized path')
        plt.plot(z0[0], z0[1], 'go', markersize=10, label='Start')
        plt.plot(zN[0], zN[1], 'ro', markersize=10, label='Goal')
        plt.plot(obstacle_center_point[0], obstacle_center_point[1], 'ko', markersize=10, label='Obstacle')
        circle = plt.Circle(obstacle_center_point, obstacle_radius, color='k', fill=False)
        plt.gca().add_artist(circle)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Optimized Flying Car Path')
        plt.legend()
        plt.grid(True)
        plt.xlim(20, 40)
        plt.ylim(-30, 15)
        plt.gca().set_aspect('equal', adjustable='box') 
        plt.show()
    else:
        print("Error message:", server_response.get().message)
except Exception as e:
    print("An error occurred:", str(e))

# Close the solver
solver.kill()