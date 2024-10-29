import numpy as np
import opengen as og
import casadi.casadi as cs
from typing import Dict, Any, Optional
from problem import Problem
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, problem: Problem):
        assert isinstance(problem, Problem)
        self.problem = problem
        self.x_sol = None
        self.x_init = None
        self.verbose = True

    def solve(self, x_init):
        self.x_init = x_init
    
        # Define optimization variables
        N = self.problem.N
        z = cs.SX.sym('z', 2*N) # z = [x_1, y_1, x_2, y_2, ...x_N, y_N] (2N × 1)
        end_points = cs.SX.sym('end_points', 4) # end_points = [x0_0, x0_1, xf_0, xf_1]
        z_start = end_points[:2]
        z_goal = end_points[2:]
        z_ = cs.vertcat(z_start, z, z_goal)
    
        # Define cost function
        cost = self.problem.get_cost(z_) 
    
        # Define constraints
        g = self.problem.get_nonlincon(z_)

        # x_y_min = [25, -28] * N
        # x_y_max = [36, 10] * N
        # bounds = og.constraints.Rectangle(x_y_min, x_y_max)
    
        # Create OpEn problem
        problem = og.builder.Problem(z, end_points, cost)\
            .with_penalty_constraints(g)\
            # .with_aug_lagrangian_constraints(g)\
            # .with_constraints(bounds)
    
        # Configure solver
        build_config = og.config.BuildConfiguration()\
            .with_build_directory("python_build")\
            .with_tcp_interface_config()
        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("path_generation")
        
        # solver_config = og.config.SolverConfiguration()\
        #     .with_tolerance(1e-4)\
        #     .with_initial_tolerance(1e-3)\
        #     .with_max_inner_iterations(100000)
        
        solver_config = og.config.SolverConfiguration()\
            .with_tolerance(1e-4)\
            .with_initial_tolerance(1e-3)\
            .with_max_inner_iterations(1e7)\
            # .with_initial_penalty(10.0)\
            # .with_penalty_weight_update_factor(5.0)\
            # .with_lbfgs_memory(15)\
            # .with_preconditioning(True)\
            # .with_delta_tolerance(1e-4)

            
        
        # Build optimizer
        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
        builder.build()
    
        # Create solver
        solver = og.tcp.OptimizerTcpManager('python_build/path_generation')
        solver.start()
    
        try:
            # Solve the problem
            x0 = np.array(self.problem.map.x_start)
            xf = np.array(self.problem.map.x_goal)
            route_endpoints = np.array([x0, xf]).flatten()
            server_response = solver.call(route_endpoints, initial_guess=x_init)
            if server_response.is_ok():
                # Process results
                solution = server_response.get()
                self.x_sol = solution.solution

                out = {
                    'x': self.x_sol,
                    'N': N,
                    'x0': x_init,
                    'time': solution.solve_time_ms / 1000,  # Convert to seconds
                    'fval': solution.cost,
                    'length': self.problem.length_of(self.x_sol),
                    'exit_status': solution.exit_status
                }

                # if self.verbose:
                #     self.print_info(out)

                return out
            else:
                print("Solver error code:", server_response.get().code)
                print("Error message:", self.get_error_code_explanation(server_response.get().code))
                # print("Error message:", server_response.get().message)
        except Exception as e:
            print("An error occurred:", str(e))
        finally:
            solver.kill()

    # def print_info(self, out):
    #     print("\n---------------------------------------------------------------------")
    #     print(f"\nProblem (N = {out['N']})")
    #     opts = self.problem.options
    #     ineq, eq = self.problem.feasibility_of(out['x0'])
    #     pow_len = '²' if opts['length_smooth'] else ''
    #     pow_pen = '²' if opts['penalty_smooth'] else ''
    #     print(f"\t min_x Σ_i{{|Δx_i|{pow_len} + Σ_{{reg}}p_{{reg}}(x_i){pow_pen}}}")
    #     rel_obs = '=' if opts['equality'] else '≤'
    #     pow_obs = '²' if opts['obstacle_smooth'] else ''
    #     print(f"\t  s.t. [p_{{obs}}(x)]_+{pow_obs} {rel_obs} 0")
    #     print(f"\t       angle(Δx_{{i-1}},Δx_i) ≤ {float(opts['maxalpha']) / np.pi * 180:.1f}°")
    #     pow_r = '²' if opts['maxratio_smooth'] else ''
    #     print(f"\t       |Δx_{{i±1}}|{pow_r} ≤ {float(opts['maxratio']):.1f}{pow_r} |Δx_i|{pow_r}")
    #     f = self.problem.get_cost()
    #     len_init = float(self.problem.length_of(out['x0']))
    #     f_value = float(f(out['x0']))
    #     ineq_value = float(ineq)
    #     eq_value = float(eq)
    #     print(f"\t   x0: len {len_init:.3f} | fval {f_value:.3f} | ineq: {ineq_value:.1e} | eq: {eq_value:.1e}")

    #     print("\nSolver")
    #     sol_length = float(out['length'])
    #     sol_fval = float(out['fval'])
    #     sol_ineq = float(out['ineq'])
    #     sol_eq = float(out['eq'])
    #     print(f"\t sol.: len {sol_length:.3f} | fval {sol_fval:.3f} | ineq: {sol_ineq:.1e} | eq: {sol_eq:.1e}")
    #     print(f"\t exit status: {out['exit_status']}")
    #     print(f"\t time: {float(out['time']):.2f} s.")

    def create_x_init(self, displacement=0):
        N = self.problem.N
        x0 = np.array(self.problem.map.x_start).flatten()
        xf = np.array(self.problem.map.x_goal).flatten()
        
        a = np.linalg.norm(xf - x0) / 2

        if abs(displacement) > 1:
            raise ValueError(f'abs(displacement) = {abs(displacement)} must be smaller than 1')

        out = np.zeros(2*N)
        if displacement == 0:
            t = np.linspace(x0[0], xf[0], N+2)[1:-1]
            out[0::2] = t
            t = np.linspace(x0[1], xf[1], N+2)[1:-1]
            out[1::2] = t
            return out

        # Create circle arc from x0 to xf
        b = displacement * a  # distance of chord from midpoint (x0 + xf) / 2
        v = x0 - xf
        alpha = np.arctan2(v[1], v[0])
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        beta = 2 * np.arctan(2*a*b / (a**2-b**2))
        radius = (a**2+b**2) / (2*b)

        t = np.linspace((np.pi-beta)/2, (np.pi+beta)/2, N+2)[1:-1]
        ell = R @ np.vstack((radius*np.cos(t), (b**2-a**2)/(2*b) + radius*np.sin(t)))
        C = (xf + x0) / 2
        ell[0, :] += C[0]
        ell[1, :] += C[1]
        out[0::2] = ell[0, :]
        out[1::2] = ell[1, :]
        return out

    def plot_trajectory(self, x, *args, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        x0 = self.problem.map.x_start
        xf = self.problem.map.x_goal
        
        # CasADiオブジェクトをNumPy配列に変換
        if isinstance(x, (cs.DM, cs.SX, cs.MX)):
            x = np.array(x).flatten()
        elif isinstance(x, list):
            x = np.array(x)
        
        # Ensure x is a 1D array
        x = x.flatten()
        
        # Reshape x into a 2D array with 2 columns
        x_reshaped = x.reshape(-1, 2)
        
        traj = np.vstack((x0, x_reshaped, xf))
        ax.plot(traj[:, 0], traj[:, 1], *args, **kwargs)

    # def plot_trajectory3D(self, x, *args, **kwargs):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     N = len(x) // 2
    #     x0 = self.problem.map.x_start
    #     xf = self.problem.map.x_goal
    #     traj = np.vstack((x0, x.reshape(-1, 2), xf))
    #     f = self.problem.get_total_penalty_function()
    #     z = np.array([f(traj[j]) for j in range(N+2)])
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(traj[:, 0], traj[:, 1], z+0.1, *args, **kwargs)

    def plot(self, color, ax=None):
        if ax is None:
            ax = plt.gca()
        if self.x_init is None:
            print('Warning: No data to plot (x_init)')
            return
        self.plot_trajectory(self.x_init, ':.', color=color, markerfacecolor='w', linewidth=0.5, ax=ax)
        if self.x_sol is not None:
            self.plot_trajectory(self.x_sol, '-o', color=color, markerfacecolor='w', linewidth=2, ax=ax)

    def get_error_code_explanation(self, error_code):
        error_codes = {
            1000: "Invalid request: Malformed or invalid JSON",
            1600: "Initial guess has incompatible dimensions",
            1700: "Wrong dimension of Langrange multipliers",
            2000: "Problem solution failed (solver error)",
            3003: "Vector `parameter` has wrong length",
        }
        return error_codes.get(error_code, "Error code not found")

if __name__ == '__main__':
    from problem import Problem
    from map import Map
    from quadratic_obstacle import QuadraticObstacle
    from region_map import RegionMap
    from polygon import polygon
    from ball import ball
    import matplotlib.pyplot as plt

    map = RegionMap()
    N = 30
    map.x_start = [35.590685, -27.711422]
    map.x_goal = [26.478673, 9.564082]
    z0, zN = map.x_start, map.x_goal
    z = (np.linspace(z0, zN, N + 2)[1:-1]).flatten()
    problem = Problem(map, N)
    problem.options['length_smooth'] = True
    problem.options['penalty_smooth'] = True
    problem.options['obstacle_smooth'] = True
    problem.options['maxalpha'] = np.pi / 4
    problem.options['maxratio'] = 2
    problem.options['maxratio_smooth'] = True

    land = [polygon([16.088709677419356, 11.006493506493506],[12.21774193548387, -7.8246753246753284],[28.245967741935484, -27.629870129870138],[33.20564516129032, -16.83441558441559],[28.48790322580645, 1.9967532467532438]),
            polygon([25.04032258064516, -24.464285714285722],[33.931451612903224, -38.26298701298702],[48.14516129032258, -22.43506493506494],[34.596774193548384, -12.207792207792211]),
            polygon([46.08870967741936, -18.214285714285722],[46.99596774193549, 18.474025974025963],[36.41129032258064, 4.756493506493506],[39.979838709677416, -9.853896103896107]),
            polygon([14.818548387096774, 19.36688311688311],[29.334677419354833, 19.52922077922078],[27.520161290322577, 14.41558441558441],[15.483870967741934, 10.519480519480515]),
            polygon([25.826612903225804, 18.717532467532468],[29.879032258064516, 4.918831168831166],[40.28225806451613, 5.649350649350643],[46.391129032258064, 18.555194805194798])
            ]
    map.new_region('Land', [0.9290, 0.6940, 0.1250])
    map.add_shapes_to_region('Land', *land)

    populated_area = [polygon([28.836, -32.708], [29.607, -36.124], [32.53, -35.464], [31.759, -32.048]),
        polygon([29.185, -27.687], [29.35, -29.043], [32.001, -28.719], [31.835, -27.362]),
        polygon([30.45, -24.494], [32.097, -24.809], [32.293, -23.781], [30.646, -23.466]),
        polygon([30.179, -22.099], [31.872, -22.221], [32.013, -20.255], [30.32, -20.134]),
        polygon([31.826, -33.217], [32.92, -33.773], [34.082, -31.487], [32.989, -30.931]),
        polygon([32.221, -29.916], [33.536, -29.916], [33.536, -28.183], [32.221, -28.183]),
        polygon([32.871, -26.783], [32.871, -28.183], [33.536, -28.183], [33.536, -26.783]),
        polygon([32.313, -25.511], [32.313, -26.28], [33.536, -26.28], [33.536, -25.511]),
        polygon([31.875, -25.511], [33.536, -25.511], [33.536, -22.838], [31.875, -22.838]),
        polygon([31.875, -20.349], [31.875, -22.838], [33.536, -22.838], [33.536, -20.349]),
        polygon([33.536, -30.856], [33.536, -33.302], [35.197, -33.302], [35.197, -30.856]),
        polygon([33.536, -28.183], [33.536, -30.856], [35.197, -30.856], [35.197, -28.183]),
        polygon([33.536, -25.511], [33.536, -28.183], [35.197, -28.183], [35.197, -25.511]),
        polygon([33.536, -22.838], [33.536, -25.511], [35.197, -25.511], [35.197, -22.838]),
        polygon([33.536, -22.106], [33.536, -22.838], [34.618, -22.838], [34.618, -22.106]),
        polygon([35.197, -28.183], [35.197, -30.856], [36.858, -30.856], [36.858, -28.183]),
        polygon([35.197, -28.183], [36.858, -28.183], [36.858, -25.511], [35.197, -25.511]),
        polygon([34.658, -24.157], [36.628, -24.963], [37.229, -23.493], [35.259, -22.687]),
        polygon([36.858, -28.183], [36.858, -30.578], [37.654, -30.578], [37.654, -28.183]),
        polygon([36.327, -27.292], [37.673, -28.391], [38.569, -27.293], [37.222, -26.195]),
        polygon([24.856, -18.992], [25.294, -20.24], [27.054, -19.623], [26.616, -18.375]),
        polygon([31.691, -19.726], [35.922, -20.728], [36.67, -17.565], [32.439, -16.563]),
        polygon([26.334, 16.705], [27.688, 13.096], [30.145, 14.018], [28.791, 17.627]),
        polygon([14.398, 21.212], [20.922, 13.157], [27.408, 18.41], [20.884, 26.465]),
        polygon([40.009, -26.859], [42.78, -28.265], [44.413, -25.048], [41.642, -23.642]),
        polygon([34.034, -20.664], [35.549, -21.925], [36.207, -21.134], [34.693, -19.873]),
        polygon([43.479, -19.521], [44.456, -20.493], [46.884, -18.051], [45.907, -17.079]),
        polygon([37.098, -8.131], [41.568, -14.026], [46.327, -10.417], [41.858, -4.522]),
        polygon([32.392, 7.207], [33.374, 5.991], [35.54, 7.741], [34.558, 8.957])
    ]
    map.new_region('Population', 'Red')
    map.add_shapes_to_region('Population', *populated_area)

    map.new_region('HistCenter', 'Green')
    hist_center = ball([33.874752, -24.981154], 1)
    map.add_shape_to_region('HistCenter', hist_center)

    problem.set_weight('Land', 4)
    problem.set_weight('Population', 13)
    problem.set_weight('HistCenter', 45)

    map.plot()

    # Define the solver
    solver = Solver(problem)
    x_init = solver.create_x_init(0)
    out = solver.solve(x_init)
    print(out)
    solver.plot('b')
    plt.axis('equal')
    plt.xlim(10, 50)
    plt.ylim(-40, 15)
    plt.show()