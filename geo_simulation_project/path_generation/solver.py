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
        self.options = {
            'max_iter': 500,
            'tol': 1e-3,
            'penalty_weight_update_factor': 5.0,
            'initial_penalty_weights': 1.0
        }

    def solve(self, x_init):
        self.x_init = x_init
    
        # Define optimization variables
        N = self.problem.N
        u = cs.SX.sym('u', 2*N)
    
        # Define cost function
        f = self.problem.get_cost()
        cost = f(u)
        assert cost.is_scalar(), "Cost function must return a scalar value"
    
        # Define constraints
        nonlincon = self.problem.get_nonlincon()
        ineq, eq = nonlincon(u)
    
        # Combine all constraints
        g = cs.vertcat(ineq, eq)

        end_points = cs.SX.sym('end_points', 4)
    
        # Create OpEn problem
        problem = og.builder.Problem(u, end_points, cost)\
            .with_penalty_constraints(g)
    
        # Configure solver
        build_config = og.config.BuildConfiguration()\
            .with_build_directory("python_build")\
            .with_tcp_interface_config()
        meta = og.config.OptimizerMeta()\
            .with_optimizer_name("path_generation")
        
        solver_config = og.config.SolverConfiguration()\
            .with_initial_penalty(self.options['initial_penalty_weights'])\
            .with_penalty_weight_update_factor(self.options['penalty_weight_update_factor'])\
            .with_max_duration_micros(500000)
        # solver_config = og.config.SolverConfiguration()\
        #     .with_tolerance(1e-4)\
        #     .with_initial_tolerance(1e-4)\
        #     .with_max_inner_iterations(1000)
        
        # Build optimizer
        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config)
        builder.build()
    
        # Create solver
        solver = og.tcp.OptimizerTcpManager('python_build/path_generation')
        solver.start()
    
        try:
            # Solve the problem
            x0 = np.array(self.problem.map.x_start).flatten()
            xf = np.array(self.problem.map.x_goal).flatten()
            end_points = np.array([x0, xf]).flatten()
            server_response = solver.call(end_points, initial_guess=x_init)
            if server_response.is_ok():
                # Process results
                solution = server_response.get()
                self.x_sol = solution.solution
                ineq, eq = self.problem.feasibility_of(self.x_sol)

                out = {
                    'x': self.x_sol,
                    'N': N,
                    'x0': x_init,
                    'eq': eq,
                    'ineq': ineq,
                    'time': solution.solve_time_ms / 1000,  # Convert to seconds
                    'fval': solution.cost,
                    'length': self.problem.length_of(self.x_sol),
                    'exit_status': solution.exit_status
                }

                if self.verbose:
                    self.print_info(out)

                return out
            else:
                print("Error message:", server_response.get().message)
        except Exception as e:
            print("An error occurred:", str(e))
        finally:
            solver.kill()

    def print_info(self, out):
        print("\n---------------------------------------------------------------------")
        print(f"\nProblem (N = {out['N']})")
        opts = self.problem.options
        ineq, eq = self.problem.feasibility_of(out['x0'])
        pow_len = '²' if opts['length_smooth'] else ''
        pow_pen = '²' if opts['penalty_smooth'] else ''
        print(f"\t min_x Σ_i{{|Δx_i|{pow_len} + Σ_{{reg}}p_{{reg}}(x_i){pow_pen}}}")
        rel_obs = '=' if opts['equality'] else '≤'
        pow_obs = '²' if opts['obstacle_smooth'] else ''
        print(f"\t  s.t. [p_{{obs}}(x)]_+{pow_obs} {rel_obs} 0")
        print(f"\t       angle(Δx_{{i-1}},Δx_i) ≤ {float(opts['maxalpha']) / np.pi * 180:.1f}°")
        pow_r = '²' if opts['maxratio_smooth'] else ''
        print(f"\t       |Δx_{{i±1}}|{pow_r} ≤ {float(opts['maxratio']):.1f}{pow_r} |Δx_i|{pow_r}")
        f = self.problem.get_cost()
        len_init = float(self.problem.length_of(out['x0']))
        f_value = float(f(out['x0']))
        ineq_value = float(ineq)
        eq_value = float(eq)
        print(f"\t   x0: len {len_init:.3f} | fval {f_value:.3f} | ineq: {ineq_value:.1e} | eq: {eq_value:.1e}")

        print("\nSolver")
        sol_length = float(out['length'])
        sol_fval = float(out['fval'])
        sol_ineq = float(out['ineq'])
        sol_eq = float(out['eq'])
        print(f"\t sol.: len {sol_length:.3f} | fval {sol_fval:.3f} | ineq: {sol_ineq:.1e} | eq: {sol_eq:.1e}")
        print(f"\t exit status: {out['exit_status']}")
        print(f"\t time: {float(out['time']):.2f} s.")
        # print("\n---------------------------------------------------------------------")
        # print(f"\nProblem (N = {out['N']})")
        # opts = self.problem.options
        # ineq, eq = self.problem.feasibility_of(out['x0'])
        # pow_len = '²' if opts['length_smooth'] else ''
        # pow_pen = '²' if opts['penalty_smooth'] else ''
        # print(f"\t min_x Σ_i{{|Δx_i|{pow_len} + Σ_{{reg}}p_{{reg}}(x_i){pow_pen}}}")
        # rel_obs = '=' if opts['equality'] else '≤'
        # pow_obs = '²' if opts['obstacle_smooth'] else ''
        # print(f"\t  s.t. [p_{{obs}}(x)]_+{pow_obs} {rel_obs} 0")
        # print(f"\t       angle(Δx_{{i-1}},Δx_i) ≤ {opts['maxalpha'] / np.pi * 180:.1f}°")
        # pow_r = '²' if opts['maxratio_smooth'] else ''
        # print(f"\t       |Δx_{{i±1}}|{pow_r} ≤ {opts['maxratio']:.1f}{pow_r} |Δx_i|{pow_r}")
        # f = self.problem.get_cost()
        # len_init = self.problem.length_of(out['x0'])
        # print(f"\t   x0: len {len_init:.3f} | fval {f(out['x0']):.3f} | ineq: {ineq:.1e} | eq: {eq:.1e}")

        # print("\nSolver")
        # print(f"\t sol.: len {out['length']:.3f} | fval {out['fval']:.3f} | ineq: {out['ineq']:.1e} | eq: {out['eq']:.1e}")
        # print(f"\t exit status: {out['exit_status']}")
        # print(f"\t time: {out['time']:.2f} s.")

    # def create_x_init(self, displacement=0):
    #     N = self.problem.N
    #     x0 = np.array(self.problem.map.x_start).flatten()
    #     xf = np.array(self.problem.map.x_goal).flatten()
        
    #     a = np.linalg.norm(xf - x0) / 2

    #     if abs(displacement) > 1:
    #         raise ValueError(f'abs(displacement) = {abs(displacement)} must be smaller than 1')

    #     out = np.zeros(2*N)
    #     if displacement == 0:
    #         t = np.linspace(x0[0], xf[0], N+2)[1:-1]
    #         out[0::2] = t
    #         t = np.linspace(x0[1], xf[1], N+2)[1:-1]
    #         out[1::2] = t
    #         return out

    #     # Create circle arc from x0 to xf
    #     b = displacement * a  # distance of chord from midpoint (x0 + xf) / 2
    #     v = x0 - xf
    #     alpha = np.arctan2(v[1], v[0])
    #     R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    #     beta = 2 * np.arctan(2*a*b / (a**2-b**2))
    #     radius = (a**2+b**2) / (2*b)

    #     t = np.linspace((np.pi-beta)/2, (np.pi+beta)/2, N+2)[1:-1]
    #     ell = R @ np.vstack((radius*np.cos(t), (b**2-a**2)/(2*b) + radius*np.sin(t)))
    #     C = (xf + x0) / 2
    #     ell[0, :] += C[0]
    #     ell[1, :] += C[1]
    #     out[0::2] = ell[0, :]
    #     out[1::2] = ell[1, :]
    #     return out

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

    # def plot_trajectory(self, x, *args, **kwargs):
    #     import matplotlib.pyplot as plt
    #     x0 = self.problem.map.x_start
    #     xf = self.problem.map.x_goal
    #     traj = np.vstack((x0, x.reshape(-1, 2), xf))
    #     plt.plot(traj[:, 0], traj[:, 1], *args, **kwargs)

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

    def plot_trajectory3D(self, x, *args, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        N = len(x) // 2
        x0 = self.problem.map.x_start
        xf = self.problem.map.x_goal
        traj = np.vstack((x0, x.reshape(-1, 2), xf))
        f = self.problem.get_total_penalty_function()
        z = np.array([f(traj[j]) for j in range(N+2)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], z+0.1, *args, **kwargs)

    def plot(self, color, ax=None):
        if ax is None:
            ax = plt.gca()
        if self.x_init is None:
            print('Warning: No data to plot (x_init)')
            return
        self.plot_trajectory(self.x_init, ':o', color=color, markerfacecolor='w', linewidth=0.5, ax=ax)
        if self.x_sol is not None:
            self.plot_trajectory(self.x_sol, '-o', color=color, markerfacecolor='w', linewidth=2, ax=ax)