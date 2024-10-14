import numpy as np
from typing import List, Callable, Tuple
from function import Function
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

class QuadraticObstacle:
    def __init__(self, *inequalities, shape_type='custom', coordinates=None):
        self.inequalities = list(inequalities)
        self.shape_type = shape_type
        self.xy_coords = coordinates
        self.area = np.nan
        self.center = cs.DM.zeros(2)

    def add(self, *inequalities):
        for ineq in inequalities:
            assert isinstance(ineq, Function), f"Expected Function, got {type(ineq)}"
            assert ineq.is_quadratic, f"Function must be quadratic. is_quadratic: {ineq.is_quadratic}"
            assert ineq.n == 2, f"Function must be 2-dimensional, got {ineq.n}-dimensional"
            self.inequalities.append(ineq)

    # def penalty_function(self, smooth=True, enlargement=0):
    #     def psi(x):
    #         x_col = cs.reshape(x, (2, 1))  # Ensure x is a 2x1 column vector
    #         result = cs.SX(1.0)
    #         for h in self.inequalities:
    #             if smooth:
    #                 result *= cs.sum1(cs.fmax(h(x_col) - enlargement, 0)**2)
    #             else:
    #                 result *= cs.sum1(cs.fmax(enlargement - h(x_col), 0))
    #         return result
    
    #     if not cs.is_equal(self.center, cs.DM.zeros(2)):
    #         center_penalty = psi(self.center)
    #         return lambda x: psi(x) / center_penalty if center_penalty != 0 else psi(x)
    #     return psi
    def penalty_function(self, smooth=True, enlargement=0):
        def psi(x):
            # x の形状を確認し、適切に処理する
            if isinstance(x, (np.ndarray, list)):
                x = cs.SX(x)
            
            if x.numel() == 2:  # 2次元ベクトルの場合
                x_col = cs.reshape(x, (2, 1))
                result = cs.SX(1.0)
                for h in self.inequalities:
                    if smooth:
                        result *= cs.sum1(cs.fmax(h(x_col) - enlargement, 0)**2)
                    else:
                        result *= cs.sum1(cs.fmax(enlargement - h(x_col), 0))
            elif x.size()[1] == 2:  # Nx2 の行列の場合
                result = cs.SX(1.0)
                for i in range(x.size()[0]):
                    x_col = x[i, :].T
                    row_result = cs.SX(1.0)
                    for h in self.inequalities:
                        if smooth:
                            row_result *= cs.sum1(cs.fmax(h(x_col) - enlargement, 0)**2)
                        else:
                            row_result *= cs.sum1(cs.fmax(enlargement - h(x_col), 0))
                    result *= row_result
            else:
                raise ValueError(f"Unexpected input shape: {x.size()}")
            return result
    
        if self.center is not None:
            center_sx = cs.SX(self.center)
            if not cs.is_equal(center_sx, cs.SX.zeros(2)):
                center_penalty = psi(center_sx)
                return lambda x: psi(x) / center_penalty if center_penalty != 0 else psi(x)
        return psi

    def linear_transform(self, A: np.ndarray, b: np.ndarray = np.zeros(2)):
        if np.linalg.norm(A @ np.linalg.inv(A) - np.eye(2)) > 1e-8:
            print("Warning: Transformation matrix A is not invertible")
        
        for h in self.inequalities:
            h.f = lambda x, h=h, A=A, b=b: h.f(A @ x + b)
            h.grad = lambda x, h=h, A=A, b=b: A.T @ h.grad(A @ x + b)

        if self.xy_coords is not None:
            transf = lambda x: np.linalg.inv(A) @ (x - b)
            self.xy_coords = np.apply_along_axis(transf, 0, self.xy_coords)

    def rotate(self, angle: float, center: np.ndarray = np.zeros(2)):
        A = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        b = center - A @ center
        self.linear_transform(A, b)

    def translate(self, v: np.ndarray):
        self.linear_transform(np.eye(2), v)

    def rescale(self, rx: float, ry: float = None, center: np.ndarray = np.zeros(2)):
        if ry is None:
            ry = rx
        A = np.diag([rx, ry])
        b = center - A @ center
        self.linear_transform(A, b)

    def contains(self, x):
        return cs.logic_and(*[h.f(x) <= 1e-14 for h in self.inequalities])

    def intersection(self, x0: np.ndarray, dir: np.ndarray) -> np.ndarray:
        # This method is complex and might require more careful translation
        # Placeholder implementation
        return None

    def set_plot_data(self, xlim, ylim=None):
        if ylim is None:
            ylim = xlim
        
        nx, ny = 30, 30
        x_try = np.linspace(xlim[0], xlim[1], nx)
        y_try = np.linspace(ylim[0], ylim[1], ny)
        X, Y = np.meshgrid(x_try, y_try)
        xy_try = np.vstack((X.flatten(), Y.flatten()))
    
        points = [p for p in xy_try.T if self.contains(p)]
        self.xy_coords = np.array(points).T if points else np.empty((2, 0))

    def set_coordinates(self, coordinates):
        self.xy_coords = np.array(coordinates)
        self.update_center()

    def update_center(self):
        if self.xy_coords is not None:
            self.center = cs.DM(np.mean(self.xy_coords, axis=1))

    def plot(self, *args, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        if self.xy_coords is None or self.xy_coords.size == 0:
            print("Warning: No coordinates available for plotting.")
            return

        try:
            print(self.xy_coords)
            if self.shape_type == 'circle':
                if self.xy_coords.shape[1] >= 2:
                    center = self.xy_coords[:, 0]
                    radius = np.linalg.norm(self.xy_coords[:, 1] - center)
                else:
                    print("Warning: Insufficient data for circle plotting.")
                    return
                circle = Circle(center, radius, **kwargs)
                ax.add_patch(circle)
            else:
                if self.xy_coords.shape[0] == 2:
                    coords = self.xy_coords.T
                else:
                    coords = self.xy_coords
                poly = Polygon(coords, closed=True, **kwargs)
                ax.add_patch(poly)
            
            ax.autoscale_view()
        except Exception as e:
            print(f"Error in plot method: {str(e)}")
            import traceback
            traceback.print_exc()

    @property
    def xlim(self):
        return (np.min(self.xy_coords[0]), np.max(self.xy_coords[0])) if self.xy_coords is not None else None

    @property
    def ylim(self):
        return (np.min(self.xy_coords[1]), np.max(self.xy_coords[1])) if self.xy_coords is not None else None

    def __len__(self):
        return len(self.inequalities)

    def __getitem__(self, key):
        return self.inequalities[key]