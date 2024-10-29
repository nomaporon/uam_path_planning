import numpy as np
from typing import List, Union, Optional, Callable, Tuple
from function import Function
import casadi as cs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

class QuadraticObstacle:
    def __init__(self, *inequalities):
        self.inequalities: List[Function] = []
        self.xy_coords = None  # for plotting
        self.area = float('nan')
        self.center = float('nan')
        self.add(*inequalities)

    # 障害物の境界関数を追加する
    def add(self, *inequalities):
        for ineq in inequalities:
            assert isinstance(ineq, Function), f"Expected Function, got {type(ineq)}"
            assert ineq.is_quadratic, f"Function must be quadratic. is_quadratic: {ineq.is_quadratic}"
            assert ineq.n == 2, f"Function must be 2-dimensional, got {ineq.n}-dimensional"
            self.inequalities.append(ineq)
        if self.xy_coords:
            self.set_plot_data(inequalities)
        
    # x: [x1, x2]
    def penalty_function(self, smooth=True, enlargement=0):
        def psi(x):
            if self.inequalities is None:
                return 0
            
            result = 1
            for h in self.inequalities:
                if smooth:
                    result *= cs.fmin(h(x) - enlargement, 0)**2
                else:
                    result *= cs.fmin(enlargement - h(x), 0)
            return result
        return psi

    def linear_transform(self, A: np.ndarray, b: Optional[np.ndarray] = None) -> None:
        """Apply a linear transformation to the obstacle."""
        if np.linalg.norm(A @ np.linalg.inv(A) - np.eye(2)) > 1e-8:
            print("Warning: Transformation matrix A is not invertible")
        
        if b is None:
            b = np.zeros(2)

        for i, h in enumerate(self.inequalities):
            h.compose(A, b)
            self.inequalities[i] = h

        if self.xy_coords is not None:
            def transf(x):
                return np.linalg.solve(A, x - b)
            
            for j in range(self.xy_coords.shape[1]):
                self.xy_coords[:, j] = transf(self.xy_coords[:, j])

    def rotate(self, angle: float, center: Optional[np.ndarray] = None) -> None:
        """Rotate the obstacle around a center point."""
        if center is None:
            center = np.zeros(2)
            
        A = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
        b = center - A @ center
        self.linear_transform(A, b)

    def translate(self, v: np.ndarray) -> None:
        """Translate the obstacle by vector v."""
        self.linear_transform(np.eye(2), v)

    def rescale(self, rx: float, ry: Optional[float] = None, 
                center: Optional[np.ndarray] = None) -> None:
        """Rescale the obstacle."""
        if ry is None:
            ry = rx
        if center is None:
            center = np.zeros(2)
        elif isinstance(ry, np.ndarray):
            center = ry
            ry = 1

        A = np.array([[rx, 0], [0, ry]])
        b = center - A @ center
        self.linear_transform(A, b)

    def contains(self, x: np.ndarray) -> bool:
        """Check if a point belongs to the obstacle."""
        for h in self.inequalities:
            if h(x) > 1e-14:
                return False
        return True

    # def intersection(self, x0: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
    #     """Return the closest intersection point along a direction."""
    #     p = None
    #     zero = np.zeros_like(direction)
    #     intervals = [[0, float('inf')]]
    #     eps = 1e-14

    #     for h in self.inequalities:
    #         a = h.grad(direction).T @ direction - (h(direction) - h(zero))
    #         b = direction.T @ h.grad(x0)
    #         c = h(x0)
    #         delta = b**2 - 4*a*c
    #         tj_L = 0
    #         tj_U = float('inf')

    #         if abs(a) < eps:  # a = 0
    #             if abs(b) < eps:  # a = b = 0
    #                 if c > eps:
    #                     return None
    #             elif b > 0:  # a = 0, b > 0
    #                 if c > 0:
    #                     return None
    #                 tj_U = -c/b
    #             else:  # a = 0, b < 0
    #                 tj_L = max(0, -c/b)
    #         elif a > 0:
    #             if delta < 0 or (-b + np.sqrt(delta))/(2*a) < 0:
    #                 return None
    #             tj_U = (-b + np.sqrt(delta))/(2*a)
    #             tj_L = max(0, (-b - np.sqrt(delta))/(2*a))
    #         else:  # a < 0
    #             if delta >= 0:
    #                 tj_L = max(b + np.sqrt(delta), 0)/(-2*a)
    #                 if b - np.sqrt(delta) >= 0:
    #                     tj_U = (b - np.sqrt(delta))/(-2*a)

    #         # Update intervals
    #         if tj_L <= tj_U:
    #             intervals = [[max(tj_L, I[0]), min(tj_U, I[1])] 
    #                        for I in intervals]
    #         else:
    #             new_intervals = []
    #             for I in intervals:
    #                 new_intervals.append([I[0], min(I[1], tj_U)])
    #                 new_intervals.append([max(tj_L, I[0]), I[1]])
    #             intervals = new_intervals

    #     # Find minimum valid t
    #     t = float('inf')
    #     for I in intervals:
    #         if I[0] <= I[1]:
    #             t = min(t, I[0])
    #             p = x0 + t * direction

    #     return p

    def set_plot_data(self, xlim, ylim=None):
        """Generate plot data for visualization."""
        coords_given = True
        
        if ylim is None:
            if isinstance(xlim[0], (int, float)):
                ylim = xlim
            else:
                ineqs = xlim
                coords_given = False

        if coords_given:
            if ylim is None:
                if self.xy_coords is not None:
                    xlim = [np.min(self.xy_coords[0, :]), np.max(self.xy_coords[0, :])]
                    ylim = [np.min(self.xy_coords[1, :]), np.max(self.xy_coords[1, :])]
                else:
                    raise ValueError("xy_coords not initialized and only one limit provided")

            ineqs = self.inequalities
            nx = ny = 30
            x_try = np.linspace(xlim[0], xlim[1], nx)
            y_try = np.linspace(ylim[0], ylim[1], ny)
            X, Y = np.meshgrid(x_try, y_try)
            xy_try = np.vstack((X.ravel(), Y.ravel()))

        elif self.xy_coords is None:
            raise ValueError("xlim (ylim) argument(s) must be provided the first time")
        else:
            xy_try = self.xy_coords

        # Find valid points
        valid_points = []
        for i in range(xy_try.shape[1]):
            p = xy_try[:, i]
            if all(h(p) <= 0 for h in ineqs):
                valid_points.append(p)

        if valid_points:
            self.xy_coords = np.array(valid_points).T
        else:
            self.xy_coords = np.zeros((2, 0))

    def set_coordinates(self, coordinates):
        self._xy_coords = np.array(coordinates)
        self.update_center()

    def update_center(self):
        if self._xy_coords is not None:
            self.center = cs.DM(np.mean(self._xy_coords, axis=1))

    def plot(self, *args, **kwargs):
        """Plot the obstacle using scatter plot."""
        if self.xy_coords is None:
            raise ValueError("Run set_plot_data(xlim [, ylim]) first")
        
        return plt.scatter(self.xy_coords[0, :], self.xy_coords[1, :], 
                         s=40, *args, **kwargs)

    def __len__(self):
        """Return the number of inequalities."""
        return len(self.inequalities)

    @property
    def xlim(self):
        """Get x-axis limits."""
        return [np.min(self.xy_coords[0, :]), np.max(self.xy_coords[0, :])]

    @property
    def ylim(self):
        """Get y-axis limits."""
        return [np.min(self.xy_coords[1, :]), np.max(self.xy_coords[1, :])]