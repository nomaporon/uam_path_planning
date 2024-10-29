import numpy as np
from typing import List, Union
from quadratic_obstacle import QuadraticObstacle
from function import Function
import casadi as cs

def ball(center: Union[float, List[float]], r1: float = None, r2: float = None) -> QuadraticObstacle:
    """
    Creates an elliptical obstacle of radiuses r1 and r2 centered at center.
    
    Args:
    center: Center of the ellipse. If only one argument is provided, it's treated as the radius.
    r1: Radius in x-direction (optional)
    r2: Radius in y-direction (optional)
    
    Returns:
    QuadraticObstacle: An elliptical obstacle
    """
    if r1 is None and r2 is None:
        r1 = center
        r2 = r1
        center = np.array([0.0, 0.0])
    elif r2 is None:
        r2 = r1
    
    center = np.array(center)
    assert center.shape == (2,)

    # func = lambda x: np.linalg.norm((x - center) / np.array([r1, r2])) ** 2 - 1

    # grad = lambda x: 2 * (x - center) / np.array([r1**2, r2**2])

    def func(x):
        # CasADiのシンボリック変数に対応するため、cs.sumsqr()を使用
        diff = x - center
        scaled_diff = cs.vertcat(diff[0]/r1, diff[1]/r2)
        return cs.sumsqr(scaled_diff) - 1

    def grad(x):
        # 勾配も同様にCasADiの関数を使用
        return 2 * cs.vertcat((x[0] - center[0])/r1**2, 
                             (x[1] - center[1])/r2**2)

    Hess = np.array([[2/r1**2, 0], [0, 2/r2**2]])

    f = Function(func, grad, Hess)
    obs = QuadraticObstacle(f)
    obs.set_plot_data([center[0]-r1, center[0]+r1], [center[1]-r2, center[1]+r2])
    obs.center = center
    obs.area = np.pi * r1 * r2

    return obs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    obs = ball([1, 1], 2, 1)
    obs.plot()
    plt.axis('equal')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()
