import numpy as np
from function import Function
from quadratic_obstacle import QuadraticObstacle
import casadi.casadi as cs

def polygon(*args):
    """
    Creates a convex polygon delimited by points P1, P2, P3, ...
    """
    if len(args) < 3:
        raise ValueError(f"Only {len(args)} vertices given. At least 3 required")
    
    N = len(args)
    assert len(args[0]) == 2, "Points should be 2D"
    P1 = np.array(args[0]).reshape(2, 1)
    Pa = P1

    if Pa.shape[1] != 1:
        raise ValueError(f"Points should be column vectors (P_1 has size {Pa.shape})")

    area = 0
    center = Pa.copy()
    for b in range(1, N):
        Pb = np.array(args[b]).reshape(2, 1)
        if Pb.shape != (2, 1):
            raise ValueError(f"P_{b+1} size {Pb.shape} different from P_1 size (2,1)")
        center += Pb

    obs = QuadraticObstacle()
    indices = [1] + [0] * (N-1)
    remaining = list(range(1, N))
    a = 0
    n = 1
    xmin, xmax = Pa[0, 0], Pa[0, 0]
    ymin, ymax = Pa[1, 0], Pa[1, 0]

    while remaining:
        n += 1
        # find n-th vertex
        for i, b in enumerate(remaining):
            test, f = are_consecutive(a, b, args, N)
            if test:
                Pb = np.array(args[b]).reshape(2, 1)
                xmin, xmax = min(xmin, Pb[0, 0]), max(xmax, Pb[0, 0])
                ymin, ymax = min(ymin, Pb[1, 0]), max(ymax, Pb[1, 0])
                indices[n-1] = b
                remaining.pop(i)
                area += Pa[0, 0]*Pb[1, 0] - Pa[1, 0]*Pb[0, 0]
                a = b
                Pa = Pb
                obs.add(f)
                break
        else:
            raise ValueError("The polygon is nonconvex")

    test, f = are_consecutive(a, 0, args, N)
    if not test:
        raise Warning("Something went wrong: couldn't close polygon")

    area += Pa[0, 0]*P1[1, 0] - Pa[1, 0]*P1[0, 0]
    obs.add(f)
    obs.set_plot_data([xmin, xmax], [ymin, ymax])
    obs.area = abs(area) / 2
    obs.center = center / N

    return obs

def are_consecutive(a_F, b_F, varargin, N):
    f_F = None
    Pa_F = np.array(varargin[a_F]).reshape(2, 1)
    Pb_F = np.array(varargin[b_F]).reshape(2, 1)
    
    def line_F(x):
        return (Pb_F[1, 0] - Pa_F[1, 0]) * (x[0] - Pa_F[0, 0]) - (Pb_F[0, 0] - Pa_F[0, 0]) * (x[1] - Pa_F[1, 0])

    sgn_F = 0
    for j_F in range(N):
        if j_F == a_F or j_F == b_F:
            continue
        sgn1_F = np.sign(line_F(np.array(varargin[j_F]).reshape(2, 1)))
        if sgn1_F == 0:
            raise ValueError("Input contains three aligned points")
        if sgn_F == 0:
            sgn_F = sgn1_F
            continue
        if sgn1_F != sgn_F:
            return False, None

    if sgn_F == 0:
        raise ValueError("The polygon is nonconvex")

    grad_F = np.array([Pb_F[1, 0] - Pa_F[1, 0], Pb_F[0, 0] - Pa_F[0, 0]])
    f_F = Function(lambda x: -sgn_F * line_F(x), lambda x: grad_F, lambda x: np.zeros((2, 2)))

    return True, f_F

# def create_ball(center, r1, r2=None):
#     if r2 is None:
#         r2 = r1
#     center = np.array([center[0], center[1]])

#     def func(x):
#         x_reshaped = cs.reshape(x, (-1, 2)).T
#         return cs.sum1(cs.sum2(((x_reshaped - center) / np.array([r1, r2]))**2)) - 1

#     def grad(x):
#         x_reshaped = cs.reshape(x, (-1, 2)).T
#         return 2 * (x_reshaped - center) / np.array([r1**2, r2**2])

#     def hess(x):
#         return cs.diag(np.array([2/r1**2, 2/r2**2]))

#     f = Function(func, grad, hess)
#     f.is_quadratic = True
#     obs = QuadraticObstacle(f)
#     obs.set_plot_data([float(center[0])-r1, float(center[0])+r1], [float(center[1])-r2, float(center[1])+r2])
#     obs.center = center
#     obs.area = cs.pi * r1 * r2

#     return obs

def create_ball(center, radius):
    center = np.array(center).flatten()
    
    def ball_function(x):
        x_col = cs.reshape(x, (2, 1))
        center_col = cs.SX(center).reshape((2, 1))
        return cs.sum1((x_col - center_col)**2) - radius**2

    def ball_gradient(x):
        x_col = cs.reshape(x, (2, 1))
        center_col = cs.SX(center).reshape((2, 1))
        return 2 * (x_col - center_col)

    def ball_hessian(x):
        return 2 * cs.SX.eye(2)

    ball_func = Function(ball_function, ball_gradient, ball_hessian)
    ball_func.is_quadratic = True

    # 円周上の点を生成
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    coordinates = np.array([x, y])

    obstacle = QuadraticObstacle(ball_func, shape_type='circle', coordinates=coordinates)
    obstacle.center = center
    return obstacle

def create_rectangle(x_min, y_min, x_max, y_max):
    def rect_function(x):
        return cs.fmax(
            cs.fmax(x_min - x[0], x[0] - x_max),
            cs.fmax(y_min - x[1], x[1] - y_max)
        )

    def rect_gradient(x):
        dx = cs.if_else(x[0] < x_min, -1, cs.if_else(x[0] > x_max, 1, 0))
        dy = cs.if_else(x[1] < y_min, -1, cs.if_else(x[1] > y_max, 1, 0))
        return cs.vertcat(dx, dy)

    def rect_hessian(x):
        return cs.SX.zeros(2, 2)  # ヘッセ行列は角を除いてゼロ

    rect_func = Function(rect_function, rect_gradient, rect_hessian)
    rect_func.is_quadratic = False

    coordinates = np.array([
        [x_min, x_max, x_max, x_min],
        [y_min, y_min, y_max, y_max]
    ])
    return QuadraticObstacle(rect_func, shape_type='rectangle', coordinates=coordinates)

def create_polygon(vertices):
    vertices = np.array(vertices)
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)

    def poly_function(x):
        return cs.fmax(
            cs.fmax(x_min - x[0], x[0] - x_max),
            cs.fmax(y_min - x[1], x[1] - y_max)
        )

    def poly_gradient(x):
        dx = cs.if_else(x[0] < x_min, -1, cs.if_else(x[0] > x_max, 1, 0))
        dy = cs.if_else(x[1] < y_min, -1, cs.if_else(x[1] > y_max, 1, 0))
        return cs.vertcat(dx, dy)

    def poly_hessian(x):
        return cs.SX.zeros(2, 2)  # ヘッセ行列は角を除いてゼロ

    poly_func = Function(poly_function, poly_gradient, poly_hessian)
    poly_func.is_quadratic = False

    coordinates = vertices.T
    return QuadraticObstacle(poly_func, shape_type='polygon', coordinates=coordinates)