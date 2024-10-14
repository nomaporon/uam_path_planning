import numpy as np
from typing import Callable, Union, Optional

class Function:
    def __init__(self, f, grad, hess, n=2):
        self.f = f
        self.grad = grad
        self.hess = hess
        self.n = n
        self._is_quadratic = True  # プライベート変数として定義
        self._is_convex = True     # プライベート変数として定義

    @property
    def n(self):
        return self._n if self._n is not None else 1

    @n.setter
    def n(self, dim):
        if not isinstance(dim, (int, float)) or dim <= 0 or int(dim) != dim:
            raise ValueError(f"Size should be a strictly positive integer (got '{dim}' instead)")
        self._n = int(dim)

    @property
    def call(self):
        return self._call

    @call.setter
    def call(self, func):
        if isinstance(func, (int, float)):
            self._is_constant = True
            self._call = lambda x: func
        elif callable(func):
            self._call = func
        else:
            raise TypeError(f"Property must be numeric or callable; got '{type(func)}' instead")

    @property
    def grad(self):
        return self._grad if not self._is_constant else lambda x: np.zeros(self.n)

    @grad.setter
    def grad(self, func):
        if func is None:
            return
        if isinstance(func, np.ndarray):
            if func.ndim != 1:
                raise ValueError("Gradient must be a 1D array")
            if self._n is None:
                self.n = len(func)
            if np.all(func == 0):
                self._is_constant = True
                return
            if self._is_constant:
                raise ValueError("Gradient should be zero for a constant function")
            if len(func) != self.n:
                raise ValueError(f"Mismatch between declared size '{self.n}' and gradient size '{len(func)}'")
            self._is_linear = True
            self._grad = lambda x: func
        elif callable(func):
            self._grad = func
        else:
            raise TypeError(f"Property must be numpy array or callable; got '{type(func)}' instead")

    @property
    def hess(self):
        return self._hess if not self._is_linear else lambda x: np.zeros((self.n, self.n))

    @hess.setter
    def hess(self, func):
        if func is None:
            return
        if isinstance(func, np.ndarray):
            if not np.allclose(func, func.T):
                raise ValueError("Hessian must be a symmetric matrix")
            if self._n is None:
                self.n = func.shape[0]
            if func.shape[0] == 1:
                func = func * np.eye(self.n)
            if func.shape != (self.n, self.n):
                raise ValueError(f"Hessian size '{func.shape}' not matching variable size '{self.n}'")
            if np.all(func == 0):
                self._is_linear = True
                return
            if self._is_linear:
                raise ValueError("Hessian should be zero for a linear function")
            self._is_quadratic = True
            self._hess = lambda x: func
        elif callable(func):
            self._hess = func
        else:
            raise TypeError(f"Property must be numpy array or callable; got '{type(func)}' instead")

    @property
    def is_quadratic(self):
        return self._is_quadratic

    @is_quadratic.setter
    def is_quadratic(self, value):
        self._is_quadratic = value

    @property
    def is_convex(self):
        return self._is_convex

    @is_convex.setter
    def is_convex(self, value):
        self._is_convex = value

    def __call__(self, x):
        return self.f(x)

    def compose(self, A: np.ndarray, b: Optional[np.ndarray] = None):
        m, n = A.shape
        if m == 1 and n == 1:
            if b is not None:
                m = len(b)
            else:
                m = self.n
            n = m
            A = A * np.eye(m)
        elif self._n is not None and self.n != m:
            raise ValueError(f"Size mismatch between function '{self.n}' and scaling matrix A '{A.shape}'")
        
        if b is None:
            b = np.zeros((m, 1))
        elif b.shape != (m, 1):
            raise ValueError(f"Size mismatch between scaling matrix A '{A.shape}' and translation vector '{b.shape}'")

        call_ = self._call
        self._call = lambda x: call_(A @ x + b)
        self._n = n

        if self._grad is not None:
            grad_ = self._grad
            self._grad = lambda x: A.T @ grad_(A @ x + b)
            if self._hess is not None:
                hess_ = self._hess
                self._hess = lambda x: A @ hess_(x) @ A.T

    def __neg__(self):
        f = Function(-self._call if callable(self._call) else -self._call)
        f._n = self._n
        if self._grad is not None:
            f._grad = lambda x: -self._grad(x)
        if self._hess is not None:
            f._hess = lambda x: -self._hess(x)
        f._is_constant = self._is_constant
        f._is_linear = self._is_linear
        f._is_quadratic = self._is_quadratic
        return f

    def __add__(self, other):
        if not isinstance(other, Function):
            other = Function(other)
        f = Function(lambda x: self(x) + other(x))
        f._n = max(self.n, other.n)
        if self._grad is not None and other._grad is not None:
            f._grad = lambda x: self._grad(x) + other._grad(x)
        if self._hess is not None and other._hess is not None:
            f._hess = lambda x: self._hess(x) + other._hess(x)
        f._is_constant = self._is_constant and other._is_constant
        f._is_linear = self._is_linear and other._is_linear
        f._is_quadratic = self._is_quadratic and other._is_quadratic
        return f

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Function):
            other = Function(other)
        f = Function(lambda x: self(x) * other(x))
        f._n = max(self.n, other.n)
        if self._grad is not None and other._grad is not None:
            f._grad = lambda x: self(x) * other._grad(x) + other(x) * self._grad(x)
        if self._hess is not None and other._hess is not None:
            f._hess = lambda x: (self(x) * other._hess(x) + other(x) * self._hess(x) + 
                                 np.outer(self._grad(x), other._grad(x)) + np.outer(other._grad(x), self._grad(x)))
        f._is_constant = self._is_constant and other._is_constant
        f._is_linear = (self._is_constant and other._is_linear) or (other._is_constant and self._is_linear)
        f._is_quadratic = ((self._is_linear and other._is_linear) or 
                           (self._is_constant and other._is_quadratic) or 
                           (other._is_constant and self._is_quadratic))
        return f