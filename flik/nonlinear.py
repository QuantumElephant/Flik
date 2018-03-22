# An experimental local optimization package
# Copyright (C) 2018 Ayers Lab <ayers@mcmaster.ca>.
#
# This file is part of Flik.
#
# Flik is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Flik is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>


r"""
Solvers for nonlinear systems using the Newton and Gauss-Newton algorithms.

Functions used to find the roots of a nonlinear system of equations given the
residual function `f` and its analytical Jacobian `J`. An exactly-determined
system (`m` equations, `m` variables) is best solved with the Newton method.
Over- or under- determined systems (`m` equations, `n` variables) must be
solved in the least-squares sense using the Gauss-Newton method.

"""


from numbers import Integral
from numbers import Real
from .approx_jacobian import CentralDiffJacobian
from .jacobian import Jacobian
from .step import step_linear
from .step import step_inv

import numpy as np


__all__ = [
    "newton_solve",
    "gauss_newton_solve",
    ]


def newton_solve(f, x_0, J=None, eps=1.0e-6, maxiter=100, method="analytic"):
    r"""
    Solve a system of nonlinear equations with the Newton method.

    Parameters
    ----------
    f : callable
        Vector-valued function corresponding to nonlinear system of equations.
        Must be of the form f(x), where x is a 1-dimensional array.
    x_0 : np.ndarray
        Solution initial guess.
    J : callable, optional
        Jacobian of function f. Must be of the form J(x), where x is a
        1-dimensional array. If none is given, then the Jacobian is calculated
        using finite differences.
    eps : float, optional
        Convergence threshold for vector function f norm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    method : str, optional
        Update method for the (approximated) J(x) or the inverse of J(x). The
        default uses Newton method.

    Returns
    -------
    result : dict
        A dictionary with the keys:
        success
            Boolean variable informing whether the algorithm succeeded or not.
        message
            Information about the cause of the termination.
        niter
            Number of actual iterations performed.
        x
            Nonlinear system of equations solution (Root).
        f
            Vector function evaluated at solution.
        J
            Jacobian evaluated at solution.
        eps
            Convergence threshold for vector function f norm.

    Raises
    ------
    TypeError
        If an argument of an invalid type or shape is passed.
    ValueError
        If an argument passed has an unreasonable value.

    """
    # Check input types and values
    if not callable(f):
        raise TypeError("Argument f should be callable")
    if not callable(J) and J is not None:
        raise TypeError("Argument J should be callable")
    if not (isinstance(x_0, np.ndarray) and x_0.ndim == 1):
        raise TypeError("Argument x_0 should be a 1-dimensional numpy array")
    if not isinstance(eps, Real):
        raise TypeError("Argument eps should be a real number")
    if not isinstance(maxiter, Integral):
        raise TypeError("Argument maxiter should be an integer number")
    if not isinstance(method, str):
        raise TypeError("Argument method should be a string")
    if eps < 0.0:
        raise ValueError("Argument eps should be >= 0.0")
    if maxiter <= 0:
        raise ValueError("Argument maxiter should be >= 1")

    # Handle default arguments
    if J is None:
        m = x_0.size
        J = CentralDiffJacobian(f, m, eps=1e-12)
    else:
        J = Jacobian(J)

    # Convert optimization parameters to predictable types
    eps = float(eps)
    maxiter = int(maxiter)

    # Calculate f_0 and J_0
    A = J(x_0)
    b = f(x_0)
    x_k = x_0
    x_k1 = x_0
    # Choose the step function
    if method in ("analytic", "good_broyden", "bfgs", "sr1", "dfp"):
        step_function = step_linear
    else:
        step_function = step_inv
        A = np.linalg.inv(A)

    # Choose the update function
    if method == "analytic":
        update = J.update_analytic
    elif method == "good_broyden":
        update = Jacobian.update_goodbroyden
    elif method == "bfgs":
        update = Jacobian.update_approx_bfgs
    elif method == "sr1":
        update = Jacobian.update_sr1
    elif method == "bad_broyden":
        update = Jacobian.update_badbroyden
    elif method == "bfgs_inv":
        update = Jacobian.update_inv_bfgs
    elif method == "sr1_inv":
        update = Jacobian.update_sr1_inv
    elif method == "dfp":
        update = Jacobian.update_dfp
    else:
        raise ValueError("Argument method is not a valid option")

    # Check f and J return types
    if not (isinstance(b, np.ndarray) and b.ndim == 1):
        raise TypeError("Argument f must return a 1-dimensional numpy array")
    if not (isinstance(A, np.ndarray) and A.ndim == 2):
        raise TypeError("Argument J must return a 2-dimensional numpy array")
    if A.shape != (x_0.size, x_0.size):
        raise TypeError("J returns a vector of mismatched size")

    # Iterations
    success = False
    message = "Maximum number of iterations reached."
    for niter in range(1, maxiter + 1):
        f_k = np.copy(b)
        b *= -1
        # Calculate step function
        dx = step_function(b, A)
        if dx is None:
            message = "Singular Jacobian; no solution found."
            break
        # Take Newton step
        x_k1 = x_k + dx
        # Evaluate function and Jacobian for next step or result
        b = f(x_k1)
        A = update(A, f_k, b, x_k, x_k1)
        x_k = x_k1
        # Check for convergence
        if np.linalg.norm(b) < eps:
            # If so, we're done (SUCCESS)
            success = True
            message = "Convergence obtained."
            break
    if step_function == step_inv:
        A = np.linalg.inv(A)
    return {
        "success": success,
        "message": message,
        "niter": niter,
        "x": x_k1,
        "f": b,
        "J": A,
        "eps": eps,
        }


def gauss_newton_solve(f, x_0, J=None, eps=1.0e-6, maxiter=100, method="analytic"):
    r"""
    Solve a system of nonlinear equations in the least squares sense.

    Parameters
    ----------
    f : callable
        Vector-valued function corresponding to nonlinear system of equations.
        Must be of the form f(x), where x is a 1-dimensional array.
    x_0 : np.ndarray
        Solution initial guess.
    J : callable, optional
        Jacobian of function f. Must be of the form J(x), where x is a
        1-dimensional array. If none is given, then the Jacobian is calculated
        using finite differences.
    eps : float, optional
        Convergence threshold for vector function f norm.
    maxiter : int, optional
        Maximum number of iterations to perform.
    method : str, optional
        Update method for the (approximated) J(x) or the inverse of J(x). The
        default uses Gauss-Newton method.

    Returns
    -------
    result : dict
        A dictionary with the keys:
        success
            Boolean variable informing whether the algorithm succeeded or not.
        message
            Information about the cause of the termination.
        niter
            Number of actual iterations performed.
        x
            Nonlinear system of equations solution (Root).
        f
            Vector function evaluated at solution.
        J
            Jacobian evaluated at solution.
        eps
            Convergence threshold for vector function f norm.

    Raises
    ------
    TypeError
        If an argument of an invalid type or shape is passed.
    ValueError
        If an argument passed has an unreasonable value.

    """
    # Check input types and values
    if not callable(f):
        raise TypeError("Argument f should be callable")
    if not (isinstance(x_0, np.ndarray) and x_0.ndim == 1):
        raise TypeError("Argument x_0 should be a 1-dimensional numpy array")
    if not isinstance(eps, Real):
        raise TypeError("Argument eps should be a real number")
    if not isinstance(maxiter, Integral):
        raise TypeError("Argument maxiter should be an integer number")
    if not isinstance(method, str):
        raise TypeError("Argument method should be a string")
    if eps < 0.0:
        raise ValueError("Argument eps should be >= 0.0")
    if maxiter <= 0:
        raise ValueError("Argument maxiter should be >= 1")
    if method not in ("analytic", "good_broyden", "bad_broyden", "bfgs"):
        raise ValueError("Argument method is not a valid option")

    # Handle default arguments
    if J is None:
        n = f(x_0).size
        m = x_0.size
        J = CentralDiffJacobian(f, n, m)
    else:
        J = Jacobian(J)

    # Convert optimization parameters to predictable types
    eps = float(eps)
    maxiter = int(maxiter)

    # Calculate f_0 and J_0
    b = f(x_0)
    A = J(x_0)

    # Check f and J return types
    if not (isinstance(b, np.ndarray) and b.ndim == 1):
        raise TypeError("f must return a 1-dimensional numpy array")
    if not (isinstance(A, np.ndarray) and A.ndim == 2):
        raise TypeError("J must return a 2-dimensional numpy array")
    if A.shape != (b.size, x_0.size):
        raise TypeError("J returns a vector of mismatched size")

    # Start iteration
    success = False
    message = "Maximum number of iterations reached."
    for niter in range(1, maxiter + 1):
        # Compute (b = -f)
        b *= -1
        # Compute pseudo-Hessian
        b = np.dot(A.T, b)
        AT_A = np.dot(A.T, A)
        # Solve for dx (A * dx = b)
        try:
            dx = np.linalg.solve(AT_A, b)
        except np.linalg.LinAlgError:
            message = "Singular Jacobian; no solution found."
            b = f(x_0)
            break
        # Take Gauss-Newton step
        x_0 += dx
        # Evaluate function and Jacobian for next step or result
        b = f(x_0)
        A = J(x_0)
        # Check for convergence
        if np.linalg.norm(b) < eps:
            # If so, we're done (SUCCESS)
            success = True
            message = "Convergence obtained."
            break

    # Return success/fail state, system data, and number of iterations
    return {
        "success": success,
        "message": message,
        "niter": niter,
        "x": x_0,
        "f": b,
        "J": A,
        "eps": eps,
        }
