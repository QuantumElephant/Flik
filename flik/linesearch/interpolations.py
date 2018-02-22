"""Interpolations for step length selection."""
import random as rd
import numpy as np
from flik.linesearch.tools import check_input


def cubic_interpolation(var, func, grad, direction, alpha_prev, alpha_current,
                        interval_begin=0, interval_end=1):
    r"""Find a next step length for functions where the gradient is not expensive to calculate.

    Parameters
    ----------
    var: {np.array(N,), float, int}
        Variables' current values.
    func: callable
        Function :math:`f(x)`.
    grad: callable
        Gradient of the function :math:`\nabla f(x)`.
    direction: {np.ndarray(N,), float, int}
        Direction vector for next step.
    alpha_prev: float
        The (i-1)-th step length estimate
    alpha_current: float
        The i-th step length estimate
    interval_begin: {float, 0}
        Starting point of the interval containing desirable step lengths.
    interval_end: {float, 1}
        End point of the interval containing desirable step lengths

    Raises
    ------
    TypeError
        If interval_begin is not an int or float
        If interval_end is not an int or float
        If not 0 <= interval_begin < interval_end <= 1
    ValueError
        If not 0 < interval_begin < interval_end <= 1.
        If not interval_begin < alpha_prev < alpha_current < interval_end.

    Returns
    -------
    alpha_next: float
        The minimizer :math:`\alpha` of :math:`f(x + \alpha p)`, where alpha_next is in the provided
        interval [interval_begin, interval_end]

    References
    ----------
    This cubic interpolation algorithm is taken from equation (3.59) in "Numerical Optimization" by
    Nocedal and Wright, Second Edition (2006).

    """
    # Check the input arguments
    check_input(var=var, func=func, grad=grad, direction=direction, alpha=alpha_prev)
    check_input(alpha=alpha_current)

    # Check intervals and alphas
    if not isinstance(interval_begin, (int, float)):
        raise TypeError('interval_begin must be an int or a float.')
    if not isinstance(interval_end, (int, float)):
        raise TypeError('interval_end must be an int or a float.')
    if not 0 <= interval_begin < interval_end <= 1:
        raise ValueError("interval_begin and interval_end must satisfy "
                         "0 <= interval_begin < interval_end <= 1")
    if not interval_begin <= alpha_prev <= interval_end:
        raise ValueError("alpha_prev must satisfy interval_begin <= alpha_prev <= interval_end.")
    if not interval_begin <= alpha_current <= interval_end:
        raise ValueError("alpha_current must satisfy interval_begin <= alpha_current <= "
                         "interval_end.")

    # Define the helper functions phi and its derivative
    def phi_current(alpha):
        """Calculate the function value at (var + alpha * direction)."""
        return func(var + alpha * direction)

    def phi_prev(alpha):
        """Calculate the derivative of the previously defined function phi."""
        return grad(var + alpha * direction).dot(direction)

    # If interval_begin or interval_end are not minimizers, then the minimizer lies in the interval
    # [interval_begin, interval_end]
    d_1 = (phi_prev(alpha_prev) + phi_prev(alpha_current) - 3 *
           (phi_current(alpha_prev) - phi_current(alpha_current)) / (alpha_prev - alpha_current))

    d_2 = (np.sign(alpha_current - alpha_prev) *
           np.sqrt(d_1 ** 2 - phi_prev(alpha_prev) * phi_prev(alpha_current)))

    alpha_next = (alpha_current -
                  (alpha_current - alpha_prev) * (phi_prev(alpha_current) + d_2 - d_1) /
                  (phi_prev(alpha_current) - phi_prev(alpha_prev) + 2*d_2))

    # Return the minimizer of phi. It can be a, b, or the calculated value for alpha_next
    d = {interval_begin: phi_current(interval_begin), interval_end: phi_current(interval_end),
         alpha_next: phi_current(alpha_next)}

    return min(d, key=d.get)


def bs1(alpha_1, alpha_2):
    """Return random number inside the interval.

    Parameters
    ----------
    alpha_1: float
        First bound
    alpha_2: float
        Second bound

    Raises
    -------
    TypeError
        If alpha_1 and alpha_2 are not floats
    ValueError
        If alpha_1 and alpha_2 are not in the interval [0,1]

    Returns
    -------
    float
        Random number between alpha_1 and alpha_2

    """
    if not isinstance(alpha_1, float):
        raise TypeError("alpha_1 should be a float")

    if not isinstance(alpha_2, float):
        raise TypeError("alpha_2 should be a float")

    if not 0.0 <= alpha_1 <= 1.0:
        raise ValueError("alpha_1 should be in the interval [0,1]")

    if not 0.0 <= alpha_2 <= 1.0:
        raise ValueError("alpha_2 should be in the interval [0,1]")

    return rd.uniform(alpha_1, alpha_2)


def bs2(alpha_1, alpha_2):
    """Bisect an alpha interval.

    Parameters
    ----------
    alpha_1: float
        First bound
    alpha_2: float
        Second bound

    Raises
    -------
    TypeError
        If alpha_1 and alpha_2 are not floats
    ValueError
        If alpha_1 and alpha_2 are not in the interval [0,1]

    Returns
    -------
    float
        The average of alpha_1 and alpha_2

    """
    if not isinstance(alpha_1, float):
        raise TypeError("alpha_1 should be a float")

    if not isinstance(alpha_2, float):
        raise TypeError("alpha_2 should be a float")

    if not 0.0 <= alpha_1 <= 1.0:
        raise ValueError("alpha_1 should be in the interval [0,1]")

    if not 0.0 <= alpha_2 <= 1.0:
        raise ValueError("alpha_2 should be in the interval [0,1]")

    return (alpha_1 + alpha_2)/2.0


def quad_approx(func, grad, val, alpha, direction):
    """
    Quadratic Approximation

    Parameters
    ----------
    func: callable
        Scalar-valued function in form f(xk), where xk is is a 1-dim array
    grad: callable
        Vector-valued gradient function in form f'(xk), where xk is 1-dim array
    val: np.ndarray(N,)
        Objective function value
    alpha: float
        Step length
    direction: np.ndarray(N,)
        Descent direction

    Raises
    ------
    TypeError
        If func is not callable
        If grad is not callable
        If variable is not 1-dim array of float
        If descent direction is not 1-dim array of float
        If alpha is not float
        If alpha_one is not float
        If func does not return float
        If grad does not return np.ndarray
    ValueError
        If descent direction and value is not the same size
        If alpha is not between zero and one
        If alpha_one is not between zero and alpha

    Returns
    -------
    alpha_one: float
        New step length
    """
    # Check parameters
    check_input(var=val, func=func, grad=grad, direction=direction, alpha=alpha)

    # Set up functions
    phi_zero = func(val)
    phi_zero_prime = grad(val).dot(direction)
    phi_alpha = func(val + alpha * direction)

    # New alpha approximation
    alpha_one = -(phi_zero_prime * alpha**2) / (2 * (phi_alpha - phi_zero - phi_zero_prime * alpha))

    # Check new alpha
    if not isinstance(alpha_one, float):
        raise TypeError('New alpha is not a float.It is {}'.format(type(alpha_one)))
    if not 0 < alpha_one < alpha:
        raise ValueError('New alpha should be greater than zero smaller than alpha.')

    return alpha_one
