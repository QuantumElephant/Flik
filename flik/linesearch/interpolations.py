"""Interpolations for step length selection."""
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
