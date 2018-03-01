"""Interpolations for step length selection."""
import numpy as np

from flik.linesearch.input import check_input


def cubic_interpolation(var, func, grad, direction, interval_begin, interval_end,
                        alpha_previous, alpha_current):
    r"""Find a next step length for functions where the gradient is not expensive to calculate.

    References
    ----------
    This cubic interpolation algorithm is taken from equation (3.59) in "Numerical Optimization" by
    Nocedal and Wright, Second Edition (2006).


    Parameters
    ----------
    var: np.array((N,1), dtype=float)
        Variables' current values
    func: callable
        Function :math:`f(x)`
    grad: callable
        Gradient of the function :math:`\nabla f(x)`
    direction: np.ndarray((N,1)) for N variables
        Direction vector for next step
    interval_begin: float
        The starting point of the interval containing desirable step lengths
    interval_end: float
        The ending point of the interval containing desirable step lengths
    alpha_previous: float
        The (i-1)-th step length estimate
    alpha_current: float
        The i-th step length estimate

    Raises
    ------
        TypeError
            according to the function check_input
        ValueError
            according to the function check_input
            if interval_begin and interval_end are not such that
                0 < interval_begin < interval_end <= 1
            if alpha_1 and alpha_2 are not such that
                interval_begin < alpha_previous < alpha_current < interval_end

    Returns
    -------
    alpha_next: float
        The minimizer :math:`\alpha` of :math:`f(x + \alpha p)`, where alpha_next is in the provided
        interval [interval_begin, interval_end]

    """
    # Check the input arguments
    #   TODO: modify check_input function to accept more than one alpha value
    check_input(var, func, grad, direction, alpha_previous)
    check_input(var, func, grad, direction, alpha_current)

    #   Check for 0 < interval_begin < interval_end <= 1
    if not (0.0 < interval_begin < interval_end <= 1.0):
        raise ValueError("interval_begin and interval_end should be according to"
                         " 0 < interval_begin < interval_end <= 1")

    #   Check for interval_begin < alpha_1 < alpha_2 < interval_end
    if not (interval_begin < alpha_previous < alpha_current < interval_end):
        raise ValueError("alpha_previous and alpha_current should be according"
                         " interval_begin < alpha_previous < alpha_current < interval_end")

    # Define the helper functions phi and its derivative
    def phi(alpha):
        """Calculate the function value at (var + alpha * direction)."""
        return func(var + alpha * direction)

    def phi_(alpha):
        """Calculate the derivative of the previously defined function phi."""
        return grad(var + alpha * direction).dot(direction)

    # If interval_begin or interval_end are not minimizers, then the minimizer lies in the interval
    # ]interval_begin, interval_end[
    d_1 = phi_(alpha_previous) + phi_(alpha_current) - 3 * (phi(alpha_previous) -
                                                            phi(alpha_current)) / (alpha_previous -
                                                                                   alpha_current)

    d_2 = np.sign(alpha_current - alpha_previous) * np.sqrt(d_1 ** 2 - phi_(alpha_previous) *
                                                            phi_(alpha_current))

    alpha_next = alpha_current - (alpha_current - alpha_previous) * (phi_(alpha_current) + d_2 -
                                                                     d_1) / (phi_(alpha_current) -
                                                                             phi_(alpha_previous) +
                                                                             2 * d_2)

    # Return the minimizer of phi. It can be a, b, or the calculated value for alpha_next
    d = {interval_begin: phi(interval_begin), interval_end: phi(interval_end),
         alpha_next: phi(alpha_next)}

    return min(d, key=d.get)
