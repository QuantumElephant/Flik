"""Zoom function to locate final step length."""

from flik.linesearch.tools import check_input
from flik.linesearch.interpolations import cubic_interpolation


def zoom(var, func, grad, direction, alpha_lo, alpha_hi, const1=1e-4, const2=1e-2):
    r"""Select final step in Line Search Algorithm.

    Parameters:
    -----------
    var: np.array((N,), dtype=float)
        Variables' current values
    func: callable
        Function :math:`f(x)`
    grad: callable
        Gradient of the function :math:`\nabla f(x)`
    direction: np.ndarray((N,), dtype=float) for N variables
        Direction vector
    alpha_lo: float
        Step length that gives the minimum function value
    alpha_hi: float
        Scaling factor that satisfies:
        :math:`phi'(\alpha_{lo})(\alpha_{hi} - \alpha_{lo}) < 0`
    const1: {float, 1e-4}
        Constant c_1 in Strong Wolfe conditions
    const2: {float, 1e-2}
        Constant c_2 in Strong Wolfe conditions

    Raises:
    -------
    TypeError:
        If const1 is not float
        If const2 is not float
    ValueError:
        If alpha_hi doesn't meet the condition:
        :math:`phi'(\alpha_{lo})(\alpha_{hi} - \alpha_{lo}) < 0`

    Returns:
    --------
    alpha_one: float
        New step length

    """
    # Check input quality
    check_input(var=var, func=func, grad=grad, direction=direction, alpha=alpha_lo)
    check_input(alpha=alpha_hi)
    if not isinstance(const1, float):
        raise TypeError("const1 should be a float")
    if not isinstance(const2, float):
        raise TypeError("const2 should be a float")
    if not 0 < const1 < const2 < 1:
        raise ValueError("The condition 0 < const1 < const2 < 1 must be satisfied")

    phi_alphalo = direction.dot(grad(var + alpha_lo*direction))
    if phi_alphalo * (alpha_hi - alpha_lo) >= 0:
        raise ValueError("alpha_hi doesn't satisfy the condition,"
                         " phi(alpha_lo)(alpha_hi - alpha_lo) < 0.")

    while True:
        alpha_one = cubic_interpolation(var, func, grad, direction, alpha_lo, alpha_hi,
                                        interval_begin=alpha_lo, interval_end=alpha_hi)
        phi_one = func(var + alpha_one*direction)
        phi_tmp = func(var) + const1*alpha_one*direction.dot(grad(var))
        if phi_one > phi_tmp or phi_one >= phi_alphalo:
            alpha_hi = alpha_one
        else:
            phi_onep = direction.dot(grad(var + alpha_one*direction))
            if abs(phi_onep) <= -const2*direction.dot(grad(var)):
                return alpha_one
            elif phi_onep*(alpha_hi - alpha_lo) >= 0.:
                alpha_hi = alpha_lo
            alpha_lo = alpha_one
