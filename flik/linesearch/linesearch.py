"""Line Search algorithms for step length selection."""

from flik.linesearch.tools import check_input
from flik.linesearch.interpolations import cubic_interpolation, bs1, bs2
from flik.linesearch.conditions import wolfe, armijo


def line_search_general(var, func, grad, direction, alpha, conditions,
                        interpolation, const1=1e-4, const2=1e-2):
    r"""Find a next-step alpha.

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
    alpha: float
        The current step length estimate
    conditions: {str, list of str}
        Conditions to be satisfied by the new step length.
        Valid options are: 'soft-wolfe', 'strong-wolfe' and 'armijo'
    interpolation: str
        Type of interpolation to be used
        Valid options are: 'cubic', 'bs1' and 'bs2'
    const1: {float, 1e-4}
        Constant c_1 for Wolfe/Armijo conditions
    const2: {float, 1e-2}
        Constant c_2 for Wolfe conditions

    Returns
    -------
    alpha_new: float
        The :math:`\alpha` of :math:`f(x + \alpha p)` that satisfies the requested conditions.

    """
    # Check input quality
    check_input(var=var, grad=grad, direction=direction, alpha=alpha)
    if isinstance(conditions, str):
        conditions = [conditions]
    elif not isinstance(conditions, list):
        raise TypeError("The conditions should be given as a str or a list of str.")
    if not isinstance(interpolation, str):
        raise TypeError("The interpolation choice should be given as a str.")
    if not isinstance(const1, float):
        raise TypeError("Constant c_1 should be given as a float.")
    if not isinstance(const2, float):
        raise TypeError("Constant c_2 should be given as a float.")
    if not 0 < const1 < const2 < 1.0:
        raise ValueError("Constants c_1 and c_2  must meet the condition:"
                         " 0 < c_1 < c_2 < 1.0")
    # Check for duplicates
    if len(conditions) != len(set(conditions)):
        raise ValueError("One or more conditions are duplicated.")
    # Check valid conditions
    for condition in conditions:
        if condition not in ['soft-wolfe', 'strong-wolfe', 'armijo']:
            raise ValueError("Condition, {}, is not implemented.".format(condition))
    # Check valid interpolation method
    if interpolation not in ['cubic', 'bs1', 'bs2']:
        raise ValueError("Interpolation choice, {}, is not implemented.".format(interpolation))

    def satisfies_conditions(alpha):
        """Check conditions."""
        output = []
        for condition in conditions:
            if condition == 'soft-wolfe':
                output.append(wolfe(grad, var, -direction, alpha_new, strong_wolfe=False))
            elif condition == 'strong-wolfe':
                output.append(wolfe(grad, var, -direction, alpha_new, strong_wolfe=True))
            elif condition == 'armijo':
                output.append(armijo(func, grad, var, -direction, alpha_new))
        print(output)
        return all(output)

    def interpolate(interpolation, alpha_new):
        """Use the prefered interpolation."""
        if interpolation == 'cubic':
            return cubic_interpolation(var, func, grad, direction, alpha, alpha_new)
        elif interpolation == 'bs1':
            return bs1(alpha, 1.0)
        elif interpolation == 'bs2':
            return bs2(alpha, 1.0)

    # Interpolate until alpha_new satisfies the conditions
    alpha_new = 1.0
    while True:
        alpha_new = interpolate(interpolation, alpha_new)
        print(alpha_new, alpha)
        for condition in conditions:
            if satisfies_conditions(alpha_new):
                return alpha_new
