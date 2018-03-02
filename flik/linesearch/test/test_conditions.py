"""Tests for flik.linesearch.conditions."""
import numpy as np
from numpy.testing import assert_raises
from flik.linesearch.conditions import wolfe


def test_wolfe():
    """Test the wolfe condition."""
    def grad(current_point):
        """Gradient of function :math:`current_point**2`."""
        return 2 * current_point

    # Set parameters
    current_point = np.array([1.8])
    current_step = np.array([-0.6])
    alpha = 0.8
    const2 = 0.9

    const2_w = 1
    assert_raises(TypeError, wolfe, grad, current_point, current_step, alpha, const2_w)
    const2_w = -0.209
    assert_raises(ValueError, wolfe, grad, current_point, current_step, alpha, const2_w)

    # Making current_point and current_step float and int
    current_point_float = 1.8
    assert wolfe(grad, current_point_float, current_step, alpha, const2)
    current_point_int = 2
    assert wolfe(grad, current_point_int, current_step, alpha, const2)
    current_step_float = -0.6
    assert wolfe(grad, current_point, current_step_float, alpha, const2)
    current_step_int = -1
    assert wolfe(grad, current_point, current_step_int, alpha, const2)

    # Checking condition
    assert wolfe(grad, current_point, current_step, alpha, const2, strong_wolfe=True)
    assert not wolfe(grad, current_point, -current_step, alpha, const2, strong_wolfe=True)

    current_point = np.array([2., 1., 3.])
    current_step = np.array([2., 1., 3.])
    alpha = 0.1
    assert wolfe(grad, current_point, current_step, alpha, const2, strong_wolfe=False)
    assert not wolfe(grad, current_point, -current_step, alpha, const2, strong_wolfe=False)
