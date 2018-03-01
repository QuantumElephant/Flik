"""Test for flik.linesearch.input."""
import numpy as np
from nose.tools import assert_raises
from flik.linesearch.tools import check_input


def test_input_control():
    """Test linesearch.input.check_input."""
    # check var
    assert_raises(TypeError, check_input, var=[1])
    assert_raises(TypeError, check_input, var=np.array([[1, 2]]))
    check_input(var=0)
    check_input(var=0.0)
    check_input(var=np.array([0]))

    # check func
    assert_raises(TypeError, check_input, func='1')
    check_input(func=lambda x: x)

    # check grad
    assert_raises(TypeError, check_input, grad='1')
    check_input(grad=lambda x: x)

    # check func_val
    assert_raises(TypeError, check_input, func_val=np.array([1]))
    check_input(func_val=1)
    check_input(func_val=1.0)

    # check grad_val
    assert_raises(TypeError, check_input, grad_val=[1])
    assert_raises(TypeError, check_input, direction=np.array(1))
    check_input(grad_val=1)
    check_input(grad_val=1.0)
    check_input(grad_val=np.array([1]))

    # check direction
    assert_raises(TypeError, check_input, direction=[1])
    assert_raises(TypeError, check_input, direction=np.array(1))
    check_input(direction=1)
    check_input(direction=1.0)
    check_input(direction=np.array([1]))

    # check alpha
    assert_raises(TypeError, check_input, alpha=np.array([1]))
    assert_raises(ValueError, check_input, alpha=2)
    assert_raises(ValueError, check_input, alpha=0)
    assert_raises(ValueError, check_input, alpha=-1)
    check_input(alpha=1)
    check_input(alpha=1.0)
    check_input(alpha=0.5)

    # check var and direction
    assert_raises(ValueError, check_input, var=1, direction=np.array([2, 3]))
    assert_raises(ValueError, check_input, var=np.array([1]), direction=np.array([2, 3]))
    check_input(var=1, direction=np.array([2]))

    # check val and grad_val
    assert_raises(ValueError, check_input, var=np.array([2, 3]), grad_val=1)
    assert_raises(ValueError, check_input, var=np.array([2, 3]), grad_val=np.array([1]))
    check_input(var=np.array([2]), grad_val=1)

    # check grad_val and direction
    assert_raises(ValueError, check_input, grad_val=1, direction=np.array([2, 3]))
    assert_raises(ValueError, check_input, grad_val=np.array([1]), direction=np.array([2, 3]))
    check_input(grad_val=1, direction=np.array([2]))

    # check var, grad_val and direction
    assert_raises(ValueError, check_input, var=1, grad_val=1, direction=np.array([2, 3]))
    check_input(var=0, grad_val=1, direction=np.array([2]))

    # do nothing
    check_input()
