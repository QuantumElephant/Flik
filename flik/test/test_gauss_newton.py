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


"""Test file for `flik.nonlinear.gauss_newton_solve`."""


import numpy as np

from nose.tools import assert_raises

from flik import gauss_newton_solve


__all__ = [
    "test_gauss_newton_inputs",
    "test_gauss_newton_inputs",
    "test_gauss_newton_dimensions",
    "test_gauss_newton_linear_solve",
    "test_gauss_newton_nonlinear_solve",
    "test_gauss_newton_nonlinear_overdetermined_solve",
    "test_gauss_newton_nonlinear_overdetermined_solve",
    "test_gauss_newton_singular_matrix_error",
    ]


def f1(x):
    """Test function."""
    return np.array([136 - x[0] - x[1] ** 2. - x[2] ** 3.,
                     1038 - x[0] - 4. * x[1] ** 2. - 8. * x[2] ** 3.,
                     3458 - x[0] - 9. * x[1] ** 2. - 27. * x[2] ** 3.])


def j1(x):
    """Test function."""
    return np.array([[-1., -2. * x[1], -3. * x[2] ** 2.],
                     [-1., -8. * x[1], -24. * x[2]**2.],
                     [-1., -18. * x[1], -81. * x[2]**2.]])


def f_lin(x):
    """Test function."""
    return np.array([644. * x[0] + 52. * x[1] - 227, 52. * x[0] + 5. * x[1] - 17])


def j_lin(_):
    """Test function."""
    return np.array([[644., 52.], [52., 5.]])


def f3(x):
    """Test function."""
    return np.array([136 - x[0] - x[1]**2. - x[2]**3.,
                     1038 - x[0] - 4. * x[1]**2. - 8. * x[2]**3.,
                     3458 - x[0] - 9. * x[1]**2. - 27. * x[2]**3.,
                     8146 - x[0] - 16. * x[1]**2. - 64 * x[2]**3.])


def j3(x):
    """Test function."""
    return np.array([[-1., -2. * x[1], -3. * x[2]**2.],
                     [-1., -8. * x[1], -24. * x[2]**2.],
                     [-1., -18. * x[1], -81. * x[2]**2.],
                     [-1., -32. * x[1], -192 * x[2]**2.]])


def f4(x):
    """Test function."""
    return np.array([4232. - x[0] - x[1]**2. - x[2]**3. - x[3]**4.,
                     66574. - x[0] - 4. * x[1]**2. - 8. * x[2]**3. - 16. * x[3]**4.,
                     335234. - x[0] - 9. * x[1]**2. - 27. * x[2]**3. - 81. * x[3]**4.])


def j4(x):
    """Test function."""
    return np.array([[-1, -2. * x[1], -3. * x[2]**2., -4. * x[3]**3.],
                     [-1., -8. * x[1], -24. * x[2]**2., -64 * x[3]**3.],
                     [-1., -18. * x[1], -81. * x[2]**2., -324 * x[3]**4.]])


def f5(x):
    """Test function."""
    return np.array([-2. * x[0]**2. - (4. / 3.) * x[1]**3.,
                     -2. * x[0]**2. - (4./3.) * x[1]**3.])


def j5(x):
    """Test function."""
    return np.array([[4. * x[0], 4. * x[1]**2.],
                     [4. * x[0], 4 * x[1]**2.]])


def test_gauss_newton_inputs():
    """Test invalid inputs to gauss_newton."""
    x0 = np.array([3., 4., 5.])
    # Test assertion: func is callable
    assert_raises(TypeError, gauss_newton_solve, "Not Callable", x0, j1)

    # Test assertion: jacob is callable
    assert_raises(TypeError, gauss_newton_solve, f1, x0, "Not Callable")

    # Test assertion: initial guess is one dimensional numpy array
    assert_raises(TypeError, gauss_newton_solve, f1, np.array([[5.]]), j1)
    assert_raises(TypeError, gauss_newton_solve, f1, 5., j1)
    assert_raises(TypeError, gauss_newton_solve, f1, [5.], j1)

    # Test assertion: eps is non-negative, non-zero real number
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, eps="string")
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, eps=1+2j)
    assert_raises(ValueError, gauss_newton_solve, f1, x0, j1, eps=-5.)

    # Test assertion: maxiter is non-negative, non-zero integer
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, maxiter="string")
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, maxxiter=1.11)
    assert_raises(ValueError, gauss_newton_solve, f1, x0, j1, maxiter=-5)
    assert_raises(ValueError, gauss_newton_solve, f1, x0, j1, maxiter=0)

    # Test assertion: method is a string
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, method=None)
    assert_raises(TypeError, gauss_newton_solve, f1, x0, j1, method=5.)

    # Test assertion: method should be analytic, good broyden or bad broyden
    assert_raises(ValueError, gauss_newton_solve, f1, x0, j1, method="goodbadbroyden")


def test_gauss_newton_dimensions():
    """Test invalid function output dimensions in gauss_newton."""
    x0 = np.array([3., 4., 5.])
    # Test assertion: f must be return one-d array
    assert_raises(TypeError, gauss_newton_solve, lambda x: x[0], x0, j1)
    assert_raises(TypeError, gauss_newton_solve, lambda x: np.array([x]), x0, j1)

    # Test assertion: Jac must return two-d array
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: x)
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: np.array([[x]]))

    # Test assertion: Output of Jac must match x0 and f
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: np.array([x[0], x[0]]))
    x0 = np.array([0.] * 3)
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: np.array([[5.] * 2] * 3))
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: np.array([[5.] * 3] * 2))
    assert_raises(TypeError, gauss_newton_solve, f1, x0, lambda x: np.array([[5.] * 3] * 4))


def test_gauss_newton_linear_solve():
    """Test that gauss_newton solves linear systems in 1 step."""
    # Obtained from pg 489 of Numerical Mathematics and Computing Sixth Edition.
    x0 = np.array([100., -200.])
    result = gauss_newton_solve(f_lin, x0, j_lin, eps=1e-20)

    x_expt = np.array([0.4864, -1.6589])
    f_expt = np.array([0., 0.])
    jac_expt = j_lin(1.)
    message_expt = "Convergence obtained."
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'], rtol=1e-4, atol=1e-4)
    assert result['success']
    assert result['message'] == message_expt


def test_gauss_newton_nonlinear_solve():
    """Test that gauss_newton solves nonlinear systems."""
    # The function being optimized is c0 + c1^2 * x^2 + c2^3 * x^3
    # The points are evaluated on [1., 2., 3.]
    x0 = np.array([20., 2.5, 500.])
    result = gauss_newton_solve(f1, x0, j1, maxiter=5000)

    f_expt = np.array([0.] * 3)
    x_expt = np.array([2., 3., 5.])
    message_expt = "Convergence obtained."
    jac_expt = j1(x_expt)
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'])
    assert result['success']
    assert result['message'] == message_expt

    # Because of symmetry the second coefficient of -3 should work
    x0 = np.array([200., -200., 200.])
    result = gauss_newton_solve(f1, x0, j1, maxiter=5000)
    x_expt = np.array([2., -3., 5.])
    jac_expt = j1(x_expt)
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'])
    assert result['success']
    assert result['message'] == message_expt


def test_gauss_newton_nonlinear_overdetermined_solve():
    """Test that gauss_newton solves nonlinear, overdetermined systems."""
    # The function being optimized is c0 + c1^2 * x^2 + c2^3 * x^3
    x0 = np.array([20., 2.5, 500.])
    result = gauss_newton_solve(f3, x0, j3)

    f_expt = np.array([0.] * 4)
    x_expt = np.array([2., 3., 5.])
    message_expt = "Convergence obtained."
    jac_expt = j3(x_expt)
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'])
    assert result['success']
    assert result['message'] == message_expt

    # Because of symmetry the second coefficient of -3 should work
    x0 = np.array([1000., -200., 500.])
    result = gauss_newton_solve(f3, x0, j3)
    x_expt = np.array([2., -3., 5.])
    message_expt = "Convergence obtained."
    jac_expt = j3(x_expt)
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'])
    assert result['success']
    assert result['message'] == message_expt


def test_gauss_newton_nonlinear_underdetermined_solve():
    """Test that gauss_newton solves nonlinear, underdetermined systems."""
    # Non-Exact Initial Guess. Here a really Good Initial guess is needed
    x_expt = np.array([2., 3., 5., 8.])
    f_expt = np.array([0.] * 3)
    x0 = np.array([2.001, 3.001, 5.001, 10.01])
    result = gauss_newton_solve(f4, x0, j4, maxiter=12)
    message_expt = "Maximum number of iterations reached."
    assert result['message'] == message_expt

    # Repeat with a higher number of iterations and a really good initial guess
    # Still it only converges to the first decimal place.
    x0 = np.array([2.00001, 3.00001, 5.00001, 8.00001])
    result = gauss_newton_solve(f4, x0, j4, maxiter=10000)
    jac_expt = j4(result["x"])
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'], rtol=1e-1, atol=1e-1)
    assert result['message'] == "Convergence obtained."
    assert result['success']


def test_gauss_newton_singular_matrix_error():
    """Test that gauss_newton raises error when Jacobian is singular."""
    x0 = np.array([5., 5.])
    result = gauss_newton_solve(f5, x0, j5)
    message_expt = "Singular Jacobian; no solution found."
    assert not result['success']
    assert result["message"] == message_expt
    assert result["eps"] == 1e-6


def test_gauss_newton_approximation_overdetermined():
    """Test gauss newton using an approximation finite diff Jacobian."""
    x0 = np.array([20., 2.5, 500.])
    result = gauss_newton_solve(f3, x0)

    f_expt = np.array([0.] * 4)
    x_expt = np.array([2., 3., 5.])
    message_expt = "Convergence obtained."
    jac_expt = j3(x_expt)
    assert np.allclose(f_expt, result['f'], rtol=1e-5, atol=1e-5)
    assert np.allclose(jac_expt, result['J'], rtol=1e-5, atol=1e-5)
    assert np.allclose(x_expt, result['x'])
    assert result['success']
    assert result['message'] == message_expt
