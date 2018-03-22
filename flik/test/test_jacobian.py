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


"""Test file for `flik.jacobian`."""


from nose.tools import assert_raises
import numpy as np
from flik import Jacobian


__all__ = [
    "test_jacobian_inputs",
    "test_update_badbroyden_j1",
    "test_update_goodbroyden_j1",
    "test_update_dfp",
    "test_secant_condition_dfp",
    "test_positive_definiteness_dfp",
    "test_update_inv_bfgs_j1",
    "test_update_approx_bfgs_j3",
    "test_update_inv_bfgs_j3",
    "test_bfgs_secant_condition",
    "test_bfgs_positive_definite",
    ]


# Seed the numpy rng for consistency
np.random.seed(101010101)


# Define some analytical test functions and Jacobians


def f1(x):
    """Test function."""
    y = np.copy(x)
    y **= 2
    y[0] += x[1]
    y[1] -= x[0]
    return y


def j1(x):
    """Test function."""
    y = np.empty((2, 2), dtype=x.dtype)
    y[0, 0] = 2.0 * x[0]
    y[0, 1] = 1.0
    y[1, 0] = -1.0
    y[1, 1] = 2.0 * x[1]
    return y


def f2(x):
    """Test function."""
    return np.array([x[0]**2 + x[1]**2 - 1, 3. * x[0]**2. + x[0] * x[1]**2. + x[2],
                     x[2]**2. + x[1]])


def j2(x):
    """Test function."""
    return np.array([[2. * x[0], 2. * x[1], 0.],
                     [6. * x[0] + x[1]**2., 2. * x[0] * x[1], 1.],
                     [0., 1., 2. * x[2]]])


def f3(x):
    """Test function."""
    return np.array([[x[0]**2 + 2 * x[1] + 5],
                     [np.power(x[0], 3) + 6 * x[0] + 3*x[1]**2]])


def j3(x):
    """Test function."""
    return np.array([[2 * x[0], 2], [6 + 3 * x[0]**2, 6 * x[1]]])


def test_jacobian_inputs():
    """Test invalid inputs to Jacobian class."""
    # Test adding a non-callable jacobian to the class.
    assert_raises(TypeError, Jacobian, jac=5.)
    # Test Jacobian calls the correct function.
    jac_obj = Jacobian(jac=j1)
    x_0 = np.random.rand(2)
    assert np.allclose(jac_obj(x_0), j1(x_0))


def test_update_goodbroyden_j1():
    """Test Jacobian.update_goodbroyden using analytical Jacobian and function."""
    # Initial point x_k
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Jacobian at initial x_k
    j1_0 = j1(x_0)

    # Define arbitrary step
    dx = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    # Compute analytical Jacobian at x_{k+1}`
    j1_1 = j1(x_1)
    delta_f = f1_1 - f1_0
    expected_ans = j1_0 + \
        np.outer((delta_f - j1_0.dot(dx)) / np.dot(dx, dx), dx)

    # Approximate Jacobian at (x_0 + dx) with Good Broyden method
    # from Jacobian and vector function at initial x_0
    actual_ans = Jacobian.update_goodbroyden(j1_0, f1_0, f1_1, x_0, x_1)

    assert np.allclose(expected_ans, actual_ans)
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_update_badbroyden_j1():
    """Test Jacobian.update_badbroyden against analytical Jacobian and function."""
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Inverse Jacobian at initial x_0
    j1_inv_0 = np.linalg.inv(j1(x_0))

    # Define arbitrary step and take step
    dx = np.asarray([0.01, 0.01])
    x_1 = x_0 + dx

    # Function at x_{k+1}
    f1_1 = f1(x_1)
    delta_f = f1_1 - f1_0
    # Compute analytical inverse Jacobian at `x_0 + dx`
    expected_ans = j1_inv_0 + np.outer((dx - j1_inv_0.dot(delta_f)) /
                                       np.dot(delta_f, delta_f), delta_f)

    # Approximate inverse Jacobian at (x_0 + dx) with Bad Broyden method
    # from inverse Jacobian and vector function at initial x_0
    actual_ans = Jacobian.update_badbroyden(j1_inv_0, f1_0, f1_1, x_0, x_1)

    assert np.allclose(expected_ans, actual_ans)
    # assert that the approximation jacobian matches the analytic jacobian.
    j1_1 = np.linalg.inv(j1(x_1))
    assert np.allclose(j1_1, actual_ans, atol=1e-1)


def test_update_dfp():
    """Test the definition of DFP."""
    # Test wikipedia definition/format.
    random_vecs = np.random.rand(10, 3) * 50. + 0.5
    x = random_vecs[0]
    dx = np.array([0.0001, 0.0001, 0.0001])
    df = f2(x + dx) - f2(x)
    b0 = j2(x)

    gamma = 1. / np.dot(df, dx)
    actual_answer = (np.eye(3)
                     - gamma * np.outer(df, dx)).dot(b0).dot(np.eye(3)
                                                             - gamma * np.outer(dx, df))
    actual_answer += gamma * np.outer(df, df)
    expected_answer = Jacobian.update_dfp(b0, f2(x), f2(x + dx), x, x + dx)
    assert np.allclose(actual_answer, expected_answer)


def test_positive_definiteness_dfp():
    """Test that DFP is positive definite."""
    # It's crucial that the x values are greater than 0.5
    # or else positive definiteness is not required.
    random_vecs = np.random.rand(10, 3) * 50. + 0.5

    dx = np.array([0.5, 0.5, 0.5])
    x = random_vecs[0]

    dfp_update = Jacobian.update_dfp(np.eye(3), f2(x), f2(x + dx), x, x + dx)
    diff = np.asarray([np.dot(x, dfp_update.dot(x)) for x in random_vecs])
    assert np.all(diff > 0.)


def test_secant_condition_dfp():
    """Test that dfp update satisfies secant condition."""
    x = (np.random.rand(1, 3) * 50. - 10.)[0]
    dx = np.array([0.5, 0.5, 0.5])
    df = f2(x + dx) - f2(x)
    dfp_update = Jacobian.update_dfp(np.eye(3), f2(x), f2(x + dx), x, x + dx)
    assert np.allclose(df, dfp_update.dot(dx))


def test_update_sr1_j1():
    """Test Jacobian.update_sr1 using analytical Jacobian and function."""
    # Initial point x_k
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Jacobian at initial x_k
    j1_0 = j1(x_0)

    # Define arbitrary step
    dx = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + dx
    # Function at x_{k+1}
    f1_1 = f1(x_1)
    # Compute analytical Jacobian at x_{k+1}`
    j1_1 = j1(x_1)
    delta_f = f1_1 - f1_0
    tmp = delta_f - j1_0.dot(dx)
    expected_ans = j1_0 + (np.outer(tmp, tmp.T)) / np.dot(tmp.T, dx)

    # Approximate Jacobian at (x_0 + dx) with Good Broyden method
    # from Jacobian and vector function at initial x_0
    actual_ans = Jacobian.update_sr1(j1_0, f1_0, f1_1, x_0, x_1)

    assert np.allclose(expected_ans, actual_ans)
    assert np.allclose(j1_1, expected_ans, atol=1e-1)


def test_update_sr1_inv_j1():
    """Test Jacobian.update_sr1_inv against analytical Jacobian and function."""
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f1_0 = f1(x_0)
    # Inverse Jacobian at initial x_0
    j1_inv_0 = np.linalg.inv(j1(x_0))

    # Define arbitrary step and take step
    dx = np.asarray([0.01, 0.01])
    x_1 = x_0 + dx

    # Function at x_{k+1}
    f1_1 = f1(x_1)
    delta_f = f1_1 - f1_0
    # Compute analytical inverse Jacobian at `x_0 + dx`
    tmp = dx - j1_inv_0.dot(delta_f)
    expected_ans = j1_inv_0 + (np.outer(tmp, tmp.T)) / np.dot(tmp.T, delta_f)

    # Approximate inverse Jacobian at (x_0 + dx) with Bad Broyden method
    # from inverse Jacobian and vector function at initial x_0
    actual_ans = Jacobian.update_sr1_inv(j1_inv_0, f1_0, f1_1, x_0, x_1)

    assert np.allclose(expected_ans, actual_ans)
    # assert that the approximation jacobian matches the analytic jacobian.
    j1_1 = np.linalg.inv(j1(x_1))
    assert np.allclose(j1_1, actual_ans, atol=1e-1)


def test_update_approx_bfgs_j3():
    """Test Jacobian.uupdate_approx_bfgs using bfgs."""
    # Initial point x_0
    # x_0 = np.random.rand(2)
    x_0 = np.array([0.8, 0.3])
    # Function at initial x_k
    f3_0 = f3(x_0)
    # Jacobian at initial x_k
    j3_0 = j3(x_0)

    # Define arbitrary step
    delta_x = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + delta_x
    # Function at `x_{k+1}`
    f3_1 = f3(x_1)

    # delta_f
    delta_f = f3_1 - f3_0
    # Jacobian at `x_{k}+1`
    j3_1 = j3(x_1)
    # Compute analytical Jacobian at `x_{k+1}`
    temp = np.dot(j3_0, delta_x)
    expected_ans = j3_0 \
        + np.outer(delta_f, delta_f.transpose())/np.inner(delta_f.transpose(), delta_x) \
        - np.outer(temp, temp.transpose())/np.inner(delta_x.transpose(), temp)
    # Compute the approximated Jacobian at `x_{k+1}`
    actual_ans = Jacobian.update_approx_bfgs(B_k=j3_0, f_k=f3_0, f_k1=f3_1, x_k=x_0, x_k1=x_1)

    # print("np.linalg.inv(j3_1)", np.linalg.inv(j3_1))
    assert np.allclose(expected_ans, actual_ans)
    assert np.allclose(j3_1, expected_ans, atol=1e-1)


def test_update_inv_bfgs_j1():
    """Test Jacobian.update_inv_bfgs_bfgs using inverse BFGS Jacobian and  function."""
    # Initial point x_0
    # x_0 = np.random.rand(2)
    x_0 = np.array([0.8, 0.3])
    # Function at initial `x_k`
    f1_0 = f1(x_0)
    # Jacobian at initial `x_k`
    j1_0 = j1(x_0)
    # Inverse Jacobian at `x_k`
    Binv_0 = np.linalg.inv(j1_0)

    # Define arbitrary step
    delta_x = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + delta_x
    # Function at `x_{k+1}`
    f1_1 = f1(x_1)

    # delta_f
    delta_f = f1_1 - f1_0
    # Jacobian at `x_{k+1}`
    # j1_1 = j1(x_1)
    # Inverse Jacobian at `x_{k+1}`
    # Binv_1 = np.linalg.inv(j1_1)

    # Compute analytical Jacobian at `x_{k+1}`
    a1 = np.eye(2) - np.outer(delta_x, delta_f.transpose())/np.inner(delta_f. transpose(), delta_x)
    a2 = np.eye(2) - np.outer(delta_f, delta_x.transpose())/np.inner(delta_f. transpose(), delta_x)
    expected_ans = np.dot(a1, np.dot(Binv_0, a2)) \
        + np.outer(delta_x, delta_x. transpose())/np.inner(delta_f.transpose(), delta_x)
    # Compute the approximated inverse Jacobian at `x_{k+1}`
    actual_ans = Jacobian.update_inv_bfgs(Binv_k=Binv_0, f_k=f1_0, f_k1=f1_1, x_k=x_0, x_k1=x_1)

    assert np.allclose(expected_ans, actual_ans)
    # assert np.allclose(Binv_1, expected_ans, atol=1e-1)


def test_update_inv_bfgs_j3():
    """Test Jacobian.update_uinv_bfgs_bfgs using inverse BFGS Jacobian and  function."""
    # Initial point x_0
    # x_0 = np.random.rand(2)
    x_0 = np.array([0.8, 0.3])
    # Function at initial `x_k`
    f3_0 = f3(x_0)
    # Jacobian at initial `x_k`
    j3_0 = j3(x_0)
    # Inverse Jacobian at `x_k`
    Binv_0 = np.linalg.inv(j3_0)

    # Define arbitrary step
    delta_x = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + delta_x
    # Function at `x_{k+1}`
    f3_1 = f3(x_1)

    # delta_f
    delta_f = f3_1 - f3_0
    # Jacobian at `x_{k+1}`
    # j3_1 = j3(x_1)
    # Inverse Jacobian at `x_{k+1}`
    # Binv_1 = np.linalg.inv(j3_1)

    # Compute analytical Jacobian at `x_{k+1}`
    a1 = np.eye(2) - np.outer(delta_x, delta_f.transpose())/np.inner(delta_f. transpose(), delta_x)
    a2 = np.eye(2) - np.outer(delta_f, delta_x.transpose())/np.inner(delta_f. transpose(), delta_x)
    expected_ans = np.dot(a1, np.dot(Binv_0, a2)) \
        + np.outer(delta_x, delta_x.transpose())/np.inner(delta_f.transpose(), delta_x)
    # Compute the approximated inverse Jacobian at `x_{k+1}`
    actual_ans = Jacobian.update_inv_bfgs(Binv_k=Binv_0, f_k=f3_0, f_k1=f3_1, x_k=x_0, x_k1=x_1)

    assert np.allclose(expected_ans, actual_ans)
    # assert np.allclose(Binv_1, expected_ans, atol=1e-1)


def test_bfgs_positive_definite():
    """Test that BFGS is positive definite."""
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Initial point x_0
    x_0 = np.random.rand(2)
    # Function at initial x_k
    f3_0 = f3(x_0)
    # Jacobian at initial x_k
    # j3_0 = j3(x_0)
    # Initial Jacobian guess
    j3_0 = np.eye(2)

    # Define arbitrary step
    delta_x = np.asarray([0.01, 0.01])
    # Take step dx
    x_1 = x_0 + delta_x
    # Function at `x_{k+1}`
    f3_1 = f3(x_1)

    # delta_f
    delta_f = f3_1 - f3_0
    # Jacobian at `x_{k}+1`
    # j3_1 = j3(x_1)
    # Updated Jacobian by approx bfgs
    j3_upd = Jacobian.update_approx_bfgs(
        B_k=j3_0, f_k=f3_0, f_k1=f3_1, x_k=x_0, x_k1=x_1)
    # Inverse of updated jacobian
    j3_upd = np.linalg.inv(j3_upd)

    assert np.inner(delta_f.transpose(), delta_x) > 0
    assert np.all(np.linalg.eigvals(j3_upd) > 0)


def test_bfgs_secant_condition():
    """Test the secant condition for BFGS."""
    x = (np.random.rand(1, 3) * 50. - 10.)[0]
    dx = np.array([0.5, 0.5, 0.5])
    df = f2(x + dx) - f2(x)
    # Compute the updated bfgs Jacobian
    bfgs_update = Jacobian.update_approx_bfgs(
        np.eye(3), f2(x), f2(x + dx), x, x + dx)
    assert np.allclose(df, bfgs_update.dot(dx))
