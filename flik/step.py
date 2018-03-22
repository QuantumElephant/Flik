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


r"""Calculate the steps for the different methods."""


import numpy as np


__all__ = [
    "step_linear",
    "step_inv",
    ]


def step_linear(b, A):
    r"""
    Obtain the Newton step for the Jacobian.

    Calculate the roots for the next step of a method that updates the
    (approximated) Jacobian matrix.

    Parameters
    ----------
    b : np.ndarray
        1-dimensional array of the negative of the function evaluated at the
        current guess of the roots -f(x_0).
    A : np.ndarray
        2-dimensional array of the Jacobian evaluated at the current guess of
        the roots J(x_0).
    Returns
    -------
    dx : np.ndarray
        Step length for the method.

    Raises
    ------
    TypeError
        If an argument of an invalid type or shape is passed.
    ValueError
        If an argument passed has an unreasonable value.

    """
    # Solve for dx (A * dx = b)
    try:
        dx = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        b *= -1
        dx = None

    return dx


def step_inv(b, A):
    r"""
    Obtain the Newton step for the inverse of the Jacobian.

    Calculate the roots for the next step of a method that updates the
    inverse of the approximated Jacobian matrix.

    Parameters
    ----------
    b : np.ndarray
        1-dimensional array of the function evaluated at the current guess of
        the roots f(x_0).
    A : np.ndarray
        2-dimensional array of the inverse of the Jacobian evaluated at the i
        current guess of the roots J(x_0).

    Returns
    -------
    dx : np.ndarray
        Step length of the method.

    Raises
    ------
    TypeError
        If an argument of an invalid type or shape is passed.
    ValueError
        If an argument passed has an unreasonable value.

    """
    # Solve for dx (dx = A * b)
    dx = np.dot(A, b)

    return dx
