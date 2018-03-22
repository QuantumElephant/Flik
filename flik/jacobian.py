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

r"""Contains the jacobian master class for all jacobian methods."""

import numpy as np

__all__ = ["Jacobian"]


class Jacobian:
    r"""
    Analytic Jacobian class.

    Holds as a parent class for the FiniteDiffJacobian class,
    which is intended for approximation-based Jacobian methods.

    Includes static methods, update_goodbroyden and
    update_badbroyden, intended to be used to update the jacobian.

    A callable object that calls the jacobian of a function,
    all given by the user.

    """

    def __init__(self, jac):
        r"""
        Construct an analytic Jacobian class for a function f.

        Also used as an initializer for the FiniteDiffJacobian objects.

        Parameters
        ----------
        jac : callable, optional
            The Jacobian of some function as a callable object.
            Used in the case that the Jacobian is known.

        Raises
        ------
        TypeError
            If an argument of an invalid type or shape is passed.
        ValueError
            If an argument passed has an unreasonable value.

        """
        if jac is not None:
            if not callable(jac):
                raise TypeError("J must be a callable object")
            self._jac = jac

    def __call__(self, x):
        r"""
        Compute the Jacobian analytically for the function 'f'.

        Parameters
        ----------
        x : np.ndarray
            Argument vector to the analytical Jacobian function.

        Returns
        -------
        np.ndarray
            Value of the analytic Jacobian at position `x`.

        Raises
        ------
        TypeError
            If Jacobian was never provided to the constructor to be called,
            or argument vector is not the same dimension as the jacobian.

        """
        return self._jac(x)

    def update_analytic(self, _dummy, _dummy2, _dummy3, _dummy4, x_k1):
        r"""
        Analytically evaluate the Jacobian at new solution vector `x_{k+1}`.

        Parameters
        ----------
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        B_k1 : np.ndarray
            Jacobian matrix at current solution vector.

        """
        B_k1 = self(x_k1)
        return B_k1

    @staticmethod
    def update_goodbroyden(B_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate Jacobian at `x + dx` using Good Broyden update.

        Parameters
        ----------
        B_k : np.ndarray
            Previous step jacobian.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        B_upd : np.ndarray
            God Broyden approximation to the jacobian.

        """
        # Evaluate df
        delta_f = f_k1 - f_k
        # Evaluate dx
        delta_x = x_k1 - x_k

        # Compute Good Broyden right hand side second term numerator
        f_j = delta_f - np.dot(B_k, delta_x)
        # Divide by delta_x norm
        f_j /= np.dot(delta_x, delta_x)
        # Compute matrix from dot product of f_j and transposed delta_x
        B_upd = np.outer(f_j, delta_x.transpose())
        # Update Jacobian
        B_upd += B_k

        return B_upd

    @staticmethod
    def update_badbroyden(Binv_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate inverse Jacobian at `x + dx` using Bad Broyden update.

        Parameters
        ----------
        Binv_k : np.ndarray
            Inverse of previous step jacobian.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        Binv_upd : np.ndarray
            Bad Broyden approximation to the inverse jacobian.

        """
        # Evaluate df
        delta_f = f_k1 - f_k
        # Evaluate dx
        delta_x = x_k1 - x_k

        # Compute Good Broyden right hand side second term numerator
        f_j = delta_x - np.dot(Binv_k, delta_f)
        # Divide by delta_x norm
        f_j /= np.dot(delta_f, delta_f)
        # Compute matrix from dot product of f_j and transposed delta_x
        B_upd = np.outer(f_j, delta_f.transpose())
        # Update Jacobian
        B_upd += Binv_k

        return B_upd

    @staticmethod
    def update_dfp(B_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate Jacobian at `x + dx` using Davidson-Fletcher-Powell (DFP) update.

        Parameters
        ----------
        B_k : np.ndarray
            Previous step jacobian.
            Jacobian from previous step.

        """
        delta_f = f_k1 - f_k
        delta_x = x_k1 - x_k
        norm = np.dot(delta_x, delta_f)

        factor1 = np.outer(B_k.dot(delta_x), delta_f) \
            + np.outer(delta_f, B_k.transpose().dot(delta_x))
        factor1 /= norm
        factor2 = (1. + (delta_x.dot(B_k.dot(delta_x))/norm))
        factor2 *= np.outer(delta_f, delta_f)
        return B_k - factor1 + (factor2 / norm)

    @staticmethod
    def update_sr1(B_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate Jacobian at `x + dx` using SR1 update.

        Parameters
        ----------
        B_k : np.ndarray
            Previous step jacobian.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        B_upd : np.ndarray
            SR1 approximation to the jacobian.

        """
        # Evaluate df
        delta_f = f_k1 - f_k
        # Evaluate dx
        delta_x = x_k1 - x_k

        # Compute SR1 right hand side second term (df - B_k*dx)
        f_j = delta_f - np.dot(B_k, delta_x)
        # Compute matrix for dot product of f_j and transposed f_j
        B_upd = np.outer(f_j, f_j.transpose())
        # Divide tmp by dot product of transposed f_j and delta_x
        B_upd /= np.dot(f_j.transpose(), delta_x)
        # Update Jacobian
        B_upd += B_k

        return B_upd

    @staticmethod
    def update_sr1_inv(Binv_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate inverse Jacobian at `x + dx` using SR1 update.

        Parameters
        ----------
        Binv_k : np.ndarray
            Inverse of previous step jacobian.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        Binv_upd : np.ndarray
            SR1 approximation to the inverse jacobian.

        """
        # Evaluate df
        delta_f = f_k1 - f_k
        # Evaluate dx
        delta_x = x_k1 - x_k

        # Compute SR1 right hand side second term (dx - Binv_k*df)
        f_j = delta_x - np.dot(Binv_k, delta_f)
        # Compute matrix for dot product of f_j and transposed f_j
        B_upd = np.outer(f_j, f_j.transpose())
        # Divide tmp by dot product of transposed f_j and delta_f
        B_upd /= np.dot(f_j.transpose(), delta_f)
        # Update Jacobian
        B_upd += Binv_k

        return B_upd

    @staticmethod
    def update_approx_bfgs(B_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate Jacobian at `x + dx` using Broyden–Fletcher–Goldfarb–Shanno (BFGS) update.

        The algorithm is based on

        .. math::
            B_{k+1} = B_k
                    + \frac{ f_k  {f_k}^{\top} }
                           { {f_k}^{\top} {\Delta x}_k }
                    - \frac{ {B_k} {\Delta x}_k { ( B_k {\Delta x}_k ) }^{\top} }
                           { { {\Delta x}^{\top}_k } B_k {\Delta x}_k }

        Parameters
        ----------
        B_k : np.ndarray
            Jacobian from previous step.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        Bk_1 : np.ndarray
            BFGS approximation to the jacobian.

        """
        # Evaluate df
        delta_f = f_k1 - f_k
        # Evaluate dx
        delta_x = x_k1 - x_k

        # Compute the second term of BFGS formula in the right side
        bfgs_term2 = np.outer(delta_f, delta_f.transpose()) / \
            np.inner(delta_f.transpose(), delta_x)
        # Compute the `B_k {\Delta x}_k` term
        Bx = np.dot(B_k, delta_x)
        # Compute the third term of BFGS formula in the right side
        bfgs_term3 = np.outer(Bx, Bx.transpose()) / \
            np.inner(delta_x.transpose(), Bx)
        # Update the Jacobian `B_{k+1}`
        B_k1 = B_k + bfgs_term2 - bfgs_term3

        return B_k1

    @staticmethod
    def update_inv_bfgs(Binv_k, f_k, f_k1, x_k, x_k1):
        r"""
        Approximate inverse Jacobian at `x + dx` using BFGS update.

        The formula is

        .. math::
        H_{k+1} = (
                    \mathcal{I} - \frac{ {\Delta x}_k {f_k}^{\top} }
                                       { {f_k}^{\top} {\Delta x}_k }
                                       )
                    H_{k}
                    (\mathcal{I} -  \frac{ f_k  {{\Delta x}^{\top}}_k }
                                         { {f_k}^{\top} {\Delta x}_k })
                + \frac{ {\Delta x}_k {\Delta x}_k^{\top } }
                       { {f_k}^{\top} {\Delta x}_k }

        where :math:`H_k = {B_k}^{-1}` is the inverse of approximated BFGS Jacobian.

        Parameters
        ----------
        Binv_k : np.ndarray
            Inverse of previous step jacobian.
        f_k : np.ndarray
            Vector function evaluated at previous step `x_k`.
        f_k1: np.ndarray
            Vector function evaluated at current step `x_{k+1}`.
        x_k : np.ndarray
            Solution of nonlinear system of equations at previous step.
        x_k1: np.ndarray
            Solution of nonlinear system of equations at current step.

        Returns
        -------
        Binv_upd : np.ndarray
            BFGS approximation to the inverse jacobian.

        """
        # Evaluate dx
        delta_x = x_k1 - x_k
        # Evaluate dfx
        delta_f = f_k1 - f_k

        # Compute the right part in the first parentheses
        fac1 = np.outer(delta_x, delta_f.transpose()) / np.inner(delta_f.transpose(), delta_x)
        # Compute the first parentheses part
        paren1 = np.eye(fac1.shape[0]) - fac1

        # Compute the right part in the second parentheses
        fac2 = np.outer(delta_f, delta_x.transpose()) / np.inner(delta_f.transpose(), delta_x)
        # Compute the second parentheses part
        paren2 = np.eye(fac2.shape[0]) - fac2
        # Compute the first term of the entire formula

        Binv_upd = np.dot(paren1, np.dot(Binv_k, paren2))
        # Update the inverse Jacobian `{B^{-1}}_{k+1}`
        Binv_upd += np.outer(delta_x, delta_x.transpose()) / \
            np.inner(delta_f.transpose(), delta_x)

        return Binv_upd
