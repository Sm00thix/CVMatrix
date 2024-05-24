"""
TODO: Write module docstring and signature
"""

from typing import Hashable, Iterable, Union

import numpy as np
from numpy import linalg as la
from numpy import typing as npt


class CVMatrix:
    """
    Implements the fast cross-validation algorithms for kernel matrix-based models
    such as PCA, PCR, PLS, and OLS. The algorithms are based on the following paper by
    O.-C. G. Engstrøm: https://arxiv.org/abs/2401.13185

    Parameters
    ----------
    X : Array-like of shape (N, K) or (N,)
        Predictor variables.
    
    cv_splits : Iterable of Hashable with N elements
        An iterable defining cross-validation splits. Each unique value in `cv_splits`
        corresponds to a different fold.
    
    Y : None or array-like of shape (N, M) or (N,), optional, default=None
        Response variables. If `None`, only :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
        will be computed and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` will not be
        computed. This is useful for models such as PCA and PCR.

    center_X : bool, optional, default=True
        Whether to center `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage.

    center_Y : bool, optional, default=True
        Whether to center `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage. This parameter is ignored if `Y` is
        `None`.

    scale_X : bool, optional, default=True
        Whether to scale `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage.

    scale_Y : bool, optional, default=True
        Whether to scale `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage. This parameter is ignored if `Y` is `None`.

    dtype : data-type, optional, default=np.float64
        The data-type of the arrays used in the computation.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.
    """

    def __init__(
        self,
        X: npt.ArrayLike,
        cv_splits: Iterable[Hashable],
        Y: Union[None, npt.ArrayLike] = None,
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        dtype: npt.DTypeLike = np.float64,
        copy: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.dtype = dtype
        self.copy = copy
        self.X = np.asarray(X, dtype=self.dtype)
        self.Y = None if Y is None else np.asarray(Y, dtype=self.dtype)
        if self.copy and X.dtype == self.dtype:
            self.X = self.X.copy()
        if self.Y is not None and self.copy and Y.dtype == self.dtype:
            self.Y = self.Y.copy()
        self.index_dict = self._generate_validation_indices_dict(cv_splits)

    def training_matrices(self, val_idx: Hashable) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`

        Returns
        -------
        Tuple of Arrays of shape (K, K) and (K, M)
            A tuple containing the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
            and :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` matrices.
        """

    def training_XTX(self, val_idx: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` for a given
        fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`

        Returns
        -------
        Array of shape (K, K)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` matrix.

        Notes
        -----
        This method is useful for models such as PCA and PCR. If you need both
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`, it is more efficient to call
        `training_matrices` instead of calling this method and `training_XTY`
        separately.
        """

    def training_XTY(self, val_idx: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given
        fold.

        Parameters
        ----------
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`

        Returns
        -------
        Array of shape (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` matrix.

        Notes
        -----
        This method is useful for models such as PLS and OLS. If you need both
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`, it is more efficient to call
        `training_matrices` instead of calling this method and `training_XTX`
        separately.
        """

    def _compute_training_x_mean(self, X_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise means of `X` for a given fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.

        Returns
        -------
        Array of shape (1, K)
            The row of column-wise means of `X`.
        """

    def _compute_training_y_mean(self, Y_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise means of `Y` for a given fold.

        Parameters
        ----------
        Y_val : Array of shape (N_val, M)
            The validation set of response variables.

        Returns
        -------
        Array of shape (1, M)
            The row of column-wise means of `Y`.
        """

    def _compute_val_mean(self, X_val_or_Y_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise means of `X` or `Y` for a given validation set.

        Parameters
        ----------
        X_val_or_Y_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of predictor or response variables.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise means of `X` or `Y`.
        """

    def _compute_training_x_std(self, X_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of `X` for a given fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.

        Returns
        -------
        Array of shape (1, K)
            The row of column-wise standard deviations of `X`.
        """

    def _compute_training_y_std(self, Y_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of `Y` for a given fold.

        Parameters
        ----------
        Y_val : Array of shape (N_val, M)
            The validation set of response variables.

        Returns
        -------
        Array of shape (1, M)
            The row of column-wise standard deviations of `Y`.
        """

    def _compute_val_std(self, X_val_or_Y_val: np.ndarray) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of `X` or `Y` for a given
        validation set.

        Parameters
        ----------
        X_val_or_Y_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of predictor or response variables.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise standard deviations of `X` or `Y`.
        """

    def _generate_validation_indices_dict(
        self, cv_splits: Iterable[Hashable]
    ) -> dict[Hashable, npt.NDArray[np.int_]]:
        """
        Generates a list of validation indices for each fold in `cv_splits`.

        Parameters
        ----------
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.

        Returns
        -------
        index_dict : dict of Hashable to Array
            A dictionary mapping each unique value in `cv_splits` to an array of
            validation indices.
        """
        index_dict = {}
        for i, num in enumerate(cv_splits):
            try:
                index_dict[num].append(i)
            except KeyError:
                index_dict[num] = [i]
        for key in index_dict:
            index_dict[key] = np.asarray(index_dict[key], dtype=int)
        return index_dict
