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
        self.X_total = None
        self.Y_total = None
        self.N = None
        self.K = None
        self.M = None
        self.val_index_dict = {}
        self.X_total_mean = None
        self.Y_total_mean = None
        self.XTX_total = None
        self.XTY_total = None
        self.sum_X_total = None
        self.sum_Y_total = None
        self.sum_sq_X_total = None
        self.sum_sq_Y_total = None
        self._init_X_total(X)
        self._init_Y_total(Y)
        self._init_XTX_total()
        self._init_XTY_total()
        self._init_val_indices_dict(cv_splits)
        self._init_total_stats()

    def training_matrices(
            self,
            return_XTX: bool,
            return_XTY: bool,
            val_idx: Hashable
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given fold.

        Parameters
        ----------
        return_XTX : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.
        return_XTY : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.
        val_idx : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`

        Returns
        -------
        Array of shape (K, K) or (K, M) or tuple of arrays of shapes (K, K) and (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and/or
            training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.
        
        Raises
        ------
        ValueError
            If both `return_XTX` and `return_XTY` are `False` or if `return_XTY` is
            `True` and `Y` is `None`.
        """
        if not return_XTX and not return_XTY:
            raise ValueError("At least one of `return_XTX` and `return_XTY` must be True.")
        if return_XTY and self.Y_total is None:
            raise ValueError("Response variables `Y` are not provided.")
        val_indices = self.val_index_dict[val_idx]
        X_val = self.X_total[val_indices]
        kwargs = {}
        if return_XTY:
            Y_val = self.Y_total[val_indices]
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            N_val = val_indices.size
            N_train = self.N - N_val
            N_total_over_N_train = self.N / N_train
            N_val_over_N_train = N_val / N_train
        if self.center_X or self.center_Y or self.scale_X:
            X_train_mean = self._compute_training_X_mean(
                X_val,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if return_XTY and (self.center_X or self.center_Y or self.scale_Y):
            Y_train_mean = self._compute_training_Y_mean(
                Y_val,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if (self.center_X or self.center_Y) and return_XTY:
            kwargs["N_train"]
            kwargs["X_train_mean"] = X_train_mean
            kwargs["Y_train_mean"] = Y_train_mean
        elif self.center_X:
            kwargs["N_train"]
            kwargs["X_train_mean"] = X_train_mean
        if self.scale_X:
            X_train_std = self._compute_training_X_std(
                X_val,
                X_train_mean,
                N_train
            )
            kwargs["X_train_std"] = X_train_std
        if self.scale_Y and return_XTY:
            Y_train_std = self._compute_training_Y_std(
                Y_val,
                Y_train_mean,
                N_train
            )
            kwargs["Y_train_std"] = Y_train_std
        if return_XTX and return_XTY:
            return self._training_XTX(X_val, **kwargs), self._training_XTY(X_val, Y_val, **kwargs)
        elif return_XTX:
            return self._training_XTX(X_val, **kwargs)
        return self._training_XTY(X_val, Y_val, **kwargs)

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
        val_indices = self.val_index_dict[val_idx]
        X_val = self.X_total[val_indices]
        kwargs = {}
        if self.center_X or self.scale_X:
            N_val = val_indices.size
            N_train = self.N - N_val
            N_total_over_N_train = self.N / N_train
            N_val_over_N_train = N_val / N_train
            X_train_mean = self._compute_training_X_mean(
                X_val,
                N_total_over_N_train,
                N_val_over_N_train
            )
            if self.center_X:
                kwargs["N_train"] = N_train
                kwargs["X_train_mean"] = X_train_mean
            if self.scale_X:
                X_train_std = self._compute_training_X_std(
                    X_val,
                    X_train_mean,
                    N_train
                )
                kwargs["X_train_std"] = X_train_std
        return self._training_XTX(X_val, **kwargs)

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
        This method is useful for models such as PLS. If you need both
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`, it is more efficient to call
        `training_matrices` instead of calling this method and `training_XTX`
        separately.
        """
        if self.Y_total is None:
            raise ValueError("Response variables `Y` are not provided.")
        val_indices = self.val_index_dict[val_idx]
        X_val = self.X_total[val_indices]
        Y_val = self.Y_total[val_indices]
        kwargs = {}
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            N_val = val_indices.size
            N_train = self.N - N_val
            N_total_over_N_train = self.N / N_train
            N_val_over_N_train = N_val / N_train
        if self.center_X or self.center_Y or self.scale_X:
            X_train_mean = self._compute_training_X_mean(
                X_val,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if self.center_X or self.center_Y or self.scale_Y:
            Y_train_mean = self._compute_training_Y_mean(
                Y_val,
                N_total_over_N_train,
                N_val_over_N_train
            )
        if self.center_X or self.center_Y:
            kwargs["N_train"]
            kwargs["X_train_mean"] = X_train_mean
            kwargs["Y_train_mean"] = Y_train_mean
        if self.scale_X:
            X_train_std = self._compute_training_X_std(
                X_val,
                X_train_mean,
                N_train
            )
            kwargs["X_train_std"] = X_train_std
        if self.scale_Y:
            Y_train_std = self._compute_training_Y_std(
                Y_val,
                Y_train_mean,
                N_train
            )
            kwargs["Y_train_std"] = Y_train_std
        return self._training_XTY(X_val, Y_val, **kwargs)

    def _init_X_total(self, X: np.ndarray) -> None:
        """
        Initializes the predictor variables.

        Parameters
        ----------
        X : Array of shape (N, K)
            The predictor variables.
        """
        self.X_total = np.asarray(X, dtype=self.dtype)
        if self.copy and X.dtype == self.dtype:
            self.X_total = self.X_total.copy()
        if self.X_total.ndim == 1:
            self.X_total = self.X_total.reshape(-1, 1)
        self.N, self.K = self.X_total.shape
    
    def _init_Y_total(self, Y: np.ndarray) -> None:
        """
        Initializes the response variables.

        Parameters
        ----------
        Y : Array of shape (N, M)
            The response variables.
        """
        if Y is None:
            return
        self.Y_total = np.asarray(Y, dtype=self.dtype)
        if self.copy and Y.dtype == self.dtype:
            self.Y_total = self.Y_total.copy()
        if self.Y_total.ndim == 1:
            self.Y_total = self.Y_total.reshape(-1, 1)
        self.M = self.Y_total.shape[1]
    
    def _init_XTX_total(self) -> None:
        """
        Initializes the total :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` matrix.
        """
        self.XTX_total = self.X_total.T @ self.X_total
    
    def _init_XTY_total(self) -> None:
        """
        Initializes the total :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` matrix.
        """
        if self.Y_total is None:
            return
        self.XTY_total = self.X_total.T @ self.Y_total
    
    def _init_total_stats(self) -> None:
        """
        Initializes the global statistics for `X` and `Y`.
        """
        if self.center_X or self.center_Y or self.scale_X:
            self.X_total_mean = np.mean(self.X_total, axis=0, keepdims=True)
        if (self.center_X or self.center_Y or self.scale_Y) and self.Y_total is not None:
            self.Y_total_mean = np.mean(self.Y_total, axis=0, keepdims=True)
        if self.scale_X:
            self.sum_X_total = np.expand_dims(np.einsum("ij->j", self.X_total), axis=0)
            self.sum_sq_X_total = np.expand_dims(
                np.einsum("ij,ij->j", self.X_total, self.X_total), axis=0
            )
        if self.scale_Y and self.Y_total is not None:
            self.sum_Y_total = np.expand_dims(np.einsum("ij->j", self.Y_total), axis=0)
            self.sum_sq_Y_total = np.expand_dims(
                np.einsum("ij,ij->j", self.Y_total, self.Y_total), axis=0
            )
    
    def _init_val_indices_dict(
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
        val_index_dict = {}
        for i, num in enumerate(cv_splits):
            try:
                val_index_dict[num].append(i)
            except KeyError:
                val_index_dict[num] = [i]
        for key in val_index_dict:
            val_index_dict[key] = np.asarray(val_index_dict[key], dtype=int)
        self.val_index_dict = val_index_dict
    
    def _training_XTX(self, X_val: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` for a given
        fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.
        
        kwargs : dict
            Additional keyword arguments used for potential centering and scaling.

        Returns
        -------
        Array of shape (K, K)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` matrix.
        """
        XTX_train = self.XTX_total - X_val.T @ X_val
        if "X_train_mean" in kwargs:
            X_train_mean = kwargs["X_train_mean"]
            N_train = kwargs["N_train"]
            XTX_train -= N_train * (X_train_mean.T @ X_train_mean)
        if "X_train_std" in kwargs:
            X_train_std = kwargs["X_train_std"]
            XTX_train /= (X_train_std.T @ X_train_std)
        return XTX_train

    def _training_XTY(self, X_val: np.ndarray, Y_val: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given
        fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.
        
        Y_val : Array of shape (N_val, M)
            The validation set of response variables.
        
        kwargs : dict
            Additional keyword arguments used for potential centering and scaling.

        Returns
        -------
        Array of shape (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` matrix.
        """
        XTY_train = self.XTY_total - X_val.T @ Y_val
        if "X_train_mean" in kwargs:
            X_train_mean = kwargs["X_train_mean"]
            Y_train_mean = kwargs["Y_train_mean"]
            N_train = kwargs["N_train"]
            XTY_train -= N_train * (X_train_mean.T @ Y_train_mean)
        if "X_train_std" in kwargs and "Y_train_std" in kwargs:
            X_train_std = kwargs["X_train_std"]
            Y_train_std = kwargs["Y_train_std"]
            return XTY_train / (X_train_std.T @ Y_train_std)
        elif "X_train_std" in kwargs:
            X_train_std = kwargs["X_train_std"]
            return XTY_train / X_train_std.T
        elif "Y_train_std" in kwargs:
            Y_train_std = kwargs["Y_train_std"]
            return XTY_train / Y_train_std
        return XTY_train

    def _compute_training_X_mean(
            self,
            X_val: np.ndarray,
            N_total_over_N_train: float,
            N_val_over_N_train: float
    ) -> np.ndarray:
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
        return (
            N_total_over_N_train * self.X_total_mean
            - N_val_over_N_train * np.mean(X_val, axis=0, keepdims=True)
        )

    def _compute_training_Y_mean(
            self,
            Y_val: np.ndarray,
            N_total_over_N_train: float,
            N_val_over_N_train: float
    ) -> np.ndarray:
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
        return (
            N_total_over_N_train * self.Y_total_mean
            - N_val_over_N_train * np.mean(Y_val, axis=0, keepdims=True)
        )

    def _compute_training_X_std(
            self,
            X_val: np.ndarray,
            X_train_mean,
            N_train: int
    ) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of `X` for a given fold.

        Parameters
        ----------
        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.
        
        X_train_mean : Array of shape (1, K)
            The row of column-wise means of `X` for the training set.

        N_train : int
            The size of the training set.

        Returns
        -------
        Array of shape (1, K)
            The row of column-wise standard deviations of `X`.
        """
        train_sum_X = self.sum_X_total - np.expand_dims(np.einsum("ij->j", X_val), axis=0)
        train_sum_sq_X = self.sum_sq_X_total - np.expand_dims(
            np.einsum("ij,ij->j", X_val, X_val), axis=0
        )
        X_train_std = np.sqrt(
            1
            / (N_train - 1)
            * (
                -2 * X_train_mean * train_sum_X
                + N_train
                * np.einsum("ij,ij -> ij", X_train_mean, X_train_mean)
                + train_sum_sq_X
            )
        )
        X_train_std[X_train_std == 0] = 1
        return X_train_std

    def _compute_training_Y_std(
            self,
            Y_val: np.ndarray,
            Y_train_mean: np.ndarray,
            N_train) -> np.ndarray:
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
        train_sum_Y = self.sum_Y_total - np.expand_dims(
            np.einsum("ij -> j", Y_val), axis=0
        )
        train_sum_sq_Y = self.sum_sq_Y_total - np.expand_dims(
            np.einsum("ij,ij -> j", Y_val, Y_val), axis=0
        )
        Y_train_std = np.sqrt(
            1
            / (N_train - 1)
            * (
                -2 * Y_train_mean * train_sum_Y
                + N_train
                * np.einsum("ij,ij -> ij", Y_train_mean, Y_train_mean)
                + train_sum_sq_Y
            )
        )
        Y_train_std[Y_train_std == 0] = 1