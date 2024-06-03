"""
TODO: Write module docstring
"""

from itertools import product
from typing import Union, Iterable, Hashable

import numpy as np
import pandas as pd

from numpy import typing as npt
from numpy.testing import assert_allclose
from . import load_data
from cvmatrix.cvmatrix import CVMatrix
from .naive_cvmatrix import NaiveCVMatrix

class TestClass:
    """
    Class for testing the CVMatrix implementation.

    This class contains methods for testing the CVMatrix class. In particular, this
    class tests for equivalency between the naive, straight-forward algorithms
    implemented in the NaiveCVMatrix class and the fast algorithms implemented in the
    CVMatrix class. The tests are performed on a real dataset of NIR spectra and ground
    truth values for 8 different grain varieties, protein, and moisture. The dataset is
    publicly available on GitHub and originates from the articles by Dreier et al. and
    Engstrøm et al. See the load_data module for more information about the dataset.
    """
    csv = load_data.load_csv()
    raw_spectra = load_data.load_spectra()

    def load_X(self) -> npt.NDArray[np.float_]:
        """
        Loads the raw spectral data.

        Returns
        -------
        npt.NDArray[np.float_]
            A copy of the raw spectral data.
        """
        return np.copy(self.raw_spectra)

    def load_Y(self, names: list[str]) -> npt.NDArray[np.float_]:
        """
        Loads target values based on the specified column names.

        Parameters
        ----------
        names : list[str]
            The names of the columns to load.
        
        Returns
        -------
        npt.NDArray[np.float_]
            A copy of the target values.
        """
        return self.csv[names].to_numpy()

    def fit_models(
            self,
            X: npt.ArrayLike,
            Y: Union[None, npt.ArrayLike],
            center_X: bool,
            center_Y: bool,
            scale_X: bool,
            scale_Y: bool,
            dtype: npt.DTypeLike
    ) -> tuple[NaiveCVMatrix, CVMatrix]:
        """
        Fits the NaiveCVMatrix and CVMatrix models.

        Parameters
        ----------
        X : npt.ArrayLike
            The predictor variables.

        Y : Union[None, npt.ArrayLike]
            The response variables.

        center_X : bool
            Whether to center `X`.

        center_Y : bool
            Whether to center `Y`.

        scale_X : bool
            Whether to scale `X`.

        scale_Y : bool
            Whether to scale `Y`.
        
        dtype : npt.DTypeLike
            The data-type of the arrays used in the computation.

        Returns
        -------
        tuple[NaiveCVMatrix, CVMatrix]
            A tuple containing the NaiveCVMatrix and CVMatrix models.
        """
        naive = NaiveCVMatrix(center_X, center_Y, scale_X, scale_Y, dtype)
        fast = CVMatrix(center_X, center_Y, scale_X, scale_Y, dtype)
        naive.fit(X, Y)
        fast.fit(X, Y)
        return naive, fast
    
    def load_cv_splits(
            self,
            naive: NaiveCVMatrix,
            fast: CVMatrix,
            cv_splits: Iterable[Hashable]
    ) -> None:
        """
        Loads cross-validation splits into the NaiveCVMatrix and CVMatrix models.

        Parameters
        ----------
        naive : NaiveCVMatrix
            The NaiveCVMatrix model.

        fast : CVMatrix
            The CVMatrix model.

        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.
        """
        naive.load_cv_splits(cv_splits)
        fast.load_cv_splits(cv_splits)
    
    def check_equivalent_matrices(
            self,
            naive: NaiveCVMatrix,
            fast: CVMatrix,
            cv_splits: Iterable[Hashable]
    ) -> None:
        """
        Checks if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent.

        Parameters
        ----------
        naive : NaiveCVMatrix
            The NaiveCVMatrix model.

        fast : CVMatrix
            The CVMatrix model.
        
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.
        """
        
        for val_fold in cv_splits:
            if naive.Y_total is not None:
                # Check if the matrices are equivalent for the training_XTX_XTY method
                # between the NaiveCVMatrix and CVMatrix models.
                naive_XTX, naive_XTY = naive.training_XTX_XTY(val_fold)
                fast_XTX, fast_XTY = fast.training_XTX_XTY(val_fold)
                assert_allclose(fast_XTX, naive_XTX)
                assert_allclose(fast_XTY, naive_XTY)
                # Check if the matrices are equivalent for the training_XTX and
                # training_XTY methods between the NaiveCVMatrix and CVMatrix models.
                # Also check if the matrices are equivalent for the training_XTX,
                # training_XTY, and training_XTX_XTY methods.
                direct_naive_XTX = naive.training_XTX(val_fold)
                direct_fast_XTX = fast.training_XTX(val_fold)
                direct_naive_XTY = naive.training_XTY(val_fold)
                direct_fast_XTY = fast.training_XTY(val_fold)
                assert_allclose(direct_fast_XTX, direct_naive_XTX)
                assert_allclose(direct_fast_XTY, direct_naive_XTY)
                assert_allclose(direct_fast_XTX, fast_XTX)
                assert_allclose(direct_fast_XTY, fast_XTY)
            else:
                # Check if the matrices are equivalent for the training_XTX method
                # between the NaiveCVMatrix and CVMatrix models.
                naive_XTX = naive.training_XTX(val_fold)
                fast_XTX = fast.training_XTX(val_fold)
                assert_allclose(fast_XTX, naive_XTX)
    
    def test_equivalent_matrices_basic(self):
        """
        Tests if the matrices computed by the NaiveCVMatrix and CVMatrix models are
        equivalent for basic settings.
        """
        X = self.load_X()
        Y = self.load_Y(["Protein", "Moisture"])
        cv_splits = self.load_Y(["split"]).squeeze()
        # X = np.array([[1, 2], [3, 4], [5, 6]])
        # Y = np.array([[7, 8], [9, 10], [11, 12]])
        # cv_splits = np.array([0, 1, 2])
        center_X = True
        center_Y = True
        scale_X = True
        scale_Y = True
        naive, fast = self.fit_models(X, Y, center_X, center_Y, scale_X, scale_Y, np.float64)
        self.load_cv_splits(naive, fast, cv_splits)
        self.check_equivalent_matrices(naive, fast, cv_splits)