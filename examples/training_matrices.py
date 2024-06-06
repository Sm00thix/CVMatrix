"""
This file demonstrates how to use CVMatrix to compute training set matrices
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
:math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` with possible centering and scaling of
`X` and `Y` using training set means and standard deviations.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

import numpy as np

from cvmatrix.cvmatrix import CVMatrix

if __name__ == '__main__':
    # Create some example data. X must have shape (N, K) or (N,) and Y must have shape
    # (N, M) or (N,). It follows that the number of samples in X and Y must be equal.
    # Y can be None if only XTX is needed.
    X = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]
    )
    Y = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ]
    )

    # The cross-validation folds must be of type Hashable (e.g., int, str, etc.)
    # They must be passed in an iterable of length equal to the number of samples in X
    # (which must in turn be equal to the number of samples in Y, if Y is provided.)
    # The splits are grouped by the values in the iterable. In this case, the first
    # the first samples is in the first fold (fold 0), the second sample is in the
    # second fold (fold "one"), the third and fourth samples are in the third fold
    # (fold 2).
    cv_splits = [0, "one", 2, 2]

    # Create a CVMatrix object with centering and scaling of X and Y. We could have
    # used any of the 16 combinations of centering and scaling. The default is to
    # center and scale both X and Y.
    cvm = CVMatrix(
        cv_splits=cv_splits, center_X=True, center_Y=True, scale_X=True, scale_Y=True
    )

    # Fit the model to the data. This will compute total XTX and XTY matrices.
    # It also computes global statistics that will be reused when determining the
    # centering and scaling of the training set. Only statistics that are relevant for
    # the chosen centering and scaling are computed.
    cvm.fit(X, Y)

    # The unique validation folds and associated indices are stored in the
    # val_folds_dict
    print(f"Validation folds: {cvm.val_folds_dict.keys()}")
    for fold, samples in cvm.val_folds_dict.items():
        print(f"Fold {fold} samples: {samples}")
    print()

    # Compute the training set matrices for each fold.
    print("Training set matrices using training_XTX_XTY:")
    for fold in cvm.val_folds_dict.keys():
        # Notice that fold is the validation fold for which we are computing the
        # training set matrices. The training set matrices are computed using all
        # samples that are not in the validation fold.
        XTX, XTY = cvm.training_XTX_XTY(fold)
        print(f"Fold {fold}:")
        print(f"Training XTX:\n{XTX}")
        print(f"Training XTY:\n{XTY}")
        print()

    # We can also get only XTX or only XTY. However, if both XTX and XTY are needed,
    # it is more efficient to call training_XTX_XTY.
    print("Training set matrices using training_XTX and training_XTY:")
    for fold in cvm.val_folds_dict.keys():
        XTX = cvm.training_XTX(fold)
        print(f"Fold {fold}:")
        print(f"Training XTX:\n{XTX}")
        print()
    for fold in cvm.val_folds_dict.keys():
        XTY = cvm.training_XTY(fold)
        print(f"Fold {fold}:")
        print(f"Training XTY:\n{XTY}")
        print()

    # We can also fit on new X and Y. This will recompute the global statistics and
    # allow us to compute training set matrices for the new data, ensuring that the
    # centering and scaling is done correctly.
    X = np.array(
        [
            [-1, 2, 3],
            [-4, 5, 6],
            [-7, 8, 9],
            [-10, 11, 12]
        ]
    )

    Y = np.array(
        [
            [-1, 2],
            [-3, 4],
            [-5, 6],
            [-7, 8]
        ]
    )

    print("Fitting on new data:")
    cvm.fit(X, Y)
    for fold in cvm.val_folds_dict.keys():
        XTX, XTY = cvm.training_XTX_XTY(fold)
        print(f"Fold {fold}:")
        print(f"Training XTX:\n{XTX}")
        print(f"Training XTY:\n{XTY}")
        print()
