# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import numpy as np
import warnings


def mape_quantile(y_true, y_pred, multioutput="uniform", quantile=0.1):
    """
    Computes the "MAPE" norm (mean absolute percentage error) but only on the `q` highest values column wise.

    This is a domain specific metric, used for example when we are interested in predicting correctly
    the highest values of a given variable.

    if multioutput is "uniform" then a single number is returned and it consists in taking the average of the previous
    vector.

    Notes
    ------
    This function completely ignores the values where `y_true` are such that `|y_true| < threshold`. It only considers
    values of y_true above the `threshold`.

    Parameters
    ----------
    y_true: ``numpy.ndarray``
        The true values. Each rows is an example, each column is a variable.

    y_pred: ``numpy.ndarray``
        The predicted values. Its shape should match the one from `y_true`

    multioutput: ``str``
        Whether or not to aggregate the returned values

    quantile: ``float``
        The highest ratio to keep. For example, if `quantile=0.1` (default) the 10% highest values are kept.

    Returns
    -------
    mape_percentile: ``float`` or ``numpy.ndarray``
        If `multioutput` is "uniform" it will return a floating point number, otherwise it will return a vector
        with as many component as the number of columns in y_true an y_pred.

    """
    if y_true.shape != y_pred.shape:
        raise RuntimeError("mape can only be computed if y_true and y_pred have the same shape")

    try:
        threshold = float(quantile)
    except Exception as exc_:
        raise exc_
    if threshold < 0.:
        raise RuntimeError("The threshold should be a positive floating point value.")

    if threshold <= 0.:
        raise RuntimeError(f"The quantile `q` should be > 0: {quantile} found")
    if threshold >= 1.:
        raise RuntimeError(f"The quantile `q` should be < 1: {quantile} found")

    index_ok = (np.abs(y_true) > threshold)
    rel_error_ = np.full(y_true.shape, fill_value=np.NaN, dtype=float)
    rel_error_[index_ok] = (y_pred[index_ok] - y_true[index_ok])/y_true[index_ok]
    # what we want to do, but does not always keep the right number of rows
    # (especially when there are equalities...)
    # quantile_ytrue = np.percentile(y_true, q=100.*(1. - quantile), axis=0)
    # rel_error_quantile = rel_error_[(y_true > quantile_ytrue).reshape(y_true.shape)].reshape((-1, y_true.shape[1]))
    # compute how many values to keep
    nb_el_to_keep = int(quantile*y_pred.shape[0])
    nb_el_to_keep = max(nb_el_to_keep, 1)
    # keep the k ith values
    index_highest = np.argpartition(np.abs(y_true), axis=0, kth=-nb_el_to_keep)[-nb_el_to_keep:]
    rel_error_quantile = rel_error_[index_highest, np.arange(y_true.shape[1]).T]
    # compute the mape on these errors
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mape_quantile = np.nanmean(np.abs(rel_error_quantile), axis=0)
    if multioutput == "uniform":
        mape_quantile = np.mean(mape_quantile)
    return mape_quantile
