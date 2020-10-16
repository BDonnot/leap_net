# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import numpy as np


def nrmse(y_true, y_pred, multioutput="uniform"):
    se_ = (y_true - y_pred)**2
    mse = np.mean(se_, axis=0)
    rmse = np.sqrt(mse)
    norm_ = (np.max(y_true, axis=0) - np.min(y_true, axis=0) + 1e-2)
    nrmse_ = rmse / norm_
    if multioutput == "uniform":
        nrmse_ = np.mean(nrmse_)
    return nrmse_
