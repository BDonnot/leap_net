# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net an implementation of the LEAP Net model.

# from keras.layers import add as keras_add
import torch
from .ltauNoAdd import LtauNoAdd


class Ltau(LtauNoAdd):
    """
    This layer implements the Ltau layer.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.

    .. note::
        This is the keras >= 3.0 version, compatible with tensorflow, pytorch and jax !

    """
    def __init__(self,
                 input_shape,
                 tau_shape,
                 name=None,
                 initializer='glorot_uniform',
                 use_bias=True,
                 trainable=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 penalty_tau=None,
                 nb_unit_per_tau_dim=1,
                 **kwargs):
        super(Ltau, self).__init__(
                 input_shape,
                 tau_shape,
                 name=name,
                 initializer=initializer,
                 use_bias=use_bias,
                 trainable=trainable,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 penalty_tau=penalty_tau,
                 nb_unit_per_tau_dim=nb_unit_per_tau_dim,
                 **kwargs
                 )

    def forward(self, x):
        inp_, _ = x
        tmp = super().call(x)
        res = torch.add(inp_, x)
        return res
