# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add as tfk_add
from tensorflow.keras.layers import multiply as tfk_multiply

import tensorflow as tf

import pdb


class Ltau(Layer):
    """
    This layer implements the Ltau layer.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.


    Examples
    --------


    """
    def __init__(self, initializer='glorot_uniform', use_bias=True, trainable=True, name=None, **kwargs):
        super(Ltau, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias=use_bias
        self.e = None
        self.d = None

    def build(self, input_shape):
        is_x, is_tau = input_shape
        self.e = Dense(is_tau[-1], kernel_initializer=self.initializer, use_bias=self.use_bias, trainable=self.trainable)
        self.d = Dense(is_x[-1], kernel_initializer=self.initializer, use_bias=False, trainable=self.trainable)

    def call(self, inputs):
        x, tau = inputs
        tmp = self.e(x)
        tmp = tfk_multiply([tau, tmp])  # element wise multiplication
        tmp = self.d(tmp)
        res = tfk_add([x, tmp])
        return res