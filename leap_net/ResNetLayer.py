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

import tensorflow as tf

import pdb


class ResNetLayer(Layer):
    """
    This layer implements the ResNet block

    This is experimental, and any usage of another resnet implementation is probably better suited than this one.

    """
    def __init__(self,
                 units,
                 initializer='glorot_uniform',
                 use_bias=True,
                 trainable=True,
                 name=None,
                 activation=None,
                 **kwargs):
        super(ResNetLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias = use_bias
        self.units = int(units)
        self.activation = activation

        self.e = None
        self.d = None

    def build(self, input_shape):
        nm_e = None
        nm_d = None
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_e'.format(self.name)

        self.e = Dense(self.units,
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       name=nm_e)
        self.d = Dense(input_shape[-1],
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       name=nm_d)

    def call(self, inputs):
        x, tau = inputs
        tmp = self.e(x)
        if self.activation is not None:
            tmp = self.activation(tmp)
        tmp = self.d(tmp)
        if self.activation is not None:
            tmp = self.activation(tmp)
        res = tfk_add([x, tmp])
        return res
