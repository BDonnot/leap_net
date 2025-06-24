# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net an implementation of the LEAP Net model.

import torch
import torch.nn as nn


class ResNetLayer(nn.Module):
    """
    This layer implements a "ResNet block" in pytorch.
    
    If inputs are `x` the "resnet block" here produce the outputs:
    `y` = `x` + `act( Dense(dim_x)(act(Dense(layer_size)(x))) )`

    This is rather experimental, and any usage of another resnet implementation is probably better suited than this one.
    
    .. note::
        This is the pytorch version.
    """
    def __init__(self,
                 input_shape,
                 units,
                 initializer='glorot_uniform',
                 use_bias=True,
                 trainable=True,
                 name=None,
                 activation="linear",
                 **kwargs):
        super(ResNetLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias = use_bias
        self.units = int(units)
        self.activation = activation
        
        nm_e = "e"
        nm_d = "d"
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_d'.format(self.name)

        self.e = nn.Linear(input_shape,
                           self.units,
                        #    kernel_initializer=self.initializer,
                           use_bias=self.use_bias,
                        #    trainable=self.trainable,
                        #    name=nm_e
                           )
        self.d =  nn.Linear(self.units,
                            input_shape,
                            # kernel_initializer=self.initializer,
                            use_bias=self.use_bias,
                            # trainable=self.trainable,
                            # name=nm_d
                            )

    def call(self, inputs, **kwargs):
        tmp = self.e(inputs)
        if self.activation is not None:
            tmp = getattr(nn, self.activation)(tmp)
        tmp = self.d(tmp)
        if self.activation is not None:
            tmp = getattr(nn, self.activation)(tmp)
        res = torch.add(inputs, tmp)
        return res
