# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import copy
import tensorflow as tf
from tensorflow.keras.layers import (Layer,
                                     Dense,
                                     multiply as tfk_multiply)


class LtauNoAdd(Layer):
    """
    This layer implements the Ltau layer.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.

    Compare to a full Ltau block, this one does not add back the input.
    
    .. warning::
        This is a legacy implementation based on tensorflow_keras (`import tensorflow.keras as keras`)
        which should be avoided and replaced by the most recent LtauNoAdd class (`from leap_net import LtauNoAdd`) 
        that uses the new keras framework, compatible with tensorflow, pytorch AND jax.
        
    """

    def __init__(self,
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
        super(LtauNoAdd, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.use_bias = use_bias
        if penalty_tau is not None:
            if float(penalty_tau) != penalty_tau:
                raise RuntimeError(f"penalty_tau should be a float, you provided {type(penalty_tau)}")
            if penalty_tau < 0.:
                raise RuntimeError(f"penalty_tau should be >= 0.")
            self.penalty_tau = float(penalty_tau)
        else:
            self.penalty_tau = None
        self.e = None
        self.d = None
        self.inter = None
        self._concat = None
        
        if int(nb_unit_per_tau_dim) != nb_unit_per_tau_dim:
            raise RuntimeError(f"nb_unit_per_tau_dim should be an integer, you provided {type(nb_unit_per_tau_dim)}")
        if nb_unit_per_tau_dim <= 0:
            raise RuntimeError(f"nb_unit_per_tau_dim should be >= 1, you provided {nb_unit_per_tau_dim}")
        self.nb_unit_per_tau_dim = nb_unit_per_tau_dim

    def build(self, input_shape):
        is_x, is_tau = input_shape
        nm_e = None
        nm_d = None
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_d'.format(self.name)
        self.e = Dense(is_tau[-1] * self.nb_unit_per_tau_dim,
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       kernel_regularizer=copy.deepcopy(self.kernel_regularizer),
                       bias_regularizer=copy.deepcopy(self.bias_regularizer),
                       activity_regularizer=copy.deepcopy(self.activity_regularizer),
                       name=nm_e)
        self.d = Dense(is_x[-1],
                       kernel_initializer=self.initializer,
                       use_bias=False,
                       trainable=self.trainable,
                       kernel_regularizer=copy.deepcopy(self.kernel_regularizer),
                       bias_regularizer=copy.deepcopy(self.bias_regularizer),
                       activity_regularizer=copy.deepcopy(self.activity_regularizer),
                       name=nm_d)
        
        if self.nb_unit_per_tau_dim != 1:
            self._concat = tf.keras.layers.Concatenate()
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'initializer': self.initializer,
            'use_bias': self.use_bias,
            'trainable': self.trainable,
            'penalty_tau': self.penalty_tau
        })
        return config

    def call(self, inputs, **kwargs):
        x, tau = inputs
        tmp = self.e(x)
        if self.nb_unit_per_tau_dim != 1:
            tau = self._concat([tau for _ in range(self.nb_unit_per_tau_dim)])
        self.inter = tfk_multiply([tau, tmp])  # element wise multiplication
        res = self.d(self.inter)  # no addition of x
        if self.penalty_tau is not None:
            self.add_loss(2. * self.penalty_tau * tf.nn.l2_loss(self.inter))
        return res
