# Copyright (c) 2025, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net an implementation of the LEAP Net model.

import torch
import torch.nn as nn


class LtauNoAdd(nn.Module):
    """
    This layer implements the Ltau layer  (without the final addition) layer in as a pytorch module.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.

    Compare to a full Ltau block, this one does not add back the input.
    
    .. note::
        This is the pytorch version
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
        super().__init__()
        # self.initializer = initializer
        self.use_bias = use_bias
        # self.kernel_regularizer = kernel_regularizer
        # self.bias_regularizer = bias_regularizer
        # self.activity_regularizer = activity_regularizer
        # self.use_bias = use_bias
        if penalty_tau is not None:
            raise RuntimeError("This is not supported for pytorch at the moment, please add it to the loss function instead")
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
        
        if int(nb_unit_per_tau_dim) != nb_unit_per_tau_dim:
            raise RuntimeError(f"nb_unit_per_tau_dim should be an integer, you provided {type(nb_unit_per_tau_dim)}")
        if nb_unit_per_tau_dim <= 0:
            raise RuntimeError(f"nb_unit_per_tau_dim should be >= 1, you provided {nb_unit_per_tau_dim}")
        self.nb_unit_per_tau_dim = nb_unit_per_tau_dim
        
        # build the thing
        nm_e = None
        nm_d = None
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_d'.format(self.name)
        self.e = nn.Linear(input_shape,
                           tau_shape * self.nb_unit_per_tau_dim,
                           bias=self.use_bias
                           )
                    #    kernel_initializer=self.initializer,
                    #    use_bias=self.use_bias,
                    #    trainable=self.trainable,
                    #    kernel_regularizer=copy.deepcopy(self.kernel_regularizer),
                    #    bias_regularizer=copy.deepcopy(self.bias_regularizer),
                    #    activity_regularizer=copy.deepcopy(self.activity_regularizer),
                    #    name=nm_e
                    #    )
        self.d = nn.Linear(tau_shape * self.nb_unit_per_tau_dim,
                           input_shape,
                           bias=False)
                    #    kernel_initializer=self.initializer,
                    #    use_bias=False,
                    #    trainable=self.trainable,
                    #    kernel_regularizer=copy.deepcopy(self.kernel_regularizer),
                    #    bias_regularizer=copy.deepcopy(self.bias_regularizer),
                    #    activity_regularizer=copy.deepcopy(self.activity_regularizer),
                    #    name=nm_d
                    #    )

    def forward(self, x):
        inp_, tau = x
        tmp = self.e(inp_)
        if self.nb_unit_per_tau_dim != 1:
            # duplicate the tau to have as many units as you want
            tau = torch.concat([tau for _ in range(self.nb_unit_per_tau_dim)])
        self.inter = torch.mul(tau, tmp)  # element wise multiplication
        res = self.d(self.inter)  # no addition of x
        if self.penalty_tau is not None:
            raise RuntimeError("This is not supported for pytorch at the moment")
            self.add_loss(2. * self.penalty_tau * keras.ops.sum(keras.ops.square(inp_)))
        return res
