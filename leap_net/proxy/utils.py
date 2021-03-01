# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import warnings

import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error  # mean_absolute_percentage_error
from leap_net.metrics import nrmse
from leap_net.metrics import pearson_r

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Parameters import Parameters


DEFAULT_METRICS = {"MSE_avg": mean_squared_error,
                   "MAE_avg": mean_absolute_error,
                   "NRMSE_avg": nrmse,
                   "pearson_r_avg": pearson_r,
                   "MSE": lambda y_true, y_pred: mean_squared_error(
                       y_true, y_pred,
                       multioutput="raw_values"),
                   "MAE": lambda y_true, y_pred: mean_absolute_error(
                       y_true, y_pred,
                       multioutput="raw_values"),
                   "NRMSE": lambda y_true, y_pred: nrmse(
                       y_true, y_pred,
                       multioutput="raw_values"),
                   "pearson_r": lambda y_true, y_pred: pearson_r(
                       y_true, y_pred,
                       multioutput="raw_values"),
                   }


def limit_gpu_usage():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_parameters():
    # generate the environment
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    param.MAX_SUB_CHANGED = 99999
    param.MAX_LINE_STATUS_CHANGED = 99999
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    return param


def create_env(env_name, use_lightsim_if_available=True):
    """create the grid2op environment with the right parameters and chronics class"""
    backend_cls = None
    if use_lightsim_if_available:
        try:
            from lightsim2grid.LightSimBackend import LightSimBackend
            backend_cls = LightSimBackend
        except ImportError as exc_:
            warnings.warn("You ask to use lightsim backend if it's available. But it's not available on your system.")

    if backend_cls is None:
        from grid2op.Backend import PandaPowerBackend
        backend_cls = PandaPowerBackend

    param = get_parameters()

    env = grid2op.make(env_name,
                       param=param,
                       backend=backend_cls(),
                       chronics_class=MultifolderWithCache
                       )
    return env


def reproducible_exp(env, agent, env_seed=None, chron_id_start=None, agent_seed=None):
    """
    ensure the reproducibility for the data, but NOT for tensorflow

    the environment need to be reset after a call to this method
    """
    if env_seed is not None:
        env.seed(env_seed)

    if chron_id_start is not None:
        # set the environment to start at the right chronics
        env.chronics_handler.tell_id(chron_id_start - 1)

    if agent_seed is not None:
        agent.seed(agent_seed)

