# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import re
import os
import warnings

import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error  # mean_absolute_percentage_error
from leap_net.metrics import nrmse
from leap_net.metrics import pearson_r

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Parameters import Parameters

from leap_net.ResNetLayer import ResNetLayer
from leap_net.agents import RandomNN1
from leap_net.proxy.AgentWithProxy import AgentWithProxy
from leap_net.proxy.ProxyLeapNet import ProxyLeapNet

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
        from grid2op.PandaPowerBackend import PandaPowerBackend
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


def main(limit_gpu_memory=True,
         total_train=int(1024)*int(1024),
         env_name="l2rpn_case14_sandbox",
         save_path="model_saved",
         save_path_tensorbaord="tf_logs",
         env_seed=42,
         agent_seed=1,
         use_lightsim_if_available=True,
         val_regex=".*99[0-9].*",
         actor_class=RandomNN1,
         # perform an evaluation on the training set at the end of training
         eval_training_set=int(1024) * int(128),  # number of powerflow the proxy will do after training
         pred_batch_size=int(1024) * int(128),  # number of powerflow that will be done by the proxy "at once"
         save_path_final_results="model_results",  # where the information about the prediction will be stored
         metrics=DEFAULT_METRICS,  # which metrics are used to evaluate the performance of the model
         verbose=1,  # do I print the results of the model
         # proxy part
         model_name="leapnet_case_14",
         sizes_enc=(20,),
         sizes_main=(150, 150),
         sizes_out=(40,),
         lr=3e-4,
         layer=ResNetLayer,
         layer_act=None,
         scale_main_layer=None,
         scale_input_dec_layer=None,
         scale_input_enc_layer=None,
         attr_x=("prod_p", "prod_v", "load_p", "load_q"),
         attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
         attr_tau=("line_status",),
         ):
    if limit_gpu_memory:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices):
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # create the environment
    env = create_env(env_name, use_lightsim_if_available=use_lightsim_if_available)

    # I select only part of the data, for training
    # env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
    # env.chronics_handler.real_data.reset()
    obs = env.reset()

    # now train the agent
    actor = actor_class(env.action_space, p=0.5)
    reproducible_exp(env, actor, env_seed=env_seed, agent_seed=agent_seed)

    proxy = ProxyLeapNet(name=model_name,
                         lr=lr,
                         layer=layer,
                         layer_act=layer_act,
                         sizes_enc=sizes_enc,
                         sizes_main=sizes_main,
                         sizes_out=sizes_out,
                         scale_main_layer=scale_main_layer,
                         scale_input_dec_layer=scale_input_dec_layer,
                         scale_input_enc_layer=scale_input_enc_layer,
                         attr_x=attr_x,
                         attr_tau=attr_tau,
                         attr_y=attr_y)
    agent_with_proxy = AgentWithProxy(actor,
                                      proxy=proxy,
                                      logdir=save_path_tensorbaord
                                      )

    # train it
    agent_with_proxy.train(env,
                           total_train,
                           save_path=save_path
                           )

    if verbose:
        print("Summary of the model used:")
        print(proxy._model.summary())

    if eval_training_set is not None:
        print("Evaluation of the model on the training set")
        proxy_eval = ProxyLeapNet(name=f"{model_name}_evalTrainSet",
                                  max_row_training_set=max(eval_training_set, pred_batch_size),
                                  eval_batch_size=pred_batch_size,  # min(total_evaluation_step, 1024*64)
                                  layer=layer)
        agent_with_proxy_eval = AgentWithProxy(actor,
                                               proxy=proxy_eval,
                                               logdir=None)
        reproducible_exp(env, actor, env_seed=env_seed, agent_seed=agent_seed)
        agent_with_proxy_eval.evaluate(env,
                                       total_evaluation_step=pred_batch_size,
                                       load_path=os.path.join(save_path, model_name),
                                       save_path=save_path_final_results,
                                       metrics=metrics,
                                       verbose=verbose
                                       )


if __name__ == "__main__":
    main()
