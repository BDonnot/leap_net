# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import re
import os

import tensorflow as tf

from leap_net.ResNetLayer import ResNetLayer
from leap_net.agents import RandomNN1
from leap_net.proxy.agentWithProxy import AgentWithProxy
from leap_net.proxy.proxyLeapNet import ProxyLeapNet

from leap_net.proxy.utils import create_env, reproducible_exp, DEFAULT_METRICS, limit_gpu_usage


def main(limit_gpu_memory=True,
         total_train=int(1024)*int(1024),  # number of observations that will be gathered
         env_name="l2rpn_case14_sandbox",
         # log during training
         save_path="model_saved",
         save_path_tensorbaord="tf_logs",
         update_tensorboard=256,  # tensorboard is updated every XXX training iterations
         save_freq=int(1024) * int(4),  # model is saved every save_freq training iterations
         ext=".h5",  # extension of the file in which you want to save the proxy
         nb_obs_init=512,  # number of observations that are sent to the proxy to be initialized
         # dataset / data generation part
         env_seed=42,
         agent_seed=1,
         use_lightsim_if_available=True,
         val_regex=".*99[0-9].*",
         actor_class=RandomNN1,
         load_dataset=True, # do you load the entire training set in memory (can take a few minutes - set it to false if you simply want to make some tests)
         # perform an evaluation on the training set at the end of training
         eval_training_set=int(1024) * int(128),  # number of powerflow the proxy will do after training
         pred_batch_size=int(1024) * int(128),  # number of powerflow that will be done by the proxy "at once"
         save_path_final_results="model_results",  # where the information about the prediction will be stored
         metrics=DEFAULT_METRICS,  # which metrics are used to evaluate the performance of the model
         verbose=1,  # do I print the results of the model
         # proxy part (sizes are not really "parametrized" this is just "something that works approximately)
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
         topo_vect_to_tau="all",
         # TODO add the other constructor parameters of the proxy
         ):
    if limit_gpu_memory:
        limit_gpu_usage()

    # create the environment
    env = create_env(env_name, use_lightsim_if_available=use_lightsim_if_available)

    # I select only part of the data, for training
    if load_dataset:
        env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
        env.chronics_handler.real_data.reset()
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
                         attr_y=attr_y,
                         topo_vect_to_tau=topo_vect_to_tau)
    agent_with_proxy = AgentWithProxy(actor,
                                      proxy=proxy,
                                      logdir=save_path_tensorbaord,
                                      update_tensorboard=update_tensorboard,
                                      save_freq=save_freq,
                                      ext=ext,
                                      nb_obs_init=nb_obs_init
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
    from tensorflow.keras.layers import Dense
    main()
