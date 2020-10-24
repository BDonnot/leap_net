# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import re
import os

from leap_net.proxy.train_proxy_case14 import create_env, reproducible_exp, DEFAULT_METRICS
from leap_net.agents import RandomN2

from leap_net.ResNetLayer import ResNetLayer
from leap_net.agents import RandomNN1
from leap_net.proxy.AgentWithProxy import AgentWithProxy
from leap_net.proxy.ProxyLeapNet import ProxyLeapNet


def main(
        # reload the model
        env_name="l2rpn_case14_sandbox",
        save_path="model_saved",
        save_path_tensorbaord="tf_logs",
        use_lightsim_if_available=True,
         val_regex=".*99[0-9].*",
         model_name="leapnet_case_14",
        layer=ResNetLayer,  # for now this is not serialized in the model

        # parameters for the evaluation
        do_dc=False,
        do_N1 = False,
        do_N2 = False,
        total_evaluation_step=int(1024) * int(128),
        pred_batch_size=int(1024) * int(128),
        save_path_final_results="model_results",  # where the information about the prediction will be stored
        metrics=DEFAULT_METRICS,  # which metrics are used to evaluate the performance of the model
        verbose=1,  # do I print the results of the model

        # enforce reproducibility
        chron_id_val=0,
        env_seed=0,
        agent_seed=42,
):
    # create the environment
    env = create_env(env_name, use_lightsim_if_available=use_lightsim_if_available)

    env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is not None)
    env.chronics_handler.real_data.reset()
    obs = env.reset()

    if do_N2:
        print("#######################\n"
              "##   SuperTest set   ##\n"
              "#######################\n")
        actor_evalN2 = RandomN2(env.action_space)
        proxy_eval = ProxyLeapNet(name=f"{model_name}_evalN2",
                                  max_row_training_set=total_evaluation_step,
                                  eval_batch_size=pred_batch_size,
                                  layer=layer
                                  )
        agent_with_proxy_evalN2 = AgentWithProxy(actor_evalN2,
                                                 proxy=proxy_eval,
                                                 logdir=None)
        reproducible_exp(env,
                         agent=actor_evalN2,
                         env_seed=env_seed,
                         agent_seed=agent_seed,
                         chron_id_start=chron_id_val)
        agent_with_proxy_evalN2.evaluate(env,
                                         total_evaluation_step=total_evaluation_step,
                                         load_path=os.path.join(save_path, model_name),
                                         save_path=save_path_final_results,
                                         metrics=metrics,
                                         verbose=1
                                         )


if __name__ == "__main__":
    main()