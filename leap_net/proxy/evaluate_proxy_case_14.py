# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import re
import os
import matplotlib.pyplot as plt
from leap_net.proxy.utils import create_env, reproducible_exp, DEFAULT_METRICS

from leap_net.tf_keras import ResNetLayer
from leap_net.agents import RandomN1, RandomN2
from leap_net.proxy.agentWithProxy import AgentWithProxy
from leap_net.proxy.proxyLeapNet import ProxyLeapNet
from leap_net.proxy.proxyBackend import ProxyBackend


def main(
        # reload the model
        env_name="l2rpn_case14_sandbox",
        save_path="model_saved",
        use_lightsim_if_available=True,
        val_regex=".*99[0-9].*",
        model_name="leapnet_case_14",
        layer=ResNetLayer,  # for now this is not serialized in the model
        # parameters for the evaluation
        do_dc=True,
        do_N1 = True,
        do_N2 = True,
        li_batch_size=tuple(),  # if you want to study the impact of the batch size
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

    if save_path_final_results is not None:
        if not os.path.exists(save_path_final_results):
            os.mkdir(save_path_final_results)
            if verbose > 0:
                print(f"Creating path \"{save_path_final_results}\" where results are stored")
    if do_dc:
        print("####################### \n"
              "## DC approximation  ## \n"
              "####################### \n")
        actor_evalN1_dc = RandomN1(env.action_space)
        proxy_dc = ProxyBackend(env._init_grid_path,
                                name=f"{model_name}_evalDC",
                                is_dc=True)
        proxy_dc.init([obs])  # dc is not stored (of course) so i need to manually load it
        agent_with_proxy_dc = AgentWithProxy(actor_evalN1_dc,
                                             proxy=proxy_dc,
                                             logdir=None)
        reproducible_exp(env,
                         agent=actor_evalN1_dc,
                         env_seed=env_seed,
                         agent_seed=agent_seed,
                         chron_id_start=chron_id_val)
        agent_with_proxy_dc.evaluate(env,
                                     load_path=None,
                                     save_path=save_path_final_results,
                                     # I do less because the current implementation takes too long
                                     total_evaluation_step=int(total_evaluation_step / 128),
                                     metrics=metrics,
                                     verbose=verbose
                                     )

    if do_N1:
        print("#######################\n"
              "##     Test set      ##\n"
              "#######################\n")
        actor_evalN1 = RandomN1(env.action_space)
        proxy_eval = ProxyLeapNet(name=f"{model_name}_evalN1",
                                  max_row_training_set=max(total_evaluation_step, pred_batch_size),
                                  eval_batch_size=pred_batch_size,
                                  layer=layer
                                  )
        agent_with_proxy_evalN1 = AgentWithProxy(actor_evalN1,
                                                 proxy=proxy_eval,
                                                 logdir=None)
        reproducible_exp(env,
                         agent=actor_evalN1,
                         env_seed=env_seed,
                         agent_seed=agent_seed,
                         chron_id_start=chron_id_val)
        agent_with_proxy_evalN1.evaluate(env,
                                         total_evaluation_step=total_evaluation_step,
                                         load_path=os.path.join(save_path, model_name),
                                         save_path=save_path_final_results,
                                         metrics=metrics,
                                         verbose=verbose
                                         )

    if do_N2:
        print("#######################\n"
              "##   SuperTest set   ##\n"
              "#######################\n")
        actor_evalN2 = RandomN2(env.action_space)
        proxy_eval = ProxyLeapNet(name=f"{model_name}_evalN2",
                                  max_row_training_set=max(total_evaluation_step, pred_batch_size),
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
                                         verbose=verbose
                                         )

    if len(li_batch_size):
        print("###########################\n"
              "##  Impact of Batch size ##\n"
              "###########################\n")
        actor_batch_size = RandomN1(env.action_space)
        times_per_pf_ms = []
        total_times_ms = []
        for pred_batch_size in li_batch_size:
            reproducible_exp(env,
                             agent=actor_batch_size,
                             env_seed=env_seed,
                             agent_seed=agent_seed,
                             chron_id_start=chron_id_val)
            proxy_eval_tmp = ProxyLeapNet(name=f"{model_name}_evalN1_{pred_batch_size}",
                                          max_row_training_set=max(total_evaluation_step, pred_batch_size),
                                          eval_batch_size=pred_batch_size,  # min(total_evaluation_step, 1024*64)
                                          layer=layer)
            agent_with_tmp = AgentWithProxy(actor_batch_size,
                                            proxy=proxy_eval_tmp,
                                            logdir=None)

            dict_metrics = agent_with_tmp.evaluate(env,
                                                   total_evaluation_step=pred_batch_size,
                                                   load_path=os.path.join(save_path, model_name),
                                                   save_path=save_path_final_results,
                                                   metrics={},
                                                   verbose=0,
                                                   save_values=False  # I do not save the arrays
                                                   )
            total_pred_time_ms = 1000.*dict_metrics["predict_time"]
            total_times_ms.append(total_pred_time_ms)
            times_per_pf_ms.append(total_pred_time_ms/pred_batch_size)
            if verbose:
                print(f'Time to compute {pred_batch_size} powerflows: {total_pred_time_ms:.2f}ms '
                      f'({total_pred_time_ms/pred_batch_size:.4f} ms/powerflow)')

        if save_path_final_results is not None:
            # save the figure of the total time
            fig, ax = plt.subplots()
            ax.plot(li_batch_size, total_times_ms)
            ax.set_title('Total computation time')
            ax.set_xlabel("Number of powerflows")
            ax.set_ylabel("Total time (ms)")
            ax.set_yscale('log')
            ax.set_xscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(save_path_final_results, f"{model_name}_total_comp_time.pdf"))

            # save the figure of the total time
            fig, ax = plt.subplots()
            ax.plot(li_batch_size, times_per_pf_ms)
            ax.set_title('Computation time per powerflow')
            ax.set_xlabel("Number of powerflows")
            ax.set_ylabel("Time per powerflow (ms / pf)")
            ax.set_yscale('log')
            ax.set_xscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(save_path_final_results, f"{model_name}_time_per_pf.pdf"))


if __name__ == "__main__":
    main()
