# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import os
import grid2op
import numpy as np
from tqdm import tqdm

from grid2op.Agent import DoNothingAgent
from grid2op.Parameters import Parameters
from grid2op.dtypes import dt_float, dt_int
from grid2op.Rules import AlwaysLegal
from leap_net.generate_data.Agents import RandomNN1, RandomN1, RandomN2


def get_agent(env, agent_name, **kwargsagent):
    if agent_name == "do_nothing":
        res = DoNothingAgent(env.action_space, **kwargsagent)
    elif agent_name == "random_n_n1":
        res = RandomNN1(env.action_space, **kwargsagent)
    elif agent_name == "random_n1":
        res = RandomN1(env.action_space, **kwargsagent)
    elif agent_name == "random_n1":
        res = RandomN2(env.action_space, **kwargsagent)
    else:
        raise NotImplementedError()
    return res


def generate_dataset(name_env,
                     nb_rows,
                     dir_out="training_dataset",
                     agent_type="do_nothing",
                     expe_type="powerline",
                     verbose=True,
                     **kwargsagent):
    # TODO seed for reproductible expriments
    # TODO remove thermal limits
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    if isinstance(name_env, str):
        env = grid2op.make(dataset=name_env, param=param, gamerules_class=AlwaysLegal)
    else:
        raise NotImplementedError()

    if isinstance(agent_type, str):
        agent = get_agent(env, agent_type, **kwargsagent)
    else:
        raise NotImplementedError()

    dir_out_abs = os.path.abspath(dir_out)
    if not os.path.exists(dir_out_abs):
        os.mkdir(dir_out_abs)

    # input X
    prod_p = np.full((nb_rows, env.n_gen), fill_value=np.NaN, dtype=dt_float)
    prod_v = np.full((nb_rows, env.n_gen), fill_value=np.NaN, dtype=dt_float)
    load_p = np.full((nb_rows, env.n_load), fill_value=np.NaN, dtype=dt_float)
    load_q = np.full((nb_rows, env.n_load), fill_value=np.NaN, dtype=dt_float)

    # input tau
    if expe_type == "powerline":
        tau = np.full((nb_rows, env.n_line), fill_value=np.NaN, dtype=dt_int)
    elif expe_type == "topo":
        tau = np.full((nb_rows, env.dim_topo), fill_value=np.NaN, dtype=dt_int)
    else:
        raise NotImplementedError()

    # output
    flow_a = np.full((nb_rows, env.n_line), fill_value=np.NaN, dtype=dt_float)
    line_v = np.full((nb_rows, env.n_line), fill_value=np.NaN, dtype=dt_float)
    flow_p = np.full((nb_rows, env.n_line), fill_value=np.NaN, dtype=dt_float)
    flow_q = np.full((nb_rows, env.n_line), fill_value=np.NaN, dtype=dt_float)

    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    t = 0
    if verbose:
        pbar = tqdm(total=nb_rows)
    while t < nb_rows:
        act = agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)
        if done:
            # action was not valid, i restart
            # TODO random chronics
            # TODO random begining of start
            obs = env.reset()
            reward = env.reward_range[0]
            done = False
            continue

        # so action is valid, i store the results
        prod_p[t, :] = obs.prod_p
        prod_v[t, :] = obs.prod_v
        load_p[t, :] = obs.load_p
        load_q[t, :] = obs.load_q
        if expe_type == "powerline":
            tau[t, :] = 1 - obs.line_status  # 0 = connected, 1 = disconnected
        elif expe_type == "topo":
            tau[t, :] = obs.topo_vect - 1  # 0 = on bus 1, 1 = on bus 2
        flow_a[t, :] = obs.a_or
        line_v[t, :] = obs.v_or
        flow_p[t, :] = obs.p_or
        flow_q[t, :] = obs.q_or
        t += 1
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    for arr_, arr_n in zip([prod_p, prod_v, load_p, load_q, tau, flow_a, flow_p, flow_q, line_v],
                           ["prod_p", "prod_v", "load_p", "load_q", "tau", "flow_a", "flow_p", "flow_q", "line_v"]):
        tmp_nm = os.path.join(dir_out_abs, "{}.npy".format(arr_n))
        np.save(file=tmp_nm, arr=arr_)
