# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.dtypes import dt_int, dt_float


class RandomSub1(BaseAgent):
    """
    This "agent" will randomly change the topology of one substation. The way the "randomly" works can be:

    - either it's random uniform on all the topologies (not recommended except for some tests)
    - or it will first draw a substation at random, and then perform a random action (uniformly) among this
      substation. We recommend this later version for normal grid (ieee118, real grid etc.)

    So basically we do not recommend to change the default "random_king='substation'"

    Notes
    -----
    This agent will modify all the substations at all steps. Make sure the `env.parameters.MAX_SUB_CHANGED` is
    big enough !

    Also the `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` need to be small enough ! Otherwise a substation cannot be
    acted upon at every step.

    """

    def __init__(self, action_space, random_kind="substation"):
        super(RandomSub1, self).__init__(action_space)
        if "set_bus" not in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a RandomSub1 agent if you cannot set the bus of the "
                                      "substations")

        # represent the action "exactly one powerline is disconnected
        self.all_to_one = action_space()
        self.all_to_one.set_bus = np.ones(action_space.dim_topo, dtype=dt_int)
        self.all_topo = []
        self.random_kind = random_kind
        self.sub_considered = []  # list of substation with more than 2 feasible topology
        self.sub_to_possible_id = np.zeros(action_space.n_sub, dtype=int)
        possible_id = 0
        for sub_id in range(action_space.n_sub):
            tmp = action_space.get_all_unitary_topologies_set(action_space, sub_id)
            if len(tmp):
                act_non_dn = tmp[1:]
                if self.random_kind == "substation":
                    self.sub_considered.append(sub_id)
                    self.all_topo.append((sub_id, act_non_dn))  # don't take into account the do nothing action
                    self.sub_to_possible_id[sub_id] = possible_id
                    possible_id += 1
                elif self.random_kind == "all":
                    self.all_topo += act_non_dn
                else:
                    raise RuntimeError(f"Unknown sampling type: \"{random_kind}\"")

    def sample_act(self, previous_sub_id=None):
        sub_id = None
        if self.random_kind == "substation":
            if previous_sub_id is None:
                id_ = self.space_prng.choice(len(self.all_topo))
            else:
                id_ = self.space_prng.choice(len(self.all_topo) - 1)
                if id_ >= self.sub_to_possible_id[previous_sub_id]:
                    # this procedure is to be sure not to "disconnect twice" the same powerline
                    id_ += 1
            sub_id, all_topo_this_sub = self.all_topo[id_]
            this_random = self.space_prng.choice(all_topo_this_sub)
        elif self.random_kind == "all":
            this_random = self.space_prng.choice(self.all_topo)
        else:
            raise RuntimeError(f"Unknown sampling type: \"{self.random_kind}\"")
        return sub_id, this_random

    def act(self, obs, reward, done=False):
        sub_id, this_random = self.sample_act()
        res = self.all_to_one + this_random
        return res
