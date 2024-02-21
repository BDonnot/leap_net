# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import grid2op
from grid2op.Parameters import Parameters
import numpy as np
from leap_net.agents import RandomSub1, RandomSub2, RandomRefSub1

import unittest


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # init the environment
        param = Parameters()
        param.NO_OVERFLOW_DISCONNECTION = True
        # i can act on all powerline / substation at once
        param.MAX_LINE_STATUS_CHANGED = 999999
        param.MAX_SUB_CHANGED = 999999
        # i can act every step on every line / substation (no cooldown)
        param.NB_TIMESTEP_COOLDOWN_LINE = 0
        param.NB_TIMESTEP_COOLDOWN_SUB = 0

        self.env = grid2op.make("l2rpn_case14_sandbox", test=True, param=param)
        self.obs = self.env.reset()

    def test_RandomSub1(self):
        agent1 = RandomSub1(self.env.action_space)
        agent1.seed(0)

        subs_index = []
        prev = 0
        for sub_id in range(self.obs.n_sub):
            nb_el = self.obs.sub_info[sub_id]
            subs_index.append((prev, prev + nb_el))
            prev += self.obs.sub_info[sub_id]

        for i in range(1000):
            act = agent1.act(None, None, None)
            lines_, subs_ = act.get_topological_impact()
            assert np.sum(lines_) == 0, f"error: RandomSub1 act on powerlines at iteration {i}"

            nb_sub_aff = 0
            for sub_id in range(self.env.n_sub):
                beg_, end_ = subs_index[sub_id]
                this_sub_topo = act.set_bus[beg_:end_]
                nb_sub_aff += np.any(this_sub_topo != 1)
            assert nb_sub_aff == 1, f"error: RandomSub1 act on multiple powerlines at iteration {i}"

    def test_RandomSub2(self):
        agent2 = RandomSub2(self.env.action_space)
        agent2.seed(0)

        subs_index = []
        prev = 0
        for sub_id in range(self.obs.n_sub):
            nb_el = self.obs.sub_info[sub_id]
            subs_index.append((prev, prev + nb_el))
            prev += self.obs.sub_info[sub_id]

        for i in range(1000):
            act = agent2.act(None, None, None)
            lines_, subs_ = act.get_topological_impact()
            assert np.sum(lines_) == 0

            nb_sub_aff = 0
            for sub_id in range(self.env.n_sub):
                beg_, end_ = subs_index[sub_id]
                this_sub_topo = act.set_bus[beg_:end_]
                nb_sub_aff += np.any(this_sub_topo != 1)
            assert nb_sub_aff == 2, f"error RandomSub2 does not act on 2 substation at iteration {i}"

    def test_RandomRefSub1(self):
        p = 0.1
        n = 1000
        agentref1 = RandomRefSub1(self.env.action_space, p=p)
        agentref1.seed(1)

        subs_index = []
        prev = 0
        for sub_id in range(self.obs.n_sub):
            nb_el = self.obs.sub_info[sub_id]
            subs_index.append((prev, prev + nb_el))
            prev += self.obs.sub_info[sub_id]

        test_ = [0, 0]
        for i in range(n):
            act = agentref1.act(None, None, None)
            lines_, subs_ = act.get_topological_impact()
            assert np.sum(lines_) == 0

            nb_sub_aff = 0
            for sub_id in range(self.env.n_sub):
                beg_, end_ = subs_index[sub_id]
                this_sub_topo = act.set_bus[beg_:end_]
                nb_sub_aff += np.any(this_sub_topo != 1)
            assert nb_sub_aff <= 1
            test_[nb_sub_aff] += 1
        # now check that i have approximately the right proba
        assert abs(test_[0] / n - (1 - p)) <= 1.96 * np.sqrt(p * (1-p)) / np.sqrt(n)
        assert abs(test_[1] / n - p) <= 1.96 * np.sqrt(p * (1-p)) / np.sqrt(n)
