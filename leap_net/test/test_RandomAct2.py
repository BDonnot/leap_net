# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import unittest
import grid2op
import warnings
import time
import numpy as np

from leap_net.agents import RandomAct2


class TestRandomAct2(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)

    def test_can_sample_simple(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 2, 1, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]}),
                    self.env.action_space({"set_line_status": [(1, -1)]}),
                    self.env.action_space({"set_line_status": [(2, -1)]}),
                    ]
        agent = RandomAct2(self.env.action_space, list_act=list_act)
        agent.seed(0)
        total_time = 0.
        nb_sim = 1000
        for i in range(nb_sim):
            act = agent.act(i, None)  # i pass the step in the act for debug, but it should be an observation otherwise
            beg_ = time.time()
            lines_impacted, _ = act.get_topological_impact()
            total_time += time.time() - beg_
            topo_vect = act.sub_set_bus
            nb_sub_impacted = 0
            prev = 0
            for sub_id, size_sub in enumerate(self.env.sub_info):
                this_sub_topo = topo_vect[prev: (prev + size_sub)]
                if np.sum(this_sub_topo == 2) > 0:
                    nb_sub_impacted += 1
                prev += size_sub
            assert np.sum(lines_impacted) + nb_sub_impacted == 2, f"action should act on two stuffs [step {i}]"
            # "2 stuffs" here is defined as:
            # - either two powerlines
            # - or two substations
            # - or a powerline and a substation

        # TODO if I have time, check the distribution of the simulated actions and make sure it's "uniform"
