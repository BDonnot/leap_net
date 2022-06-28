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

import numpy as np

from leap_net.agents import RandomAct1


class TestRandomAct1(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)

    def test_fail_if_nothing(self):
        with self.assertRaises(RuntimeError):
            agent = RandomAct1(self.env.action_space, list_act=[])

    def test_fail_if_only_do_nothing(self):
        with self.assertRaises(RuntimeError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                agent = RandomAct1(self.env.action_space, list_act=[self.env.action_space()])

    def test_fail_if_not_glop_action(self):
        with self.assertRaises(RuntimeError):
            agent = RandomAct1(self.env.action_space, list_act=[None])

    def test_fail_if_only_one_sub(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}})
                    ]
        with self.assertRaises(RuntimeError):
            agent = RandomAct1(self.env.action_space, list_act=list_act)

    def test_doesnt_fail_if_not_only_one_sub(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]})
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)

    def test_can_sample_simple(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]})
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)
        for i in range(10):
            agent.act(None, None)

    def test_can_combine_topo_with_powerline(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]})
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)
        res = agent._combine_actions(list_act[0], list_act[2])
        assert np.sum(res.line_set_status == -1) == 1, "line 0 should be disconnected"
        assert res.line_set_status[0] == -1, "line 0 should be disconnected"
        *_, info = self.env.step(res)
        assert not info["is_illegal"], "this action should be legal"
        assert not info["is_ambiguous"], "this action should not be ambiguous"

    def test_can_sample_more_complex_1(self):
        """i prevent to sample the topo action, it should sample the line one"""
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]})
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)
        act_id, sub_id_act, act = agent.sample_act(previous_act_id=2, sub_id_act=1)
        # check that it's the last action that is sampled
        assert act_id == 2
        assert sub_id_act is None
        assert np.sum(act.line_set_status == -1) == 1, "line 0 should be disconnected"
        assert act.line_set_status[0] == -1, "line 0 should be disconnected"

    def test_can_sample_more_complex_2(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]}),
                    self.env.action_space({"set_line_status": [(1, -1)]}),
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)
        act2 = 0
        act3 = 0
        for _ in range(20):
            act_id, sub_id_act, act = agent.sample_act(previous_act_id=1, sub_id_act=1)
            # check that it's the last action that is sampled
            assert act_id == 2 or act_id == 3
            if act_id == 2:
                act2 += 1
            else:
                act3 += 1
        # the test above is mainly to "check" that it samples randomly among the two possible action, and that
        # it does not always sample the same one
        assert act2 == 9
        assert act3 == 11

    def test_can_sample_more_complex_3(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 2, 1, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}})
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)
        act0 = 0
        act1 = 0
        act2 = 0
        act3 = 0
        # I sample uniformely on substation "before"
        for _ in range(20):
            act_id, sub_id_act, act = agent.sample_act(previous_act_id=2, sub_id_act=2)
            # check that it's the last action that is sampled
            assert act_id == 0 or act_id == 1
            if act_id == 0:
                act0 += 1
            else:
                act1 += 1
        # the test above is mainly to "check" that it samples randomly among the two possible action, and that
        # it does not always sample the same one
        assert act0 == 9
        assert act1 == 11

        # I sample uniformely on substation "after"
        for _ in range(20):
            act_id, sub_id_act, act = agent.sample_act(previous_act_id=2, sub_id_act=1)
            # check that it's the last action that is sampled
            assert act_id == 2 or act_id == 3
            if act_id == 2:
                act2 += 1
            else:
                act3 += 1
        # the test above is mainly to "check" that it samples randomly among the two possible action, and that
        # it does not always sample the same one
        assert act2 == 8
        assert act3 == 12

    def test_can_sample_more_complex_4(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 2, 1, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 1, 2))]}}),
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(0)

        act0 = 0
        act1 = 0
        act4 = 0
        act5 = 0
        # I sample uniformely on substation "after" or "before"
        for _ in range(1000):
            act_id, sub_id_act, act = agent.sample_act(previous_act_id=2, sub_id_act=2)
            # check that it's the last action that is sampled
            assert act_id == 0 or act_id == 1 or act_id == 4 or act_id == 5
            if act_id == 0:
                act0 += 1
            elif act_id == 1:
                act1 += 1
            elif act_id == 4:
                act4 += 1
            elif act_id == 5:
                act5 += 1
        # the test above is mainly to "check" that it samples randomly among the two possible action, and that
        # it does not always sample the same one
        assert act0 == 255
        assert act1 == 253
        assert act4 == 241
        assert act5 == 251

    def test_can_sample_more_complex_5(self):
        list_act = [self.env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 2))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 2, 1, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(2, (2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 2, 1))]}}),
                    self.env.action_space({"set_bus": {"substations_id": [(3, (2, 1, 2, 1, 1, 2))]}}),
                    self.env.action_space({"set_line_status": [(0, -1)]}),
                    ]
        agent = RandomAct1(self.env.action_space, list_act=list_act)
        agent.seed(1)

        act0 = 0
        act1 = 0
        act4 = 0
        act5 = 0
        act6 = 0
        # I sample uniformely on substation "after" or "before"
        for _ in range(1000):
            act_id, sub_id_act, act = agent.sample_act(previous_act_id=2, sub_id_act=2)
            # check that it's the last action that is sampled
            assert act_id == 0 or act_id == 1 or act_id == 4 or act_id == 5 or act_id == 6 or act_id == 7
            if act_id == 0:
                act0 += 1
            elif act_id == 1:
                act1 += 1
            elif act_id == 4:
                act4 += 1
            elif act_id == 5:
                act5 += 1
            elif act_id == 6:
                act6 += 1
        # the test above is mainly to "check" that it samples randomly among the two possible action, and that
        # it does not always sample the same one
        assert act0 == 198
        assert act1 == 188
        assert act4 == 213
        assert act5 == 196
        assert act6 == 205
