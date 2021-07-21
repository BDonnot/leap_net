# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import grid2op
import numpy as np
import unittest
import warnings
from leap_net.proxy import ProxyLeapNet


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = ProxyLeapNet(attr_tau=("line_status",),
                                  topo_vect_to_tau="all")

        # valid only for this environment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        # the number of elements per substations are:
        # [3, 6, 4, 6, 5, 7, 3, 2, 5, 3, 3, 3, 4, 3]
        self.obs = self.env.reset()
        self.proxy.init([self.obs])

    def test_tau_from_topo_vect_all(self):
        # for the complete topology
        obs = self.env.reset()
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 0

        # the "first" topology: change the first element of the substation 0
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[0] == 1.

        # second topology: change the second element of substation 0.
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[1] == 1.

        # the last topology identified: everything on bus 2 for substation 13 (last one)
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[389] == 1.

        # as there are 3 elements on the substation 0, there are 7 (=2^3 - 1) different possible topologies
        # for this one
        # And this will be the "first topology of substation 1"
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[7] == 1.

        # change the substation 12
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[382] == 1.

        # if i change 2 substation, i have 2 "1" on the tau vector
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        act = self.env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = self.proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 2
        assert tau[389] == 1.
        assert tau[0] == 1.
