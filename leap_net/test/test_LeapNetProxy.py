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
import tempfile


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                  topo_vect_to_tau="all")

        # valid only for this environment
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True)
        # the number of elements per substations are:
        # [3, 6, 4, 6, 5, 7, 3, 2, 5, 3, 3, 3, 4, 3]
        self.obs = self.env.reset()
        self.proxy.init([self.obs])

    def test_load_save(self):
        # test for "all"
        with tempfile.TemporaryDirectory() as path:
            self.proxy.save_data(path)
            proxy_loaded = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                        topo_vect_to_tau="all")
            proxy_loaded.load_data(path)
            self._aux_test_tau_from_topo_vect_all(proxy_loaded)

        # test the different load / save mechanics for the different encodings"
        with tempfile.TemporaryDirectory() as path:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="raw")
            proxy.init([self.obs])
            proxy.save_data(path)
            proxy_loaded = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                        topo_vect_to_tau="raw")
            proxy_loaded.load_data(path)
            self._aux_test_tau_default(proxy_loaded)

        with tempfile.TemporaryDirectory() as path:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="given_list",
                                 kwargs_tau=[(0, (2, 1, 1)), (0, (1, 2, 1)), (1, (2, 1, 1, 1, 1, 1)),
                                             (12, (2, 1, 1, 2)), (13, (2, 1, 2)), (13, (1, 2, 2))]
                                 )
            proxy.init([self.obs])
            proxy.save_data(path)
            proxy_loaded = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                        topo_vect_to_tau="given_list")
            proxy_loaded.load_data(path)
            self._aux_test_tau_from_list_topo(proxy_loaded)

        with tempfile.TemporaryDirectory() as path:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="online_list",
                                 kwargs_tau=7
                                 )
            proxy.init([self.obs])
            proxy.save_data(path)
            proxy_loaded = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                        topo_vect_to_tau="online_list",)
            proxy_loaded.load_data(path)
            self._aux_test_tau_from_online_topo(proxy_loaded)

    def test_tau_default(self):
        self._aux_test_tau_default()

    def _aux_test_tau_default(self, proxy=None):
        if proxy is None:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="raw")
            proxy.init([self.obs])

        # valid only for this environment
        env = self.env
        # the number of elements per substations are:
        # [3, 6, 4, 6, 5, 7, 3, 2, 5, 3, 3, 3, 4, 3]
        obs = env.reset()
        proxy.init([obs])

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[0] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[1] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[3] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 4
        assert np.all(res[[50, 51, 52, 53]] == 1.)

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 3
        assert np.all(res[[54, 55, 56]] == 1.)

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 4
        assert res[0] == 1.
        assert np.all(res[[54, 55, 56]] == 1.)

    def test_tau_from_topo_vect_all(self):
        self._aux_test_tau_from_topo_vect_all()

    def _aux_test_tau_from_topo_vect_all(self, proxy=None):
        # for the complete topology
        if proxy is None:
            proxy = self.proxy

        obs = self.env.reset()
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 0

        # the "first" topology: change the first element of the substation 0
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[0] == 1.

        # second topology: change the second element of substation 0.
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[1] == 1.

        # the last topology identified: everything on bus 2 for substation 13 (last one)
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[389] == 1.

        # as there are 3 elements on the substation 0, there are 7 (=2^3 - 1) different possible topologies
        # for this one
        # And this will be the "first topology of substation 1"
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[7] == 1.

        # change the substation 12
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 1
        assert tau[382] == 1.

        # if i change 2 substation, i have 2 "1" on the tau vector
        obs = self.env.reset()
        act = self.env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = self.env.step(act)
        act = self.env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = self.env.step(act)
        tau = proxy.topo_vect_handler(obs)
        assert np.sum(tau) == 2
        assert tau[389] == 1.
        assert tau[0] == 1.

    def test_tau_from_list_topo(self):
        self. _aux_test_tau_from_list_topo()

    def _aux_test_tau_from_list_topo(self, proxy=None):
        if proxy is None:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="given_list",
                                 kwargs_tau=[(0, (2, 1, 1)), (0, (1, 2, 1)), (1, (2, 1, 1, 1, 1, 1)),
                                             (12, (2, 1, 1, 2)), (13, (2, 1, 2)), (13, (1, 2, 2)), (1, (2, 1, 2, 1, 2, 1))]
                                 )
            proxy.init([self.obs])

        env = self.env
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[0] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[1] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[2] == 1.

        # I test that -1 are still considered as 1
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, -1, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        assert not done
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[2] == 1.

        # I test that -1 are still considered as 1, even if everything is on bus 1
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (1, -1, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        assert not done
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        # test that everything on bys 2 and everything on bus 1 leads to the same result
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        # check the symmetry 2 <-> 1
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 1, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 2
        assert res[0] == 1.
        assert res[4] == 1.

        # test the symmetry (gain...) why not
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (1, 2, 2, 2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[2] == 1.

        # test that if a line is disconnected, we are still able to match the topologies
        env = self.env
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, -1, 1, 2, 1))]}})#(2, 1, 2, 1, 2, 1)
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[6] == 1.

        env = self.env
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, -1, 2, 1, 2, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[6] == 1.

    def test_tau_from_online_topo(self):
        self._aux_test_tau_from_online_topo()

    def _aux_test_tau_from_online_topo(self, proxy=None):
        if proxy is None:
            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="online_list",
                                 kwargs_tau=7
                                 )
            proxy.init([self.obs])

        env = self.env

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[0] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[1] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[2] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        act = env.action_space({"set_bus": {"substations_id": [(13, (2, 1, 2))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 2
        assert res[0] == 1.
        assert res[3] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[4] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 2, 1, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[5] == 1.

        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 1
        assert res[6] == 1.

        # "overflow" in the number of topologies => like ref topology
        obs = env.reset()
        act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 2, 1, 1))]}})
        obs, reward, done, info = env.step(act)
        with warnings.catch_warnings():
            # check that it raises the right warnings
            warnings.filterwarnings("error")
            with self.assertRaises(UserWarning):
                res = proxy.topo_vect_handler(obs)
        # check despite the warning, the things are processed correctly
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = proxy.topo_vect_handler(obs)
        assert np.sum(res) == 0
