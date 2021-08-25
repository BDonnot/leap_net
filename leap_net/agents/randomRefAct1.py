# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from grid2op.dtypes import dt_float
from leap_net.agents.randomAct1 import RandomAct1


class RandomRefAct1(RandomAct1):
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

    Also the `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` need to be large enough ! Otherwise a substation cannot be
    acted upon at every step.

    Same for `env.parameters.NB_TIMESTEP_COOLDOWN_LINE`

    """

    def __init__(self, action_space, p=0.5, list_act=()):
        super(RandomRefAct1, self).__init__(action_space, list_act=list_act)
        self.p = dt_float(p)
        self._1_p = 1. - self.p

    def act(self, obs, reward, done=False):
        ur = self.space_prng.uniform()
        if ur < self._1_p:
            res = self.all_to_one
        else:
            res = super().act(obs, reward, done)
        return res
