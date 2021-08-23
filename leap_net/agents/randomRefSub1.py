# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from grid2op.dtypes import dt_float
from leap_net.agents.randomSub1 import RandomSub1


class RandomRefSub1(RandomSub1):
    """
    This "agent" will randomly disconnect 0 or 1 powerline with probability p to disconnect 1 powerline
    and with probability 1-p to reconnect everything.

    The way the "randomly" works can be:

    - either it's random uniform on all the topologies (not recommended except for some tests)
    - or it will first draw a substation at random, and then perform a random action (uniformly) among this
      substation. We recommend this later version for normal grid (ieee118, real grid etc.)

    At each step, there is a probability `p`


    Notes
    -----
    This agent will modify all the substations at all steps. Make sure the `env.parameters.MAX_SUB_CHANGED` is
    big enough !

    Also the `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` need to be small enough ! Otherwise a substation cannot be
    acted upon at every step.

    """

    def __init__(self, action_space, p, random_kind="substation"):
        super(RandomRefSub1, self).__init__(action_space, random_kind=random_kind)
        self.p = dt_float(p)
        self._1_p = 1. - self.p

    def act(self, obs, reward, done=False):
        ur = self.space_prng.uniform()
        if ur < self._1_p:
            res = self.all_to_one
        else:
            res = super().act(obs, reward, done)
        return res
