# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from grid2op.dtypes import dt_float
from leap_net.agents.randomAct1 import RandomAct1


class RandomAct2(RandomAct1):
    """
    This combines two actions from `RandomAct1` each steps.

    The "rules" to combine the actions ensured that each action is composed of two sub actions and acts on two
    "stuffs" (either two powerlines, or two substation or 1 powerline and 1 substation).

    The rules are then the following:

    - two actions acting on the same substations cannot be composed together
    - the same action cannot be composed with itself.

    Notes
    -----
    This agent will modify all the substations at all steps. Make sure the `env.parameters.MAX_SUB_CHANGED` is
    big enough !

    Also the `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` need to be large enough ! Otherwise a substation cannot be
    acted upon at every step.

    Same for `env.parameters.NB_TIMESTEP_COOLDOWN_LINE`
    """

    def __init__(self, action_space, list_act=()):
        super(RandomAct2, self).__init__(action_space, list_act=list_act)

    def act(self, obs, reward, done=False):
        act_id1, sub_id_act1, this_random1 = self.sample_act()
        if sub_id_act1 is not None:
            sub_id_act1 = sub_id_act1[0]
        act_id2, sub_id_act2, this_random2 = self.sample_act(act_id1, sub_id_act1)
        res = self._combine_actions(self.all_to_one, this_random1)
        res = self._combine_actions(res, this_random2)
        return res
