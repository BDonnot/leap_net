# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import copy
import warnings
import numpy as np
from typing import Union

from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.dtypes import dt_int, dt_float


class RandomAct1(BaseAgent):
    """
    This agent will send random action from a provided list (provided with argument
    `list_act`)

    The actions should meet some properties:

    - each action should be unique (not checked, but expect errors if this is not met)
    - taken as a whole, actions should act on different substation (meaning that either no action act on any substation
      or there are 2 different actions acting on two different substation)
    - each action should act on at most one substation
    - do nothing action will not be added (but it's not an error)
    - actions should use "set" and not "change" (ie `set_bus` and `set_line_status` but NOT `change_bus` nor
      `change_line_status`)
    - at least one action should do something (full do nothing action or an empty list are not allowed)
    - disconnection of element using "set_bus" are not allowed (not checked but expect some bugs if you do that)

    Notes
    -----
    This agent will modify all the substations at all steps. Make sure the `env.parameters.MAX_SUB_CHANGED` is
    big enough !

    Also the `env.parameters.NB_TIMESTEP_COOLDOWN_SUB` need to be large enough ! Otherwise a substation cannot be
    acted upon at every step.

    Same for `env.parameters.NB_TIMESTEP_COOLDOWN_LINE`
    """

    def __init__(self, action_space, list_act=()):
        super(RandomAct1, self).__init__(action_space)
        if "set_bus" not in action_space.subtype.authorized_keys:
            raise NotImplementedError("Impossible to have a RandomSub1 agent if you cannot set the bus of the "
                                      "substations")

        # represent the action "exactly one powerline is disconnected
        self.all_to_one = action_space()
        self.all_to_one.set_bus = np.ones(action_space.dim_topo, dtype=dt_int)
        self._all_actions = []
        self._subs_impacted_by_action = []
        self._n_sub = self.action_space.n_sub
        self._affect_lines = self.action_space.n_sub
        self._nb_act = np.zeros(self._n_sub + 1, dtype=int)  # [sub_0, sub_1, ..., sub_n, powerlines]

        # self._subs_impacted_to_act_id:
        #     sub_0: {id of actions acting on sub_0 (and possibly other sub)},
        #     sub_1: {id of actions acting on sub 1 (and possibly other sub)},
        #     ..., sub_n: {...},
        #     powerlines: {id of actions acting only on powerlines}
        self._subs_impacted_to_act_id = [[] for _ in range(self._n_sub + 1)]

        # add the actions in the list, and perform lots of checks! (to avoid misuse)
        subs_impact = np.zeros(action_space.n_sub, dtype=int)
        warning_different_act_same_sub_issued = False
        all_act_topo = True
        act_id = 0
        for act_ref_id, act in enumerate(list_act):
            if not isinstance(act, BaseAction):
                # act should be a grid2op actions...
                raise RuntimeError(f"The list_act should contain only valid grid2op actions. Found {type(act)}.")
            if np.any(act.line_change_status):
                # act should not "change" the status of powerline but "set" them
                raise RuntimeError("action of type \"line_change_status\" are not supported at the moment. Please use "
                                   "`line_change_status`.")
            if np.any(act.change_bus):
                # act should not "change" the bus of elements but "set" them
                raise RuntimeError("action of type \"change_bus\" are not supported at the moment please use "
                                   "`change_bus`.")

            # TODO do i check for "-1" in the set_bus ?
            # TODO do i check for "0" in the "set_bus" of the affected substation ?

            # all below is to make sure that, when combining the action, i actually result in the
            # combined actions and not just the second one (this would be the case if i combine two actions
            # on the same substation)
            lines_impacted, subs_impacted = act.get_topological_impact()
            if np.sum(lines_impacted) + np.sum(subs_impacted) == 0:
                # action is do nothing
                warnings.warn("One of the provided action is do nothing. We will NOT add it there.")
                continue

            if np.sum(subs_impacted) > 1:
                raise RuntimeError(f"For now RandomAct1 (and especially RandomAct2) only works when provided actions "
                                   f"act each on at most one substation. Please check element {act_ref_id} of the "
                                   f"provided action")
            subs_id = np.where(subs_impacted)[0]
            if np.any(subs_impact[subs_id] > 1):
                if not warning_different_act_same_sub_issued:
                    subs_this = np.where(subs_impact[subs_id] > 1)[0]
                    warnings.warn(f"More than one unitary action affect substation: {subs_this}. Note that when "
                                  f"combining "
                                  f"actions, the actions acting on the same substation will not be combined "
                                  f"together, as "
                                  f"it would NOT result in a combined action otherwise.")
                    warning_different_act_same_sub_issued = True

            # TODO check that action is not already added, maybe ?

            subs_impact[subs_id] += 1
            # explicitly put `None` if nothing is modified
            self._subs_impacted_by_action.append(subs_id if subs_id.size > 0 else None)
            if subs_id.size == 0:
                # i found an action not acting on a substation
                all_act_topo = False
                self._nb_act[self._affect_lines] += 1
                self._subs_impacted_to_act_id[self._affect_lines].append(act_id)
            else:
                self._nb_act[subs_id] += 1
                for sub_id_tmp in subs_id:
                    self._subs_impacted_to_act_id[sub_id_tmp].append(act_id)

            self._all_actions.append(copy.deepcopy(act))
            act_id += 1

        if len(self._all_actions) == 0:
            # no action provided... impossible to build myself
            raise RuntimeError("You provided only invalid actions or no action at all. Please check `list_act` "
                               "and provide a suitable set of actions.")

        if all_act_topo:
            # all provided actions affect topology
            # i need to check that some of them affect different substation
            if np.sum(subs_impact >= 1) <= 1:
                # all actions affect the same substation...
                raise RuntimeError("All your actions affect a substation, and it happens they all affect THE SAME "
                                   "substation. This is not possible to use RandomAct1 in such circumstances ("
                                   "mainly because we want to combine the action from the agent so...)")

    def _combine_actions(self, act1, act2):
        """some kind of "overload" of the `+` operator of grid2op to take into account the disconnected powerline"""
        res = act1 + act2
        for act in (act1, act2):
            set_status = act.line_set_status
            li = [(l_id, -1) for l_id, el in enumerate(set_status) if el == -1]
            if li:
                # force disconnection of the disconnected powerline in this action
                res.line_or_set_bus = li
                res.line_ex_set_bus = li
                res.line_set_status = li
        return res

    def sample_act(self, previous_act_id: Union[int, None] = None, sub_id_act: Union[int, None] = None):
        """
        This function has a simple behaviour: it samples, uniformly at random, one action among the possible actions.

        Possible actions are determined by `previous_act_id` and `sub_id_act` (both should be integers or None)

        `previous_act_id` is ignored if `sub_id_act` is not None: in this case, we don't want to sample an
        action acting on the substation `sub_id_act` so this will include `previous_act_id`

        """
        if previous_act_id is None and sub_id_act is None:
            act_id = self.space_prng.choice(len(self._all_actions))
        else:
            if sub_id_act is not None:
                # i cannot sample an action on this substation
                nb_act = np.sum(self._nb_act[:sub_id_act]) + np.sum(self._nb_act[(sub_id_act + 1):])
                act_id_tmp = self.space_prng.choice(nb_act)
                # now i need to convert it to a "real" id:
                # for that i need to find which "type" this action can be
                is_act_topo_before_sub_id = np.cumsum(self._nb_act[:sub_id_act])
                if act_id_tmp < is_act_topo_before_sub_id[-1]:
                    this_sub = np.where(act_id_tmp < is_act_topo_before_sub_id)[0][0]
                    if this_sub > 0:
                        act_id_tmp -= is_act_topo_before_sub_id[this_sub]
                    act_id = self._subs_impacted_to_act_id[this_sub][act_id_tmp]
                else:
                    act_id_tmp -= is_act_topo_before_sub_id[-1]
                    is_act_topo_before_sub_id = np.cumsum(self._nb_act[(sub_id_act + 1):])
                    this_sub = np.where(act_id_tmp < is_act_topo_before_sub_id)[0][0]
                    # this_sub is after sub_id_act, it's NOT the real substation ID !
                    real_sub_id = this_sub + (sub_id_act + 1)
                    if this_sub > 0:
                        act_id_tmp -= is_act_topo_before_sub_id[this_sub - 1]
                    act_id = self._subs_impacted_to_act_id[real_sub_id][act_id_tmp]
            else:
                # i sampled a previous action, but that does not come from a substation, i just need to
                # sample randomly from all actions except the previous one
                act_id = self.space_prng.choice(len(self._all_actions) - 1)
                if act_id >= previous_act_id:
                    act_id += 1

        this_random_act = self._all_actions[act_id]
        sub_id_act = self._subs_impacted_by_action[act_id]
        return act_id, sub_id_act, this_random_act

    def act(self, obs, reward, done=False):
        act_id, sub_id_act, this_random = self.sample_act()
        res = self._combine_actions(self.all_to_one, this_random)
        return res
