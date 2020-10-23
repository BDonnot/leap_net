# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import copy
import numpy as np

from leap_net.proxy.BaseProxy import BaseProxy

# this will be used to compute the DC approximation
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend


class ProxyBackend(BaseProxy):
    """
    This class implement a "proxy" based on a grid2op backend.

    Only the default PandaPowerBackend is implemented here.
    """
    def __init__(self,
                 path_grid_json,  # complete path where the grid is represented as a json file
                 name="dc_approx",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # TODO refacto that in BaseProxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # TODO refacto that in BaseProxy
                 is_dc=True
                 ):
        BaseProxy.__init__(self, name=name, lr=0, max_row_training_set=1, eval_batch_size=1, train_batch_size=1)

        # datasets
        self._supported_output = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        self.is_dc = is_dc
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in attr_x:
                raise RuntimeError(f"The DC approximation need the variable \"{el}\" to be computed.")
        for el in attr_y:
            if not el in self._supported_output:
                raise RuntimeError(f"This solver cannot output the variable \"{el}\" at the moment. "
                                   f"Only possible outputs are \"{self._supported_output}\".")

        # specific part to dc model
        self.attr_x = attr_x  # TODO refacto that in BaseProxy
        self.attr_y = attr_y  # TODO refacto that in BaseProxy
        self.solver = PandaPowerBackend()
        self.solver.set_env_name(self.name)
        self.solver.load_grid(path_grid_json)  # the real powergrid of the environment
        self.solver.assert_grid_correct()
        self._bk_act_class = _BackendAction.init_grid(self.solver)
        self._act_class = CompleteAction.init_grid(self.solver)

        # internal variables (speed optimisation)
        self._indx_var = {}
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            self._indx_var[el] = attr_x.index(el)

        # the dataset (computed on the fly)
        self._my_x = None  # TODO refacto that in BaseProxy
        self._my_y = None  # TODO refacto that in BaseProxy
        self._sz_x = None  # TODO refacto that in BaseProxy
        self._sz_y = None  # TODO refacto that in BaseProxy

    def build_model(self):
        """build the neural network used as proxy"""
        pass

    def store_obs(self, obs):
        """
        store the observation into the "training database"
        """
        # save the observation in the database
        for attr_nm, inp in zip(self.attr_x, self._my_x):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_y, self._my_y):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)

        super().store_obs(obs)

    def get_output_sizes(self):
        return copy.deepcopy(self._sz_y)

    def init(self, obss):
        """
        Initialize all the meta data and the database for training

        Parameters
        ----------
        obs

        Returns
        -------

        """
        if self.max_row_training_set != 1:
            raise RuntimeError("For now, a proxy based on a backend can only work with a database of 1 element ("
                               "the backend is applied sequentially to each element)")
        obs = obss[0]
        self.__db_full = False
        # save the input x
        self._my_x = []
        self._sz_x = []
        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_x.append(sz)
            self._my_x.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))

        # save the output y
        self._my_y = []
        self._sz_y = []
        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_y.append(sz)
            self._my_y.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))

    def get_true_output(self, obs):
        """
        Returns, from the observation the true output that has been computed by the environment.

        This "true output" is computed based on the observation and corresponds to what the proxy is meant to
        approximate (but the reference)
        Parameters
        ----------
        obs

        Returns
        -------

        """
        res = []
        for attr_nm in self.attr_y:
            res.append(self._extract_obs(obs, attr_nm))
        return res

    def get_metadata(self):
        """
        returns the metadata (model shapes, attribute used, sizes, etc.)

        this is used when saving the model
        """
        res = {}
        res["attr_x"] = [str(el) for el in self.attr_x]
        res["attr_y"] = [str(el) for el in self.attr_y]

        res["_sz_x"] = [int(el) for el in self._sz_x]
        res["_sz_y"] = [int(el) for el in self._sz_y]

        res["_time_train"] = float(self._time_train)
        res["_time_predict"] = float(self._time_predict)
        return res

    def load_metadata(self, dict_):
        """
        load the metadata of this neural network (also called meta parameters) from a dictionary

        Notes
        -----
        modify self!

        """
        self.attr_x = tuple([str(el) for el in dict_["attr_x"]])
        self.attr_y = tuple([str(el) for el in dict_["attr_y"]])

        self._sz_x = [int(el) for el in dict_["_sz_x"]]
        self._sz_y = [int(el) for el in dict_["_sz_y"]]

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

    def get_attr_output_name(self, obs):
        """
        Get the name (that will be used when saving the model) of each ouput of the proxy.

        It is recommended to overide this function

        Parameters
        ----------
        obs

        Returns
        -------

        """
        return copy.deepcopy(self.attr_y)

    def save_weights(self, path, ext=".h5"):
        pass

    def load_weights(self, path, ext=".h5"):
        pass

    def _extract_data(self, indx_train):
        """
        set the solver to the appropriate state
        """
        if indx_train.shape[0] != 1:
            raise RuntimeError("Proxy Backend only supports running on 1 state at a time. "
                               "Please set \"train_batch_size\" and \"eval_batch_size\" to 1.")
        res = self._bk_act_class()
        act = self._act_class()
        act.update({"set_bus": self._my_x[self._indx_var["topo_vect"]][0, :],
                    "injection": {
                        "prod_p": self._my_x[self._indx_var["prod_p"]][0, :],
                        "prod_v": self._my_x[self._indx_var["prod_v"]][0, :],
                        "load_p": self._my_x[self._indx_var["load_p"]][0, :],
                        "load_q": self._my_x[self._indx_var["load_q"]][0, :],
                        }
                    })
        res += act
        self.solver.apply_action(res)
        return None, None

    def _make_predictions(self, data, training=False):
        """
        compute the dc powerflow
        """
        self.solver.runpf(is_dc=self.is_dc)
        return None

    def _post_process(self, predicted_state):
        """
        retrieve the variables of interest from the backend

        Parameters
        ----------
        predicted_state

        Returns
        -------

        """
        predicted_state = []
        tmp = {}
        tmp["p_or"], tmp["q_or"], tmp["v_or"], tmp["a_or"] = self.solver.lines_or_info()
        tmp["p_ex"], tmp["q_ex"], tmp["v_ex"], tmp["a_ex"] = self.solver.lines_ex_info()
        tmp1, tmp2, tmp["load_v"] = self.solver.loads_info()
        tmp1, tmp["prod_q"], tmp2 = self.solver.generators_info()
        for el in self.attr_y:
            predicted_state.append(1. * tmp[el].reshape(1, -1))  # the "1.0 * " is here to force the copy...
        return predicted_state
