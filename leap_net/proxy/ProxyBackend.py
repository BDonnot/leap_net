# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from leap_net.proxy.BaseProxy import BaseProxy

# this will be used to compute the DC approximation
from grid2op.dtypes import dt_int
from grid2op.Action._BackendAction import _BackendAction
from grid2op.Action import CompleteAction
from grid2op.Backend import PandaPowerBackend


class ProxyBackend(BaseProxy):
    """
    This class implement a "proxy" based on a grid2op backend.

    Only the default PandaPowerBackend is implemented here though the method used are generic and rely only the
    interface defined by `grid2op.Backend`.

    It should not cause any trouble to extend this class to deal with other type of Backends than PandaPowerBackend.
    """
    def __init__(self,
                 path_grid_json,  # complete path where the grid is represented as a json file
                 name="dc_approx",
                 is_dc=True,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # input that will be given to the proxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # output that we want the proxy to predict
                 ):
        BaseProxy.__init__(self,
                           name=name,
                           max_row_training_set=1,
                           eval_batch_size=1,
                           attr_x=attr_x,
                           attr_y=attr_y)

        # datasets
        self._supported_output = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"}
        self.is_dc = is_dc
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in self.attr_x:
                raise RuntimeError(f"The DC approximation need the variable \"{el}\" to be computed.")
        for el in self.attr_y:
            if not el in self._supported_output:
                raise RuntimeError(f"This solver cannot output the variable \"{el}\" at the moment. "
                                   f"Only possible outputs are \"{self._supported_output}\".")

        # specific part to dc model
        self.solver = PandaPowerBackend()
        self.solver.set_env_name(self.name)
        self.solver.load_grid(path_grid_json)  # the real powergrid of the environment
        self.solver.assert_grid_correct()
        self._bk_act_class = _BackendAction.init_grid(self.solver)
        self._act_class = CompleteAction.init_grid(self.solver)

        # internal variables (speed optimisation)
        self._indx_var = {}
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            self._indx_var[el] = self.attr_x.index(el)

    def build_model(self):
        """build the neural network used as proxy"""
        pass

    def init(self, obss):
        """
        initialize the meta data needed for the model to run (obss is a list of observations)

        One of the property of a backend is that it is (for PandaPower or LightSim at least) not able to compute
        more than one powerflow at a time.
        This is why we checked here that the dataset size was 1.

        Parameters
        ----------
        obss: ``list`` of ``grid2op.Observation``
            List of observations used to inialize this model, for example on which the model will compute the mean
            and standard deviation to scale the data.

        """
        if self.max_row_training_set != 1:
            raise RuntimeError("For now, a proxy based on a backend can only work with a database of 1 element ("
                               "the backend is applied sequentially to each element)")
        super().init(obss)
        # convert back topology to integer, because, well, topology is integer and not "self.dtype"
        arr_ = self._my_x[self._indx_var["topo_vect"]]
        self._my_x[self._indx_var["topo_vect"]] = arr_.astype(dt_int)

    def _extract_data(self, indx_train):
        """
        The mechanism to set a backend is a bit more complex than for other proxies based on neural networks
        for example.

        This is why we had to overload this function.
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
        compute the dc powerflow.

        In the formalism of grid2op backends, this is done with calling the function "runpf"
        """
        self.solver.runpf(is_dc=self.is_dc)
        return None

    def _post_process(self, predicted_state):
        """
        This is a little "hack" to retrieve from the backend only the data that are necessary (in the `_attr_y`).

        The idea here is to loop through the variables, and extract it from the solver (using the method
        `solver.lines_or_info`, `solver.lines_ex_info`, `solver.loads_info` and `solver.generators_info`

        Parameters
        ----------
        predicted_state: ``list`` of ``float``
            For each variables, it contains the (raw) predictions of the proxy

        Returns
        -------
        res: ``list`` of ``float``
            For each variables, it should return the post processed values.

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
