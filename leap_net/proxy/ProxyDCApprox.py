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
from grid2op.Backend import PandaPowerBackend

# TODO
class ProxyDCApprox(BaseProxy):
    def __init__(self,
                 path_grid_json,  # complete path where the grid is represented as a json file
                 name="dc_approx",
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v"),
                 ):
        BaseProxy.__init__(self, name=name, lr=0, max_row_training_set=1, eval_batch_size=1, train_batch_size=1)

        # datasets
        for el in ("prod_p", "prod_v", "load_p", "load_q", "topo_vect"):
            if not el in attr_x:
                raise RuntimeError(f"The DC approximation need the variable \"{el}\" to be computed.")

        # specific part to leap net model
        self.attr_x = attr_x
        self.attr_y = attr_y
        self.solver = PandaPowerBackend()
        self.solver.init


    def build_model(self):
        """build the neural network used as proxy"""
        if self._model is not None:
            # model is already initialized
            return
        #TODO init the pandapower grid

    def store_obs(self, obs):
        """
        store the observation into the "training database"
        """
        super().store_obs(obs)

        # save the observation in the database
        for attr_nm, inp in zip(self.attr_x, self._my_x):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_tau, self._my_tau):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_y, self._my_y):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)

    def get_output_sizes(self):
        return copy.deepcopy(self._sz_y)

    def init(self, obs):
        """
        Initialize all the meta data and the database for training

        Parameters
        ----------
        obs

        Returns
        -------

        """
        self.__db_full = False
        # save the input x
        self._my_x = []
        self._m_x = []
        self._sd_x = []
        self._sz_x = []
        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_x.append(sz)
            self._my_x.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            self._m_x.append(self._get_mean(arr_, obs, attr_nm))
            self._sd_x.append(self._get_sd(arr_, obs, attr_nm))

        # save the output y
        self._my_y = []
        self._m_y = []
        self._sd_y = []
        self._sz_y = []
        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_y.append(sz)
            self._my_y.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            self._m_y.append(self._get_mean(arr_, obs, attr_nm))
            self._sd_y.append(self._get_sd(arr_, obs, attr_nm))

        # save the tau vectors
        self._my_tau = []
        self._m_tau = []
        self._sd_tau = []
        self._sz_tau = []
        for attr_nm in self.attr_tau:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_tau.append(sz)
            self._my_tau.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            self._m_tau.append(self._get_mean(arr_, obs, attr_nm))
            self._sd_tau.append(self._get_sd(arr_, obs, attr_nm))

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
        res["attr_tau"] = [str(el) for el in self.attr_tau]
        res["attr_y"] = [str(el) for el in self.attr_y]

        res["_m_x"] = []
        for el in self._m_x:
            self._save_dict(res["_m_x"], el)
        res["_m_y"] = []
        for el in self._m_y:
            self._save_dict(res["_m_y"], el)
        res["_m_tau"] = []
        for el in self._m_tau:
            self._save_dict(res["_m_tau"], el)
        res["_sd_x"] = []
        for el in self._sd_x:
            self._save_dict(res["_sd_x"], el)
        res["_sd_y"] = []
        for el in self._sd_y:
            self._save_dict(res["_sd_y"], el)
        res["_sd_tau"] = []
        for el in self._sd_tau:
            self._save_dict(res["_sd_tau"], el)

        res["_sz_x"] = [int(el) for el in self._sz_x]
        res["_sz_y"] = [int(el) for el in self._sz_y]
        res["_sz_tau"] = [int(el) for el in self._sz_tau]

        res["sizes_enc"] = [int(el) for el in self.sizes_enc]
        res["sizes_main"] = [int(el) for el in self.sizes_main]
        res["sizes_out"] = [int(el) for el in self.sizes_out]

        res["_time_train"] = float(self._time_train)
        res["_time_predict"] = float(self._time_predict)
        return res

    def save_tensorboard(self, tf_writer, training_iter, batch_losses):
        """save extra information to tensorboard"""
        for output_nm, loss in zip(self.attr_y, batch_losses):
            tf.summary.scalar(f"{output_nm}", loss, training_iter,
                              description=f"MSE for {output_nm}")

        # TODO add the "evaluate on validation episode"

    def load_metadata(self, dict_):
        """
        load the metadata of this neural network (also called meta parameters) from a dictionary

        Notes
        -----
        modify self!

        """
        self.attr_x = tuple([str(el) for el in dict_["attr_x"]])
        self.attr_tau = tuple([str(el) for el in dict_["attr_tau"]])
        self.attr_y = tuple([str(el) for el in dict_["attr_y"]])

        for key in ["_m_x", "_m_y", "_m_tau", "_sd_x", "_sd_y", "_sd_tau"]:
            setattr(self, key, [])
            for el in dict_[key]:
                self._add_attr(key, el)

        self._sz_x = [int(el) for el in dict_["_sz_x"]]
        self._sz_y = [int(el) for el in dict_["_sz_y"]]
        self._sz_tau = [int(el) for el in dict_["_sz_tau"]]

        self.sizes_enc = [int(el) for el in dict_["sizes_enc"]]
        self.sizes_main = [int(el) for el in dict_["sizes_main"]]
        self.sizes_out = [int(el) for el in dict_["sizes_out"]]

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

    def _extract_data(self, indx_train):
        """
        extract from the training dataset, the data with indexes indx_train

        The model will be trained with :

        .. code-block:: python

            data = self._extract_data(indx_train)
            batch_losses = self._model.train_on_batch(*data)

        Returns
        -------

        """
        tmpx = [tf.convert_to_tensor((arr[indx_train, :] - m_) / sd_) for arr, m_, sd_ in zip(self._my_x, self._m_x, self._sd_x)]
        tmpy = [tf.convert_to_tensor((arr[indx_train, :] - m_) / sd_) for arr, m_, sd_ in zip(self._my_y, self._m_y, self._sd_y)]
        tmpt = [tf.convert_to_tensor((arr[indx_train, :] - m_) / sd_) for arr, m_, sd_ in zip(self._my_tau, self._m_tau, self._sd_tau)]
        return (tmpx, tmpt), tmpy

    def _post_process(self, predicted_state):
        """
        This function is used to post process the data that are the output of the proxy.

        Parameters
        ----------
        predicted_state

        Returns
        -------

        """
        tmp = [el.numpy() for el in predicted_state]
        resy = [arr * sd_ + m_ for arr, m_, sd_ in zip(tmp, self._m_y, self._sd_y)]
        return resy

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