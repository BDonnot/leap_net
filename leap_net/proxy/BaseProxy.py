# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import os
import time
import copy
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable


class BaseProxy(ABC):
    """
    Base class you have to implement if you want to use easily a proxy

    Attributes
    ----------

    name: ``str``
        The name of the proxy (used for example when saving or restoring data)

    max_row_training_set: ``int``
        An integer > 0. It represents the maximum of rows there will be in the "training set" of our proxy.

    eval_batch_size: ``int``
        An integer > 0. It represents the batch size of the proxy when it is used to make predictions. Sometimes
        for computation time reasons, it can be useful to batch multiple data and predict more than one state at
        the same time.

    # TODO documentation of the database, attr_x, and attr_y
    """
    DEBUG = False

    def __init__(self,
                 name,
                 max_row_training_set=int(1e5),
                 eval_batch_size=1,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # input that will be given to the proxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # output that we want the proxy to predict
                 ):
        # name
        self.name = name

        # data type
        self.dtype = np.float32

        # model optimizer
        self._schedule_lr_model = None
        self._optimizer_model = None

        # to fill the training / test dataset
        self.max_row_training_set = max_row_training_set
        self.eval_batch_size = eval_batch_size
        if self.max_row_training_set < self.eval_batch_size:
            raise RuntimeError(f"You cannot use a batch size of {self.eval_batch_size} with a dataset counting at"
                               f" most {self.max_row_training_set} rows. "
                               "Please either increase \"max_row_training_set\" or decrease \"batch_size\""
                               "(hint: batch_size>=max_row_training_set).")
        # training part
        self.train_iter = 0  # number of training iteration
        self.last_id = 0  # last index in the database
        self._global_iter = 0  # total number of data received
        self.__db_full = None  # is the "training database" full
        self.__first_eval = True

        # the model
        self._model = None

        # timers
        self._time_predict = 0
        self._time_train = 0

        # for the prediction
        self._last_id_eval = 0

        # database part
        self.attr_x = attr_x
        self.attr_y = attr_y
        self._my_x = None
        self._my_y = None
        self._sz_x = None
        self._sz_y = None
        self._metadata_loaded = False

    #######################################################################
    ## All functions bellow should be implemented in your specific proxy ##
    #######################################################################
    @abstractmethod
    def build_model(self):
        """
        build the model that is used inside of the proxy

        can be called multiple times

        """
        pass

    @abstractmethod
    def _make_predictions(self, data, training=False):
        """
        Make a prediction with the proxy on a new grid state.

        It's analogous to the `train_model` but instead of training it gives the prediction of the neural network.

        It's called with:

        .. code-block:: python

            data = self._extract_data([last_index])
            tmp = self.make_predictions(data)
            res = self._post_process(tmp)

        It's not part of the public API that is used outside of your proxy (private method), and it is only used
        with the given code above.

        Parameters
        ----------
        data

        Returns
        -------

        """
        pass

    #######################################################################
    ## All functions bellow can be implemented in your specific but if   ##
    ## this is the case we recommend to call the method of this base class ##
    ########################################################################
    def init(self, obss):
        """
        initialize the meta data needed for the model to run (obss is a list of observations)

        This function may be overridden but in that case we recommend to call the method of the super class.

        If it's overridden, we also recommend to call `_init_database_shapes`

        """

        self.__db_full = False
        obs = obss[0]
        # save the input x
        self._my_x = []
        self._sz_x = []
        self._my_y = []
        self._sz_y = []

        # init the dimension of everything
        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_x.append(sz)
        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obs, attr_nm)
            sz = arr_.size
            self._sz_y.append(sz)
        self._init_database_shapes()

    def _init_database_shapes(self):
        """
        init the database to have the proper size.

        This function is automatically called by the "init" method or by the "load_metadata" method.

        This function may be overridden but in that case we recommend to call the method of the super class
        """
        # init the database
        self._my_x = []
        for sz in self._sz_x:
            self._my_x.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
        self._my_y = []
        for sz in self._sz_y:
            self._my_y.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))

    def store_obs(self, obs):
        """
        This method update all the intermediate for you.

        You need to make a derivate function and call "super().store_obs()"
        to benefit at maximum from this Proxy interface.

        This function may be overridden but in that case we recommend to call the method of the super class
        """
        # update the counters
        # store only the first batch if DEBUG otherwise store all
        for attr_nm, inp in zip(self.attr_x, self._my_x):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_y, self._my_y):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)

        self._global_iter += 1
        self.last_id += 1
        if self.last_id >= self.max_row_training_set - 1:
            self.__db_full = True
        self.last_id %= self.max_row_training_set

    def load_metadata(self, dict_):
        """
        this function is used when loading the proxy to restore the meta data

        This function may be overridden but in that case we recommend to call the method of the super class

        Notes
        -----
        This function is expected to modify the instance on which it is called (*ie* `self`)

        """
        self.attr_x = tuple([str(el) for el in dict_["attr_x"]])
        self.attr_y = tuple([str(el) for el in dict_["attr_y"]])

        self._sz_x = [int(el) for el in dict_["_sz_x"]]
        self._sz_y = [int(el) for el in dict_["_sz_y"]]

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

        self._init_database_shapes()

        self._metadata_loaded = True

    def get_metadata(self):
        """
        should return a dictionary containing all the metadata of this class in a format that is compatible
        with json serialization.

        This function may be overridden but in that case we recommend to call the method of the super class

        Notes
        -----
        We assume that the metadata are "json serializable".

        """
        res = dict()
        res["attr_x"] = [str(el) for el in self.attr_x]
        res["attr_y"] = [str(el) for el in self.attr_y]
        res["_sz_x"] = [int(el) for el in self._sz_x]
        res["_sz_y"] = [int(el) for el in self._sz_y]

        res["_time_train"] = float(self._time_train)
        res["_time_predict"] = float(self._time_predict)

        return res

    def load_data(self, path, ext=".h5"):
        """
        You need to override this function if the proxy, in order to be functional need to load some data
        that are not loaded by the `load_metadata` function.

        This function may be overridden but in that case we recommend to call the method of the super class

        Returns
        -------

        """
        pass

    def save_data(self, path, ext=".h5"):
        """
        Save extra information that might be required by the model. For example this saves the weights of
        some neural network if needed.

        Notes
        -----
        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"

        """
        pass

    #######################################################################
    ##  All functions bellow can be implemented in your specific proxy   ##
    #######################################################################
    def _extract_data(self, indx_train):
        """
        extract from the training dataset, the data with indexes `indx_train`

        The model will be trained with :

        .. code-block:: python

            data = self._extract_data(indx_train)
            batch_losses = self._model.train_on_batch(*data)

        This function can be overridden

        Returns
        -------

        """

        tmpx = [arr[indx_train, :] for arr in self._my_x]
        tmpy = [arr[indx_train, :] for arr in self._my_y]
        return tmpx, tmpy

    def get_output_sizes(self):
        """
        Should return the list of the dimension of the output of the proxy.

        This function can be overridden

        Returns
        -------

        """
        return copy.deepcopy(self._sz_y)

    def get_attr_output_name(self, obs):
        """
        Get the name (that will be used when saving the model) of each ouput of the proxy.

        This function may be overridden

        Parameters
        ----------
        obs

        Returns
        -------

        """
        return copy.deepcopy(self.attr_y)

    def get_true_output(self, obs):
        """
        Returns, from the observation the true output that has been computed by the environment.

        This "true output" is computed based on the observation and corresponds to what the proxy is meant to
        approximate (but the reference)

        This function may be overridden

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

    def save_tensorboard(self, tf_writer, training_iter, batch_losses):
        """
        save extra information to tensorboard

        This function can be overridden
        """
        pass

    def _post_process(self, predicted_state):
        """
        This function is used to post process the data that are the output of the proxy.

        Parameters
        ----------
        predicted_state

        Returns
        -------

        """
        return predicted_state

    #######################################################
    ## We don't recommend to change anything bellow this ##
    #######################################################
    def train(self, tf_writer=None):
        """
        Train the proxy (if tf_writer is not None, it is expected that the proxy save the computation graph

        If your model need to be trained, we recommend you to inherit from "BaseNNProxy" instead

        Parameters
        ----------
        tf_writer

        Returns
        -------
        None if the proxy has not been trained at this iteration, or the losses
        """
        raise RuntimeError("This model cannot be trained!")

    def predict(self, force=False):
        """
        Make the prediction using the proxy.

        Prediction are made "on the fly", which means that the batch size is 1. TODO this is not true anymore

        We do not recommend to override this function.

        Notes
        -----
        It can only be called if the proxy has been "filled" with the observations before.

        It can be filled by batches

        Returns
        -------
        res:
            All the predictions made by the proxy. Note that these predictions should be directly comparable with
            the input data and (so if a scaling is applied, they should be unscaled)
        """
        if (self._global_iter % self.eval_batch_size != 0) and (not force):
            return None
        data = self._extract_data(np.arange(self._last_id_eval, self._global_iter) % self.max_row_training_set)

        if self.__first_eval:
            # evaluate at "blank" the first time so that tensorflow / keras can load the model
            res = self._make_predictions(data, training=False)
            self.__first_eval = False

        beg_ = time.time()
        res = self._make_predictions(data, training=False)
        self._time_predict += time.time() - beg_
        res = self._post_process(res)
        self._last_id_eval = self._global_iter
        return res

    def _save_dict(self, li, val):
        """
        save the metadata of this proxy into a valid representation of self.

        It is a helper to convert data in float format from either a list or a single numpy floating point.

        We don't recommend to override this function.

        Parameters
        ----------
        li
        val

        Returns
        -------

        """
        if isinstance(val, Iterable):
            li.append([float(el) for el in val])
        else:
            li.append(float(val))

    def _add_attr(self, attr_nm, val):
        """
        add an attribute to myself based on the value (that can either be a list or a single element)
        used in "from_dict" for example

        We don't recommend to override this function.

        """
        if isinstance(val, Iterable):
            getattr(self, attr_nm).append(np.array(val).astype(self.dtype))
        else:
            getattr(self, attr_nm).append(self.dtype(val))

    def _get_adds_mults_from_name(self, obss, attr_nm):
        """
        extract the scalers (mean and std) used for the observation

        We don't recommend to overide this function, modify the function `_get_mean` and `_get_sd` instead

        obss is a list of observation obtained from running some environment with just the "actor" acting on
        the grid. The size of this list is set by `AgentWithProxy.nb_obs_init`

        """
        obs = obss[0]
        add_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        mult_tmp = np.std([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype) + 1e-1

        if attr_nm in ["prod_p"]:
            # mult_tmp = np.array([max((pmax - pmin), 1.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
            #                     dtype=self.dtype)
            # default values are good enough
            pass
        elif attr_nm in ["prod_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_p", "load_q"]:
            # default values are good enough
            pass
        elif attr_nm in ["load_v", "prod_v"]:
            # default values are good enough
            # stds are almost 0 for loads, this leads to instability
            add_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
            mult_tmp = 1.0  # np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        elif attr_nm in ["v_or", "v_ex"]:
            # default values are good enough
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            mult_tmp = np.mean([self._extract_obs(ob, attr_nm) for ob in obss], axis=0).astype(self.dtype)
        elif attr_nm == "hour_of_day":
            add_tmp = self.dtype(12.)
            mult_tmp = self.dtype(12.)
        elif attr_nm == "minute_of_hour":
            add_tmp = self.dtype(30.)
            mult_tmp = self.dtype(30.)
        elif attr_nm == "day_of_week":
            add_tmp = self.dtype(4.)
            mult_tmp = self.dtype(4)
        elif attr_nm == "day":
            add_tmp = self.dtype(15.)
            mult_tmp = self.dtype(15.)
        elif attr_nm in ["target_dispatch", "actual_dispatch"]:
            add_tmp = self.dtype(0.)
            mult_tmp = np.array([max((pmax - pmin), 1.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
                                dtype=self.dtype)
        elif attr_nm in ["p_or", "p_ex", "q_or", "q_ex"]:
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            mult_tmp = np.array([max(np.abs(val), 1.0) for val in getattr(obs, attr_nm)], dtype=self.dtype)
        elif attr_nm in ["a_or", "a_ex"]:
            add_tmp = self.dtype(0.)  # because i multiply by the line status, so i don't want any bias
            mult_tmp = np.abs(obs.a_or / (obs.rho + 1e-2))  # which is equal to the thermal limit
            mult_tmp[mult_tmp <= 1.] = 1.0
        elif attr_nm == "line_status":
            # encode back to 0: connected, 1: disconnected
            add_tmp = self.dtype(1.)
            mult_tmp = self.dtype(-1.0)

        return add_tmp, mult_tmp

    def get_total_predict_time(self):
        """
        get the total time spent to make the prediction with the proxy

        We don't recommend to overide this function
        """
        return self._time_predict

    def _is_db_full(self):
        return self.__db_full

    def _extract_obs(self, obs, attr_nm):
        """
        Extract a given attribute from an observation.

        This function can be overridden if you want to extract non attribute (for example results of a method) of
        an observation.

        Parameters
        ----------
        obs:
            Grid2op action on which we want to extract something

        attr_nm:
            Name of the attribute we want to extract

        Returns
        -------
        res:
            The array representing what need to be extracted from the observation
        """
        return getattr(obs, attr_nm)
