# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import os
import time
import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable

import tensorflow as tf
import tensorflow.keras.optimizers as tfko


class BaseProxy(ABC):
    """
    Base class you have to implement if you want to use easily a proxy
    """
    def __init__(self,
                 name,
                 lr=1e-4,
                 max_row_training_set=int(1e5),
                 train_batch_size=32,
                 eval_batch_size=1024):
        # name
        self.name = name

        # data type
        self.dtype = np.float32

        # model optimizer
        self._lr = lr
        self._schedule_lr_model = None
        self._optimizer_model = None

        # to fill the training / test dataset
        self.max_row_training_set = max_row_training_set
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        if self.max_row_training_set < self.train_batch_size:
            raise RuntimeError(f"You cannot use a batch size of {self.train_batch_size} with a dataset counting at"
                               f" most {self.max_row_training_set} rows. "
                               "Please either increase \"max_row_training_set\" or decrease \"batch_size\""
                               "(hint: batch_size>=max_row_training_set).")
        if self.max_row_training_set < self.eval_batch_size:
            raise RuntimeError(f"You cannot use a batch size of {self.eval_batch_size} with a dataset counting at"
                               f" most {self.max_row_training_set} rows. "
                               "Please either increase \"max_row_training_set\" or decrease \"batch_size\""
                               "(hint: batch_size>=max_row_training_set).")
        # training part
        self.train_iter = 0  # number of training iteration
        self.last_id = 0  # last index in the database
        self._global_iter = 0  # total number of data received
        self.__db_full = False  # is the "training database" full
        self.__need_save_graph = True  # save the tensorflow computation graph

        # the model
        self._model = None

        # timers
        self._time_predict = 0
        self._time_train = 0

        # for the prediction
        self._last_id_eval = 0

    #######################################################################
    ## All functions bellow should be implemented in your specific proxy ##
    #######################################################################
    @abstractmethod
    def build_model(self):
        """
        build the neural network used as proxy

        can be called multiple times

        """
        pass

    @abstractmethod
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

        pass

    @abstractmethod
    def init(self, obss):
        """initialize the meta data needed for the model to run (obss is a list of observations)"""
        pass

    @abstractmethod
    def load_metadata(self, dict_):
        """this function is used when loading the proxy to restore the meta data"""
        pass

    @abstractmethod
    def get_metadata(self):
        """should return a dictionary containing all the metadata of this class in a format that is compatible
        with json serialization.
        """
        pass

    @abstractmethod
    def get_output_sizes(self):
        """
        Should return the list of the dimension of the output of the proxy.

        This function should be overridden

        Returns
        -------

        """
        pass

    @abstractmethod
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
        # TODO refactorize with another method maybe
        pass

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
        return [f"output_{i}" for i, _ in enumerate(self.get_true_output(obs))]

    def store_obs(self, obs):
        """
        This method update all the intermediate for you.

        You need to make a derivate function and call "super().store_obs()"
        to benefit at maximum from this Proxy interface.

        This function should be overridden
        """
        # update the counters
        self._global_iter += 1
        self.last_id += 1
        if self.last_id >= self.max_row_training_set -1:
            self.__db_full = True
        self.last_id %= self.max_row_training_set

    #######################################################################
    ## All functions bellow can be implemented in your specific proxy ##
    #######################################################################
    def _get_mean(self, arr_, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)

        obss is a list of observation
        """
        add_, mul = self._get_adds_mults_from_name(obss, attr_nm)
        return add_

    def _get_sd(self, arr_, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)

        obss is a list of observation
        """
        add_, mul_ = self._get_adds_mults_from_name(obss, attr_nm)
        return mul_

    def save_tensorboard(self, tf_writer, training_iter, batch_losses):
        """
        save extra information to tensorboard

        This function can be overridden
        """
        pass

    def save_weights(self, path, ext=".h5"):
        """
        save the weights of the neural network
        path is the full path (including file name and extension)

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        This function is used when loading back your proxy

        Notes
        -----
        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"

        """
        self._model.save(os.path.join(path, f"weights{ext}"))

    def load_weights(self, path, ext=".h5"):
        """
        load the weight of the neural network
        path is the full path (including file name and extension).

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        This function is used when loading back your proxy.
        Notes
        -----
        This function is only called when the metadata (number of layer, size of each layer etc.)
        have been properly restored

        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"
        """
        self._model.load_weights(os.path.join(path, f"weights{ext}"))

    def _make_optimiser(self):
        """
        helper function to create the proper optimizer (Adam) with the learning rates and its decay
        parameters.

        This function can be overridden (for example if you don't use tensorflow).

        It's not part of the public API that is used outside of your proxy (private method).

        """
        # schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return None, tfko.Adam(learning_rate=self._lr)

    def _train_model(self, data):
        """
        perform the training step. For model coded in tensorflow in a regular supervised learning
        setting it can be summarize as follow:

        .. code-block:: python

            self._model.train_on_batch(*data)

        This function is called with something like:

        .. code-block:: python

            data = self._extract_data(indx_train)
            loss = self.train_model(data)

        This function can be overridden (for example if your proxy does not use tensorflow)

        It's not part of the public API that is used outside of your proxy (private method).

        Parameters
        ----------
        data:
            A tuple of np array that can be used for training

        Returns
        -------
        The loss of the batch

        """
        return self._model.train_on_batch(*data)

    def _make_predictions(self, data):
        """
        Make a prediction with the proxy on a new grid state.

        It's analogous to the `train_model` but instead of training it gives the prediction of the neural network.

        It's called with:

        .. code-block:: python

            data = self._extract_data([last_index])
            tmp = self.make_predictions(data)
            res = self._post_process(tmp)

        This function can be overridden (for example if your proxy does not use tensorflow)

        It's not part of the public API that is used outside of your proxy (private method).

        Parameters
        ----------
        data

        Returns
        -------

        """
        return self._model(data)

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

    #######################################################
    ## We don't recommend to change anything bellow this ##
    #######################################################
    def train(self, tf_writer=None):
        """
        Train the proxy (if tf_writer is not None, it is expected that the proxy save the computation graph

        We don't recommend to override this function.

        Parameters
        ----------
        tf_writer

        Returns
        -------
        None if the proxy has not been trained at this iteration, or the losses
        """
        if self._global_iter % self.train_batch_size != 0:
            return None

        if self.__db_full:
            tmp_max = self.max_row_training_set
        else:
            tmp_max = self.last_id
        indx_train = np.random.choice(np.arange(tmp_max),
                                      size=self.train_batch_size,
                                      replace=False)

        data = self._extract_data(indx_train)
        # for el in data[1]: print(np.mean(el))
        if tf_writer is not None and self.__need_save_graph:
            tf.summary.trace_on()

        beg_ = time.time()
        batch_losses = self._train_model(data)
        self._time_train += time.time() - beg_
        if tf_writer is not None and self.__need_save_graph:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            self.__need_save_graph = False
            tf.summary.trace_off()
        return batch_losses

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
        beg_ = time.time()
        res = self._make_predictions(data)
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
        elif attr_nm in ["load_v", "prod_v", "v_or", "v_ex"]:
            # default values are good enough
            pass
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
            mult_tmp = np.array([max(np.abs(val), 1.0) for val in getattr(obs, attr_nm)], dtype=self.dtype)
        elif attr_nm in ["a_or", "a_ex"]:
            mult_tmp = np.abs(obs.a_or / (obs.rho + 1e-2))  # which is equal to the thermal limit
            mult_tmp[mult_tmp <= 1e-2] = 1.0
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
