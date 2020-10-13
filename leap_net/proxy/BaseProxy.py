# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Iterable

import tensorflow as tf
import tensorflow.keras.optimizers as tfko


class BaseProxy(ABC):
    """
    Base class you have to implement if you want to use easily a proxy
    """
    def __init__(self, name, lr=1e-4, max_row_training_set=int(1e5), batch_size=32):
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
        self.batch_size = batch_size

        # training part
        self.train_iter = 0  # number of training iteration
        self.last_id = 0  # last index in the database
        self._global_iter = 0  # total number of data received
        self.__db_full = False  # is the "training database" full
        self.__need_save_graph = True  # save the tensorflow computation graph

        # the model
        self._model = None

    #######################################################################
    ## All functions bellow should be implemented in your specific proxy ##
    #######################################################################
    @abstractmethod
    def build_model(self):
        """build the neural network used as proxy"""
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
    def init(self, obs):
        """initialize the meta data needed for the model to run"""
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
    def _get_mean(self, arr_, obs, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)

        """
        add_, mul = self._get_adds_mults_from_name(obs, attr_nm)
        return add_

    def _get_sd(self, arr_, obs, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)
        """
        add_, mul_ = self._get_adds_mults_from_name(obs, attr_nm)
        return mul_

    def save_tensorboard(self, tf_writer, training_iter, batch_losses):
        """
        save extra information to tensorboard

        This function can be overridden
        """
        pass

    def save_weights(self, path):
        """
        save the weights of the neural network
        path is the full path (including file name and extension)

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        """
        self._model.save(path)

    def load_weights(self, path):
        """
        load the weight of the neural network
        path is the full path (including file name and extension)

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        Notes
        -----
        This function is only called when the metadata (number of layer, size of each layer etc.)
         have been properly restored
        """

        self._model.load_weights(path)

    def _make_optimiser(self):
        """
        helper function to create the proper optimizer (Adam) with the learning rates and its decay
        parameters.

        This function can be overridden (for example if you don't use tensorflow).
        """
        # schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return None, tfko.Adam(learning_rate=self._lr)

    def train_model(self, data):
        """
        perform the training step. For model coded in tensorflow in a regular supervised learning
        setting it can be summarize as follow:

        .. code-block:: python

            self._model.train_on_batch(*data)

        This function can be overridden (for example if your proxy does not use tensorflow)

        Parameters
        ----------
        data:
            A tuple of np array that can be used for training

        Returns
        -------
        The loss of the batch

        """
        return self._model.train_on_batch(*data)

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
        if self._global_iter % self.batch_size != 0:
            return None

        if self.__db_full:
            tmp_max = self.max_row_training_set
        else:
            tmp_max = self.last_id
        indx_train = np.random.choice(np.arange(tmp_max),
                                      size=self.batch_size,
                                      replace=False)

        data = self._extract_data(indx_train)

        if tf_writer is not None and self.__need_save_graph:
            tf.summary.trace_on()
        batch_losses = self._model.train_on_batch(*data)
        if tf_writer is not None and self.__need_save_graph:
            with tf_writer.as_default():
                tf.summary.trace_export("model-graph", 0)
            self.__need_save_graph = False
            tf.summary.trace_off()
        return batch_losses

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

    def _get_adds_mults_from_name(self, obs, attr_nm):
        """
        extract the scalers (mean and std) used for the observation

        We don't recommend to overide this function, modify the function `_get_mean` and `_get_sd` instead

        """
        # TODO authorize some "blank run" to compute these scaler values
        if attr_nm in ["prod_p"]:
            add_tmp = np.array([0.5 * (pmax + pmin) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
                               dtype=self.dtype)
            mult_tmp = np.array([max((pmax - pmin), 0.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)],
                                dtype=self.dtype)
        elif attr_nm in ["prod_q"]:
            add_tmp = self.dtype(0.)
            mult_tmp = np.array([max(abs(val), 1.0) for val in obs.prod_q], dtype=self.dtype)
        elif attr_nm in ["load_p", "load_q"]:
            add_tmp = np.array([val for val in getattr(obs, attr_nm)], dtype=self.dtype)
            mult_tmp = self.dtype(2)
        elif attr_nm in ["load_v", "prod_v", "v_or", "v_ex"]:
            add_tmp = self.dtype(0.)
            mult_tmp = np.array([val for val in getattr(obs, attr_nm)], dtype=self.dtype)
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
            mult_tmp = np.array([(pmax - pmin) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)], dtype=self.dtype)
        elif attr_nm in ["a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex"]:
            add_tmp = self.dtype(0.)
            mult_tmp = np.array([max(val, 1.0) for val in getattr(obs, attr_nm)], dtype=self.dtype)
        elif attr_nm == "line_status":
            # encode back to 0: connected, 1: disconnected
            add_tmp = self.dtype(1.)
            mult_tmp = self.dtype(-1.0)
        else:
            add_tmp = self.dtype(0.)
            mult_tmp = self.dtype(1.0)
        return add_tmp, mult_tmp
