# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

import copy
import time
import warnings
import shutil
import tempfile

import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers as tfko
from tensorflow.keras.layers import Dense

from leap_net.proxy.BaseProxy import BaseProxy


class BaseNNProxy(BaseProxy):
    """
    This class serves as base class for all proxy made from neural network with tensorflow

    Attributes
    ----------
    train_batch_size: ``int``
        An integer > 0. It represents the training batch size of the proxy. When the proxy is trained, it tells
        how many "state" are feed at once.

    lr: ``float``
        The learning rate (discarded when the proxy do not need to be learned)
    """
    def __init__(self,
                 name,
                 lr=1e-4,
                 train_batch_size=32,
                 max_row_training_set=int(1e5),
                 eval_batch_size=1024,
                 layer=Dense,
                 layer_act=None,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  # input that will be given to the proxy
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),  # output that we want the proxy to predict
                 ):
        BaseProxy.__init__(self,
                           name=name,
                           max_row_training_set=max_row_training_set,
                           eval_batch_size=eval_batch_size,
                           attr_x=attr_x,
                           attr_y=attr_y
                           )

        self._layer_fun = layer
        self._layer_act = layer_act

        self._lr = lr

        self.train_iter = 0  # number of training iteration
        # model optimizer
        self._schedule_lr_model = None
        self._optimizer_model = None

        self.train_batch_size = train_batch_size
        if self.max_row_training_set < self.train_batch_size:
            raise RuntimeError(f"You cannot use a batch size of {self.train_batch_size} with a dataset counting at"
                               f" most {self.max_row_training_set} rows. "
                               "Please either increase \"max_row_training_set\" or decrease \"batch_size\""
                               "(hint: batch_size>=max_row_training_set).")

        self.__need_save_graph = True  # save the tensorflow computation graph

    def _make_predictions(self, data, training=False):
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
        return self._model(data, training=training)

    def save_data(self, path, ext=".h5"):
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

    def load_data(self, path, ext=".h5"):
        """
        load the weight of the neural network
        path is the full path (including file name and extension).

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        This function is used when loading back your proxy.

        Notes
        -----
        This function is only called when the metadata (number of layer, size of each layer etc.)
        have been properly restored (so it supposes load_metadata has been sucessfully called)

        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"

        The file at the "path" is not directly read. It is first copied into a temporary directory, and then
        is read. This is to avoid any data corruption when an instance of the model is reading and another
        is writing to the same file.
        """
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_file = f"weights{ext}"
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(os.path.join(path, nm_file), nm_tmp)
            # load this copy (make sure the proper file is not corrupted)
            self._model.load_weights(nm_tmp)

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
        losses = self._model.train_on_batch(*data)
        return losses

    def load_metadata(self, dict_):

        self.attr_x = tuple([str(el) for el in dict_["attr_x"]])
        self.attr_y = tuple([str(el) for el in dict_["attr_y"]])

        self._sz_x = [int(el) for el in dict_["_sz_x"]]
        self._sz_y = [int(el) for el in dict_["_sz_y"]]

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

        self._init_database_shapes()
        super().load_metadata(dict_)

    def get_metadata(self):
        res = super().get_metadata()
        if self._layer_act is not None:
            res["_layer_act"] = str(self._layer_act)
        else:
            # i don't store anything if it's None
            pass

        return res

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

        if self._is_db_full():
            tmp_max = self.max_row_training_set
        else:
            tmp_max = self.last_id

        # TODO seed
        # TODO mode "i keep only the last of the dataset
        indx_train = np.random.choice(np.arange(tmp_max),
                                      size=self.train_batch_size,
                                      replace=False)

        if self.DEBUG:
            indx_train = np.arange(self.train_batch_size)

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

    def _get_mean(self, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)

        obss is a list of observation
        """
        add_, mul = self._get_adds_mults_from_name(obss, attr_nm)
        return add_

    def _get_sd(self, obss, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data

        This function can be overridden (for example if you want more control on how to scale the data)

        obss is a list of observation
        """
        add_, mul_ = self._get_adds_mults_from_name(obss, attr_nm)
        return mul_
