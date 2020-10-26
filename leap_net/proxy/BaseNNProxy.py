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

    We recommend to  inherit from this class if you are to code a neural network based proxy (using tensorflow).
    In particular the :func:`BaseNNProxy.train` function is not trivial to implement correctly.


    Attributes
    ----------
    train_iter: ``int``
        The current number of training iteration

    train_batch_size: ``int``
        An integer > 0. It represents the training batch size of the proxy. When the proxy is trained, it tells
        how many "state" are feed at once.

    _lr: ``float``
        The learning rate (discarded when the proxy do not need to be learned)

    _layer_fun: ``tensorflow.keras.layers``
        The "function" representing each layers. Be careful, this is not serialized and so if you want to
        save / reload the model, you need to specify the same function manually.

    _layer_act: ``str``
        The activation function of each layers. Should be a string.

    _optimizer_model:
        Represents the optimizer of the neural network

    _schedule_lr_model:
        internal, do not use
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
            tmp = self._make_predictions(data)
            res = self._post_process(tmp)

        This function can be overridden (for example if your proxy does not use tensorflow)

        It's not part of the public API that is used outside of your proxy (private method).

        Parameters
        ----------
        data:
            The input data on which the model will make the predictions (list of numpy arrays)
        training: ``bool``
            Whether this method is used to make prediction during training, or not

        Returns
        -------
        res:
            A list of numpy array that will be post process afterwards. In particular
             this list must count as many elements as there
            are elements in `attr_y`.
        """
        return self._model(data, training=training)

    def load_metadata(self, dict_):
        """
        this function is used when loading the proxy to restore the meta data

        This function may be overridden but in that case we recommend to call the method of the super class (like
        this is done here when calling `super().load_metadata(dict_)` )

        Notes
        -----
        This function is expected to modify the instance on which it is called (*ie* `self`)

        Parameters
        ----------
        dict_: ``dict``
            The dictionary of parameter that is used to initialize this instance.

        """

        self.attr_x = tuple([str(el) for el in dict_["attr_x"]])
        self.attr_y = tuple([str(el) for el in dict_["attr_y"]])

        self._sz_x = [int(el) for el in dict_["_sz_x"]]
        self._sz_y = [int(el) for el in dict_["_sz_y"]]

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

        self._init_database_shapes()
        super().load_metadata(dict_)

    def get_metadata(self):
        """
        should return a dictionary containing all the metadata of this class in a format that is compatible
        with json serialization.

        This function may be overridden but in that case we recommend to call the method of the super class (like
        is done here)

        Notes
        -----
        We assume that the metadata are "json serializable".

        Returns
        --------
        res: ``dict``
            A dictionary containing the necessary information to initialize the class as when it was saved.

        """
        res = super().get_metadata()
        if self._layer_act is not None:
            res["_layer_act"] = str(self._layer_act)
        else:
            # i don't store anything if it's None
            pass

        return res

    def save_data(self, path, ext=".h5"):
        """
        Save extra information that might be required by the model. For example this saves the weights of
        some neural network if needed.

        This function should be overridden

        Notes
        -----
        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"

        Parameters
        ----------
        path: ``str``
            The path at which the model will be saved
        ext: ``str``
            The extension used to save the model (for example ".h5" should output a file named xxx.h5)

        """
        self._model.save(os.path.join(path, f"weights{ext}"))

    def load_data(self, path, ext=".h5"):
        """
        You need to override this function if the proxy, in order to be functional need to load some data
        that are not loaded by the `load_metadata` function.

        This function may be overridden but in that case we recommend to call the method of the super class

        load the weight of the neural network
        path is the full path (including file name and extension).

        This function can be overridden (for example if your proxy does not use tensorflow or is made of multiple
        submodule that need to be saved independently)

        This function is used when loading back your proxy.

        Notes
        -----
        This function is only called when the metadata (number of layer, size of each layer etc.)
        have been properly restored (so it supposes load_metadata has been successfully called)

        We suppose that there is a "." preceding the extension. So ext=".h5" is valid, but not ext="h5"

        We recommend that the file at the "path" is not directly read.
        Rather, it is better to first copied it into a temporary directory, and then
        read it. This is to avoid any data corruption when an instance of the model is reading and another
        is writing to the same file.

        """
        with tempfile.TemporaryDirectory() as path_tmp:
            nm_file = f"weights{ext}"
            nm_tmp = os.path.join(path_tmp, nm_file)
            # copy the weights into this file
            shutil.copy(os.path.join(path, nm_file), nm_tmp)
            # load this copy (make sure the proper file is not corrupted)
            self._model.load_weights(nm_tmp)

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

    def _make_optimiser(self):
        """
        helper function to create the proper optimizer (Adam) with the learning rates and its decay
        parameters.

        This function can be overridden (for example if you don't use tensorflow).

        It's not part of the public API that is used outside of your proxy (private method).

        """
        # schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return None, tfko.Adam(learning_rate=self._lr)

    def save_tensorboard(self, tf_writer, training_iter, batch_losses):
        """
        save extra information to tensorboard

        In this case i save all the losses for all individual output
        """
        for output_nm, loss in zip(self.attr_y, batch_losses):
            tf.summary.scalar(f"{output_nm}", loss, training_iter,
                              description=f"MSE for {output_nm}")

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
