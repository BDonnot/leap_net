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

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import multiply as tfk_multiply

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input

from leap_net.proxy.BaseProxy import BaseProxy
from leap_net.LtauNoAdd import LtauNoAdd


class ProxyLeapNet(BaseProxy):
    def __init__(self,
                 name="leap_net",
                 max_row_training_set=int(1e5),
                 train_batch_size=32,
                 eval_batch_size=1024,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                 attr_tau=("line_status",),
                 sizes_enc=(20, 20, 20),
                 sizes_main=(150, 150, 150),
                 sizes_out=(100, 40),
                 lr=1e-4,
                 scale_main_layer=None,  # increase the size of the main layer
                 scale_input_dec_layer=None,  # scale the input of the decoder
                 scale_input_enc_layer=None,  # scale the input of the encoder
                 layer=Dense,  # TODO (for save and restore)
                 layer_act=None  # TODO (for save and restore)
                 ):
        BaseProxy.__init__(self,
                           name=name,
                           lr=lr,
                           max_row_training_set=max_row_training_set,
                           train_batch_size=train_batch_size,
                           eval_batch_size=eval_batch_size)

        self._layer_fun = layer
        self._layer_act = layer_act

        # datasets
        self._my_x = None
        self._my_y = None
        self._my_tau = None

        # sizes
        self._sz_x = None
        self._sz_y = None
        self._sz_tau = None

        # scaler
        self._m_x = None
        self._m_y = None
        self._m_tau = None
        self._sd_x = None
        self._sd_y = None
        self._sd_tau = None

        # specific part to leap net model
        # TODO to make sure it's integers
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out = sizes_out
        self.attr_x = attr_x
        self.attr_y = attr_y
        self.attr_tau = attr_tau
        self._scale_main_layer = scale_main_layer
        self._scale_input_dec_layer = scale_input_dec_layer
        self._scale_input_enc_layer = scale_input_enc_layer

        # not to load multiple times the meta data
        self._metadata_loaded = False
        self.tensor_line_status = None

        # small stuff with powerlines
        self._line_attr = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "v_or", "v_ex"}
        self._idx = None
        self._where_id = None
        self.tensor_line_status = None
        try:
            self._idx = self.attr_tau.index("line_status")
            self._where_id = "tau"
        except ValueError:
            try:
                self._idx = self.attr_x.index("line_status")
                self._where_id = "x"
            except ValueError:
                warnings.warn("We strongly recommend you to get the \"line_status\" as an input vector")

    def build_model(self):
        """build the neural network used as proxy"""
        if self._model is not None:
            # model is already initialized
            return
        self._model = Sequential()
        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                    zip(self._sz_x, self.attr_x)]
        inputs_tau = [Input(shape=(el,), name="tau_{}".format(nm_)) for el, nm_ in
                      zip(self._sz_tau, self.attr_tau)]

        # tensor_line_status = None
        if self._idx is not None:
            # line status is encoded: 1 disconnected, 0 connected
            # I invert it here
            if self._where_id == "x":
                self.tensor_line_status = inputs_x[self._idx]
            elif self._where_id == "tau":
                self.tensor_line_status = inputs_tau[self._idx]
            else:
                raise RuntimeError("Unknown \"where_id\"")
            self.tensor_line_status = 1.0 - self.tensor_line_status

        # encode each data type in initial layers
        encs_out = []
        for init_val, nm_ in zip(inputs_x, self.attr_x):
            lay = init_val

            if self._scale_input_enc_layer is not None:
                # scale up to have higher dimension
                lay = Dense(self._scale_input_enc_layer,
                            name=f"scaling_input_encoder_{nm_}")(lay)
            for i, size in enumerate(self.sizes_enc):
                lay_fun = self._layer_fun(size,
                                          name="enc_{}_{}".format(nm_, i),
                                          activation=self._layer_act)
                lay = lay_fun(lay)
                if self._layer_act is None:
                    # add a non linearity if not added in the layer
                    lay = Activation("relu")(lay)
            encs_out.append(lay)

        # concatenate all that
        lay = tf.keras.layers.concatenate(encs_out)

        if self._scale_main_layer is not None:
            # scale up to have higher dimension
            lay = Dense(self._scale_main_layer, name="scaling_inputs")(lay)

        # i do a few layer
        for i, size in enumerate(self.sizes_main):
            lay_fun = self._layer_fun(size,
                                      name="main_{}".format(i),
                                      activation=self._layer_act)
            lay = lay_fun(lay)
            if self._layer_act is None:
                # add a non linearity if not added in the layer
                lay = Activation("relu")(lay)

        # now i do the leap net to encode the state
        encoded_state = lay
        for input_tau, nm_ in zip(inputs_tau, self.attr_tau):
            tmp = LtauNoAdd(name=f"leap_{nm_}")([lay, input_tau])
            encoded_state = tf.keras.layers.add([encoded_state, tmp], name=f"adding_{nm_}")

        # i predict the full state of the grid given the input variables
        outputs_gm = []
        model_losses = {}
        # model_losses = []
        lossWeights = {}  # TODO
        for sz_out, nm_ in zip(self._sz_y,
                               self.attr_y):
            lay = encoded_state
            if self._scale_input_dec_layer is not None:
                # scale up to have higher dimension
                lay = Dense(self._scale_input_dec_layer,
                            name=f"scaling_input_decoder_{nm_}")(lay)
                lay = Activation("relu")(lay)

            for i, size in enumerate(self.sizes_out):
                lay_fun = self._layer_fun(size,
                                          name="{}_{}".format(nm_, i),
                                          activation=self._layer_act)
                lay = lay_fun(lay)
                if self._layer_act is None:
                    # add a non linearity if not added in the layer
                    lay = Activation("relu")(lay)

            # predict now the variable
            name_output = "{}_hat".format(nm_)
            # force the model to output 0 when the powerline is disconnected
            if self.tensor_line_status is not None and nm_ in self._line_attr:
                pred_ = Dense(sz_out, name=f"{nm_}_force_disco")(lay)
                pred_ = tfk_multiply((pred_, self.tensor_line_status), name=name_output)
            else:
                pred_ = Dense(sz_out, name=name_output)(lay)

            outputs_gm.append(pred_)
            model_losses[name_output] = "mse"
            # model_losses.append(tf.keras.losses.mean_squared_error)

        # now create the model in keras
        self._model = Model(inputs=(inputs_x, inputs_tau),
                            outputs=outputs_gm,
                            name="model")
        # and "compile" it
        self._schedule_lr_model, self._optimizer_model = self._make_optimiser()
        self._model.compile(loss=model_losses, optimizer=self._optimizer_model)

    def store_obs(self, obs):
        """
        store the observation into the "training database"
        """
        # save the observation in the database
        if not self.DEBUG or self._global_iter < self.train_batch_size:
            # store only the first batch if DEBUG otherwise store all
            for attr_nm, inp in zip(self.attr_x, self._my_x):
                inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
            for attr_nm, inp in zip(self.attr_tau, self._my_tau):
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
        self.__db_full = False
        # save the input x
        self._my_x = []
        if not self._metadata_loaded:
            self._m_x = []
            self._sd_x = []
            self._sz_x = []
        for attr_nm in self.attr_x:
            arr_ = self._extract_obs(obss[0], attr_nm)
            sz = arr_.size
            self._my_x.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            if not self._metadata_loaded:
                self._sz_x.append(sz)
                self._m_x.append(self._get_mean(arr_, obss, attr_nm))
                self._sd_x.append(self._get_sd(arr_, obss, attr_nm))

        # save the output y
        self._my_y = []
        if not self._metadata_loaded:
            self._m_y = []
            self._sd_y = []
            self._sz_y = []
        for attr_nm in self.attr_y:
            arr_ = self._extract_obs(obss[0], attr_nm)
            sz = arr_.size
            self._my_y.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            if not self._metadata_loaded:
                self._sz_y.append(sz)
                self._m_y.append(self._get_mean(arr_, obss, attr_nm))
                self._sd_y.append(self._get_sd(arr_, obss, attr_nm))

        # save the tau vectors
        self._my_tau = []
        if not self._metadata_loaded:
            self._m_tau = []
            self._sd_tau = []
            self._sz_tau = []
        for attr_nm in self.attr_tau:
            arr_ = self._extract_obs(obss[0], attr_nm)
            sz = arr_.size
            self._my_tau.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))
            if not self._metadata_loaded:
                self._sz_tau.append(sz)
                self._m_tau.append(self._get_mean(arr_, obss, attr_nm))
                self._sd_tau.append(self._get_sd(arr_, obss, attr_nm))

        self._metadata_loaded = True

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

        if self._scale_main_layer is not None:
            res["_scale_main_layer"] = int(self._scale_main_layer)
        else:
            # i don't store anything is it's None
            pass
        if self._scale_input_dec_layer is not None:
            res["_scale_input_dec_layer"] = int(self._scale_input_dec_layer)
        else:
            # i don't store anything is it's None
            pass
        if self._scale_input_enc_layer is not None:
            res["_scale_input_enc_layer"] = int(self._scale_input_enc_layer)
        else:
            # i don't store anything is it's None
            pass
        if self._layer_act is not None:
            res["_layer_act"] = str(self._layer_act)
        else:
            # i don't store anything is it's None
            pass

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
        if "_scale_main_layer" in dict_:
            self._scale_main_layer = int(dict_["_scale_main_layer"])
        else:
            self._scale_main_layer = None
        if "_scale_input_dec_layer" in dict_:
            self._scale_input_dec_layer = int(dict_["_scale_input_dec_layer"])
        else:
            self._scale_input_dec_layer = None
        if "_scale_input_enc_layer" in dict_:
            self._scale_input_enc_layer = int(dict_["_scale_input_enc_layer"])
        else:
            self._scale_input_enc_layer = None
        if "_layer_act" in dict_:
            self._layer_act = str(dict_["_layer_act"])
        else:
            self._layer_act = None

        self._time_train = float(dict_["_time_train"])
        self._time_predict = float(dict_["_time_predict"])

        self._metadata_loaded = True

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


        # tf.convert_to_tensor(
        tmpx = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_x, self._m_x, self._sd_x)]
        tmpt = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_tau, self._m_tau, self._sd_tau)]
        tmpy = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_y, self._m_y, self._sd_y)]

        # tmp_line_status = 1.0
        # TODO if i do it here, i need to do it also on the post process, and this is not great
        # if self._idx is not None:
        #     if self._where_id == "tau":
        #         tmp_line_status = tmpt[self._idx]
        #     elif self._where_id == "x":
        #         tmp_line_status = tmpx[self._idx]
        #     else:
        #         raise RuntimeError("Unknown self._where_id")
        # tmpy = [tf.convert_to_tensor((arr[indx_train, :] - m_) / sd_ * tmp_line_status if attr_n in self.line_attr else 1.0)
        #         for arr, m_, sd_, attr_n in zip(self._my_y, self._m_y, self._sd_y, self.attr_y)]

        tmpx = [tf.convert_to_tensor(el) for el in tmpx]
        tmpt = [tf.convert_to_tensor(el) for el in tmpt]
        tmpy = [tf.convert_to_tensor(el) for el in tmpy]
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
