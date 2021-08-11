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

from leap_net.proxy.baseNNProxy import BaseNNProxy
from leap_net.LtauNoAdd import LtauNoAdd


class ProxyLeapNet(BaseNNProxy):
    """
    This class demonstrate how to implement a proxy based on a neural network with the leap net architecture.

    This proxy is fully functional and some examples of training / evaluation can be found in the scripts
    `train_proxy_case_14.py`, `train_proxy_case_118.py`, `evaluate_proxy_case_14.py` and
    `evaluate_proxy_case_118.py`.

    It scales the data and has 3 different datasets:

    - `_my_x` : present in the base class :attr:`BaseNNProxy._my_x` representing the regular input to the neural
      network
    - `_my_y` : present in the base class :attr:`BaseNNProxy._my_y` representing what the neural network need
      to predict
    - `_my_tau`: representing the "tau" vectors.

    So this class also demonstrates how the generic interface can be adapted in case you want to deal with different
    data scheme (in this case 2 inputs and 1 outputs)

    """
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
                 layer_act=None,
                 topo_vect_to_tau="raw",  # see code for now  # TODO doc
                 kwargs_tau=None,  # optionnal kwargs depending on the method chosen for building tau from the observation
                 ):
        BaseNNProxy.__init__(self,
                             name=name,
                             lr=lr,
                             max_row_training_set=max_row_training_set,
                             train_batch_size=train_batch_size,
                             eval_batch_size=eval_batch_size,
                             attr_x=attr_x,
                             attr_y=attr_y,
                             layer=layer,
                             layer_act=layer_act)
        # datasets
        self._my_tau = None
        self._sz_tau = None

        # scalers
        self._m_x = None  # TODO move that into the baseNN class
        self._m_y = None  # TODO move that into the baseNN class
        self._m_tau = None
        self._sd_x = None  # TODO move that into the baseNN class
        self._sd_y = None  # TODO move that into the baseNN class
        self._sd_tau = None

        # specific part to leap net model
        # TODO to make sure it's integers
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out = sizes_out
        self.attr_tau = attr_tau
        self._scale_main_layer = scale_main_layer
        self._scale_input_dec_layer = scale_input_dec_layer
        self._scale_input_enc_layer = scale_input_enc_layer

        # not to load multiple times the meta data

        # small stuff with powerlines (force prediction of 0 when powerline is disconnected)
        # attributes that are stored as lines
        self._line_attr = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "v_or", "v_ex"}
        self.tensor_line_status = None
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

        # for handling the topo vect
        self.topo_vect_to_tau = topo_vect_to_tau
        self.nb_diff_topo_per_sub = None  # used for topo_vect_handler == "all"
        self.nb_diff_topo = None  # used for topo_vect_handler == "all" and "given_list"
        self.subs_index = None  # used for topo_vect_handler == "all" and "given_list"
        self.power_of_two = None  # used for topo_vect_handler == "all"
        self.kwargs_tau = kwargs_tau  # used for topo_vect_handler == "given_list" and "online_list"
        self._nb_max_topo = None  # used for topo_vect_handler "online_list"
        self._current_used_topo_max_id = None  # used for topo_vect_handler "online_list"
        self.dict_topo = None
        if topo_vect_to_tau == "raw":
            self.topo_vect_handler = self._raw_topo_vect
        elif topo_vect_to_tau == "all":
            self.topo_vect_handler = self._all_topo_encode
        elif topo_vect_to_tau == "given_list":
            self.topo_vect_handler = self._given_list_topo_encode
        elif topo_vect_to_tau == "online_list":
            self.topo_vect_handler = self._online_list_topo_encode
        else:
            raise RuntimeError(f"Unknown way to encode the topology vector in a tau vector (\"{topo_vect_to_tau}\")")

    def build_model(self):
        """build the neural network used as proxy, in this case a leap net."""
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

        This would not be necessary to overide it in "regular" model, but in this case we also need to store
        the "tau".

        The storing of X and Y is done automatically in the base class, hence the call of `super().store_obs(obs)`
        """
        # save the specific part to tau
        for attr_nm, inp in zip(self.attr_tau, self._my_tau):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)

        # save the observation in the database
        super().store_obs(obs)

    def _init_sub_index(self, obs):
        self.subs_index = []
        prev = 0
        for sub_id in range(obs.n_sub):
            nb_el = obs.sub_info[sub_id]
            self.subs_index.append((prev, prev + nb_el))
            prev += obs.sub_info[sub_id]

    def _process_topo_list(self, obs, topo_list):
        """
        one of the utilitary function for self.topo_vect_to_tau == "given_list"

        `topo_list` should have been given at initialization with `kwargs_tau`
        """
        if len(topo_list) == 0:
            raise RuntimeError("No topology provided, please check the \"topo_list\" that you provided in the "
                               "`kwarg_tau` argument")
        res = {}
        topo_id = 0
        for component, (sub_id, sub_topo) in enumerate(topo_list):
            if len(sub_topo) != obs.sub_info[sub_id]:
                raise RuntimeError(f"The provided topology for substation {sub_id} counts {len(sub_topo)} values "
                                   f"while there are {obs.sub_info[sub_id]} connected to it."
                                   f"Please check the \"topo_list\" that you provided in the "
                                   "`kwarg_tau` argument")
            topo_array = np.array(sub_topo)
            if np.all(topo_array == 2):
                raise RuntimeError(f"If using this topology encoding, you must not provide topology with \"all "
                                   f"\"connected to bus 2\" as it takes into account the symmetries. Check or remove "
                                   f"the input of substation {sub_id}."
                                   f"Please check the \"topo_list\" that you provided in the "
                                   "`kwarg_tau` argument")

            if np.any(topo_array <= 0):
                raise RuntimeError(f"Topologies should be only represented as either 1 or 2. We found 0 or negative "
                                   f"number for topology of susbtation {sub_id}"
                                   f"Please check the \"topo_list\" that you provided in the "
                                   "`kwarg_tau` argument"
                                   )
            if np.any(topo_array > 2):
                raise RuntimeError(f"Topologies should be only represented as either 1 or 2. We found "
                                   f"number > 3 in topology of susbtation {sub_id}"
                                   f"Please check the \"topo_list\" that you provided in the "
                                   "`kwarg_tau` argument"
                                   )
            topo_1 = (sub_id, tuple([int(el) for el in sub_topo]))
            topo_2 = (sub_id, tuple([2 if el == 1 else 1 for el in sub_topo]))
            if topo_1 in res:
                warnings.warn(f"Topology {sub_topo} of substation {sub_id} ({component}th element of the provided) "
                              f"vector is already encoded previously."
                              f"Please check the \"topo_list\" that you provided in the "
                              "`kwarg_tau` argument")
            else:
                res[topo_1] = topo_id
                res[topo_2] = topo_id
                topo_id += 1
        nb_diff_topo = topo_id
        return res, nb_diff_topo

    def init(self, obss):
        """
        Initialize all the meta data and the database for training

        Parameters
        ----------
        obs

        Returns
        -------

        """

        # init the handler for the topologie, if anything related to topology is done
        # NB this should be done before any call to "self._extract_obs"
        if self.topo_vect_to_tau == "all":
            obs = obss[0]
            self.nb_diff_topo_per_sub = (2 ** obs.sub_info) - 1
            self.nb_diff_topo = np.sum(self.nb_diff_topo_per_sub)
            self._init_sub_index(obs)
            self.power_of_two = 2 ** np.arange(np.max(obs.sub_info))
        elif self.topo_vect_to_tau == "given_list":
            obs = obss[0]
            self._init_sub_index(obs)
            self.dict_topo, self.nb_diff_topo = self._process_topo_list(obs, self.kwargs_tau)
        elif self.topo_vect_to_tau == "online_list":
            obs = obss[0]
            self._init_sub_index(obs)
            if self.kwargs_tau is None:
                raise RuntimeError("Impossible to use the topo_vect_to_tau=\"online_list\" tau encoding without "
                                   "providing a "
                                   "kwargs_tau that should be a > 0 integer.")
            try:
                nb_max_topo = int(self.kwargs_tau)
            except ValueError as exc_:
                raise RuntimeError(f"When using topo_vect_to_tau=\"online_list\" encoding"
                                   f"the \"kwargs_tau\" should be a > 0 integer we found {self.kwargs_tau}") from exc_
            if nb_max_topo <= 0:
                raise RuntimeError(f"When using topo_vect_to_tau=\"online_list\" encoding"
                                   f"the \"kwargs_tau\" should be a > 0 integer it is currently {nb_max_topo}")

            self._nb_max_topo = nb_max_topo
            self.dict_topo = {}
            self._current_used_topo_max_id = 0

        if not self._metadata_loaded:
            # ini the vector tau
            self._sz_tau = []
            for attr_nm in self.attr_tau:
                arr_ = self._extract_obs(obss[0], attr_nm)
                sz = arr_.size
                self._sz_tau.append(sz)

        # init the rest (attributes of the base class)
        super().init(obss)

        # deals with normalization #TODO some of it might be done in the base class
        # initialize mean and standard deviation
        # but only if the model is being built, not if it has been reloaded
        if not self._metadata_loaded:
            # for the input
            self._m_x = []
            self._sd_x = []
            for attr_nm in self.attr_x:
                self._m_x.append(self._get_mean(obss, attr_nm))
                self._sd_x.append(self._get_sd(obss, attr_nm))

            # for the output
            self._m_y = []
            self._sd_y = []
            for attr_nm in self.attr_y:
                self._m_y.append(self._get_mean(obss, attr_nm))
                self._sd_y.append(self._get_sd(obss, attr_nm))

            # for the tau vectors
            self._m_tau = []
            self._sd_tau = []
            for attr_nm in self.attr_tau:
                self._m_tau.append(self._get_mean(obss, attr_nm))
                self._sd_tau.append(self._get_sd(obss, attr_nm))

        self._metadata_loaded = True

    def _save_dict_topo(self, path):
        """utility functions to save self.dict_topo as json, because json default dump function
        does not like dictionnary keys that are tuple...

        what we would like to do but cannot:

        .. code-block:: python

            with open(os.path.join(path, "dict_topo.json"), "w", encoding="utf-8") as f:
                json.dump(obj=self.dict_topo, fp=f)
        """

        import os
        import json
        dict_serialized = {}
        for (sub_id, topo_descr), topo_id in self.dict_topo.items():
            rest_ = ','.join([str(el) for el in topo_descr])
            new_key = f"{sub_id}@{rest_}"
            dict_serialized[new_key] = topo_id

        with open(os.path.join(path, "dict_topo.json"), "w", encoding="utf-8") as f:
            json.dump(obj=dict_serialized, fp=f)

    def _load_dict_topo(self, path):
        """to load back the topo data...

        what we would like to do but cannot:

        .. code-block:: python

            with open(os.path.join(path, "dict_topo.json"), "r", encoding="utf-8") as f:
                self.dict_topo = json.load(fp=f)

        """
        import os
        import json

        with open(os.path.join(path, "dict_topo.json"), "r", encoding="utf-8") as f:
            dict_serialized = json.load(fp=f)

        self.dict_topo = {}
        for encoded_key, topo_id in dict_serialized.items():
            sub_id, rest_ = encoded_key.split("@")
            topo_descr = tuple([int(el) for el in rest_.split(",")])
            decoded_key = (int(sub_id), topo_descr)
            self.dict_topo[decoded_key] = topo_id

    def _save_subs_index(self, path):
        """utility to save self.subs_index because json does not like "int64"...
        Nothing to do at loading time as python is perfectly fine with regular int
        """
        import os
        import json
        with open(os.path.join(path, "subs_index.json"), "w", encoding="utf-8") as f:
            json.dump(obj=[(int(el), int(ell)) for el, ell in self.subs_index], fp=f)

    def save_data(self, path, ext=".h5"):
        import os
        if self.topo_vect_to_tau == "all":
            np.save(file=os.path.join(path, "nb_diff_topo_per_sub.npy"),
                    arr=self.nb_diff_topo_per_sub)
            np.save(file=os.path.join(path, "nb_diff_topo.npy"),
                    arr=self.nb_diff_topo)
            np.save(file=os.path.join(path, "power_of_two.npy"),
                    arr=self.power_of_two)
            self._save_subs_index(path)
        elif self.topo_vect_to_tau == "given_list":
            np.save(file=os.path.join(path, "nb_diff_topo.npy"),
                    arr=self.nb_diff_topo)
            self._save_dict_topo(path)
            self._save_subs_index(path)
        elif self.topo_vect_to_tau == "online_list":
            np.save(file=os.path.join(path, "_nb_max_topo.npy"),
                    arr=self._nb_max_topo)
            np.save(file=os.path.join(path, "_current_used_topo_max_id.npy"),
                    arr=self._current_used_topo_max_id)
            self._save_dict_topo(path)
            self._save_subs_index(path)

    def load_data(self, path, ext=".h5"):
        import os
        import json
        # TODO factorize the different stuff used for different encoding
        if self.topo_vect_to_tau == "all":
            self.nb_diff_topo_per_sub = np.load(file=os.path.join(path, "nb_diff_topo_per_sub.npy"))
            self.nb_diff_topo = np.load(file=os.path.join(path, "nb_diff_topo.npy"))
            self.power_of_two = np.load(file=os.path.join(path, "power_of_two.npy"))
            with open(os.path.join(path, "subs_index.json"), "r", encoding="utf-8") as f:
                self.subs_index = json.load(fp=f)
        elif self.topo_vect_to_tau == "given_list":
            with open(os.path.join(path, "subs_index.json"), "r", encoding="utf-8") as f:
                self.subs_index = json.load(fp=f)
            self._load_dict_topo(path)
            self.nb_diff_topo = np.load(file=os.path.join(path, "nb_diff_topo.npy"))
        elif self.topo_vect_to_tau == "online_list":
            with open(os.path.join(path, "subs_index.json"), "r", encoding="utf-8") as f:
                self.subs_index = json.load(fp=f)
            self._load_dict_topo(path)
            self._nb_max_topo = int(np.load(file=os.path.join(path, "_nb_max_topo.npy")))
            self._current_used_topo_max_id = int(np.load(file=os.path.join(path, "_current_used_topo_max_id.npy")))

    def get_metadata(self):
        res = super().get_metadata()
        # save attribute for the "extra" database
        res["attr_tau"] = [str(el) for el in self.attr_tau]
        res["_sz_tau"] = [int(el) for el in self._sz_tau]

        # save means and standard deviation
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

        # store the sizes
        res["sizes_enc"] = [int(el) for el in self.sizes_enc]
        res["sizes_main"] = [int(el) for el in self.sizes_main]
        res["sizes_out"] = [int(el) for el in self.sizes_out]

        # store some information about some transformations we can do
        if self._scale_main_layer is not None:
            res["_scale_main_layer"] = int(self._scale_main_layer)
        else:
            # i don't store anything if it's None
            pass
        if self._scale_input_dec_layer is not None:
            res["_scale_input_dec_layer"] = int(self._scale_input_dec_layer)
        else:
            # i don't store anything if it's None
            pass
        if self._scale_input_enc_layer is not None:
            res["_scale_input_enc_layer"] = int(self._scale_input_enc_layer)
        else:
            # i don't store anything if it's None
            pass
        return res

    def _init_database_shapes(self):
        """
        Again this method is only overriden because the leap net takes inputs in two different ways: the X's
        and the tau's
        """
        super()._init_database_shapes()
        self._my_tau = []
        for sz in self._sz_tau:
            self._my_tau.append(np.zeros((self.max_row_training_set, sz), dtype=self.dtype))

    def load_metadata(self, dict_):
        """
        load the metadata of this neural network (also called meta parameters) from a dictionary
        """
        self.attr_tau = tuple([str(el) for el in dict_["attr_tau"]])
        self._sz_tau = [int(el) for el in dict_["_sz_tau"]]
        super().load_metadata(dict_)

        for key in ["_m_x", "_m_y", "_m_tau", "_sd_x", "_sd_y", "_sd_tau"]:
            setattr(self, key, [])
            for el in dict_[key]:
                self._add_attr(key, el)

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

    def _extract_data(self, indx_train):
        """
        extract from the training dataset, the data with indexes `indx_train`

        The model will be trained with a code equivalent to:

        .. code-block:: python

            data = self._extract_data(indx_train)
            batch_losses = self._train_model(data)

        This function is also used for the evaluation of the model in the following manner:

        .. code-block:: python

            data = self._extract_data(indx_val)
            res = self._make_predictions(data, training=False)

        Here we needed to override it for two reasons:

        - we use 3 different data (X,tau, Y) this is specific to leap net
        - we wanted to scale the data passed to the neural networks

        Parameters
        ----------
        indx_train: ``numpy.ndarray``, ``int``
            The index of the data that needs to be retrieved from the database `_my_x` and `_my_y`

        Returns
        -------
        X:
            The value of the input data
        Y:
            The value of the desired output of the proxy

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

        In our case we needed to code it because we applied some scaling when the data were "extracted" from the
        internal database (we overide :func:`ProxyLeapNet._extract_data`)
        """
        tmp = [el.numpy() for el in predicted_state]
        resy = [arr * sd_ + m_ for arr, m_, sd_ in zip(tmp, self._m_y, self._sd_y)]
        return resy

    # customization of the "topo_vect" retrieval
    def _extract_obs(self, obs, attr_nm):
        if attr_nm == "topo_vect":
            res = self.topo_vect_handler(obs)
        else:
            res = super()._extract_obs(obs, attr_nm)
        return res

    def _raw_topo_vect(self, obs):
        """
        This functions encode the topology vector in:

        - 0 if the element is on bus 1 or disconnected
        - 1 if the element is on bus 2

        The resulting vector has the dimension "dim_topo"

        Even if there are only one topological change at a substation, encoded this way the tau vector can have
        multiple components set to 1.

        Parameters
        ----------
        obs

        Returns
        -------

        Examples
        --------

        Once initialized, you have the following steps:


        ..code-block:: python

            import grid2op
            import numpy as np
            from leap_net.proxy import ProxyLeapNet

            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="raw")
            obs = env.reset()
            proxy.init([obs])

            # valid only for this environment
            env = grid2op.make("l2rpn_case14_sandbox")
            # the number of elements per substations are:
            # [3, 6, 4, 6, 5, 7, 3, 2, 5, 3, 3, 3, 4, 3]
            obs = env.reset()
            proxy.init([obs])

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[0] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[1] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[3] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 4
            assert np.all(res[[50, 51, 52, 53]] == 1.)

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 3
            assert np.all(res[[54, 55, 56]] == 1.)

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 4
            assert res[0] == 1.
            assert np.all(res[[54, 55, 56]] == 1.)

        """
        res = np.zeros(obs.dim_topo)
        res[obs.topo_vect == 2] = 1
        return res

    def _all_topo_encode(self, obs):
        """
        This function encodes the topology vector "topo_vect" as followed:
            - for all substation, it creates a one-hot encoded vector of size "the number of possible topology" for
              this substation
            - then it concatenates everything

        The resulting tau vector counts then as many component as there are unary topological changes.

        It will be all 0 if all elements of the grid are connected to bus 1)

        It counts as many "1" as the number of substation not in their reference topology

        It cannot have more than "env.n_sub" one at a time.

        Parameters
        ----------
        obs

        Returns
        -------
        the topology vector

        Examples
        --------

        Once initialized, you have the following steps:


        ..code-block:: python

            import grid2op
            import numpy as np
            from leap_net.proxy import ProxyLeapNet

            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="all")
            obs = env.reset()
            proxy.init([obs])

            # valid only for this environment
            env = grid2op.make("l2rpn_case14_sandbox")
            # the number of elements per substations are:
            [3, 6, 4, 6, 5, 7, 3, 2, 5, 3, 3, 3, 4, 3]
            obs = env.reset()
            proxy.init([obs])

            # for the complete topology
            obs = env.reset()
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 0

            # the "first" topology: change the first element of the substation 0
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 1
            assert tau[0] == 1.

            # second topology: change the second element of substation 0.
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 1
            assert tau[1] == 1.

            # the last topology identified: everything on bus 2 for substation 13 (last one)
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 1
            assert tau[389] == 1.

            # as there are 3 elements on the substation 0, there are 7 (=2^3 - 1) different possible topologies
            # for this one
            # And this will be the "first topology of substation 1"
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 1
            assert tau[7] == 1.

            # change the substation 12
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 1
            assert tau[382] == 1.

            # if i change 2 substation, i have 2 "1" on the tau vector
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            tau = proxy.topo_vect_handler(obs)
            assert np.sum(tau) == 2
            assert tau[389] == 1.
            assert tau[0] == 1.

        """
        res = np.zeros(self.nb_diff_topo)
        # retrieve the topology
        topo_vect = 1 * obs.topo_vect
        # ignore (for now) the disconnected elements
        topo_vect[topo_vect <= 0] = 1
        # convert to 0 => on bus 1, 1 => on bus 2
        topo_vect -= 1
        # and now put the right number
        prev = 0
        for sub_id in range(obs.n_sub):
            # TODO there might be a way to optimize that, but maybe I need to check the time it takes first.
            beg_, end_ = self.subs_index[sub_id]
            nb_el = obs.sub_info[sub_id]
            index_one_this = np.sum(topo_vect[beg_:end_] * self.power_of_two[:nb_el]) - 1
            if index_one_this >= 0:
                res[prev + index_one_this] = 1
            prev += 2 ** nb_el - 1
        return res

    def _given_list_topo_encode(self, obs):
        """
        This methods require a pre selected set of substation topology that you can have (that should give at the
        initialization in the keyword argument `kwarg_tau`).

        It will then assign a component of a tau vector for each "substation topology".

        It acts basically as a subset of the :func:`ProxyLeapNet._all_topo_encode` but considering not all
        topologies but only a subset of the possible ones.

        Topologies (remember, given in `kwarg_tau`) are represented by a tuple with 2 elements:
        - substation id
        - topology of this substation given by a vector counting as many components of the elements connected to this
          substation

        The complete list of topologies should be given as list / numpy array or any other iterators over the
        topology tuples.

        The "all connected to bus 1" topology will be encoded by 0, as always.

        The resulting "tau" vector will count as many component as the length of different unary topologies.

        **NB** in the description of the topologies, we expect vectors with only 1 and 2 (no 0, no -1 etc.)

        **NB** if an object is disconnected, this method will behave as if it is connected to bus 1.

        **NB** if the topology seen in the observation is not found in the list of possible unary change,
        it raises a warning and returns the "tau_ref" vector. It will NOT raise an error in this case

        **NB** as opposed to other methods (:func:`ProxyLeapNet._all_topo_encode` or
        :func:`ProxyLeapNet._raw_topo_vect`), it also search if the inverse of the topology
        (so swapping buses 1<->2 is in the provided list -- the constraint that disconnected object are
        on bus 1 still applies! )

        Parameters
        ----------
        obs

        Returns
        -------
        the topology vector

        Examples
        --------

        Once initialized, you have the following steps:

        ..code-block:: python

            import grid2op
            import numpy as np
            from leap_net.proxy import ProxyLeapNet

            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="given_list",
                                 kwargs_tau=[(0, (2, 1, 1)), (0, (1, 2, 1)), (1, (2, 1, 1, 1, 1, 1)),
                                             (12, (2, 1, 1, 2)), (13, (2, 1, 2)), (13, (1, 2, 2))]
                                 )
            obs = env.reset()
            proxy.init([obs])

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[0] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[1] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[2] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 0

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 0

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 1, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 2
            assert res[0] == 1.
            assert res[4] == 1.

        """
        res = np.zeros(self.nb_diff_topo)
        # retrieve the topology
        topo_vect = 1 * obs.topo_vect

        # and now put the right number
        for sub_id in range(obs.n_sub):
            # TODO there might be a way to optimize that, but what for ?
            beg_, end_ = self.subs_index[sub_id]
            this_sub_topo = topo_vect[beg_:end_]
            disco = this_sub_topo == -1
            conn = ~disco
            if np.all(this_sub_topo[conn] == 2) or np.all(this_sub_topo[conn] == 1):
                # complete / reference topology, so i don't do anything
                continue

            # so i have a different topology that the reference one
            lookup = (sub_id, tuple([el if el >= 1 else 1 for el in this_sub_topo]))
            if lookup in self.dict_topo:
                res[self.dict_topo[lookup]] = 1.
            else:
                warnings.warn(f"Topology {lookup} is not found on the topo dictionary")
        return res

    def _online_list_topo_encode(self, obs):
        """
        This method behaves exaclyt like :func:`ProxyLeapNet._given_list_topo_encode` with one difference: you do not
        need to provide any list of topologies at the initialization. But rather when a new topology for a substation
        will be encounter it will be added to a new component of tau.

        It can store up to `kwargs_tau` different topologies. Afterwards, each new topologies will be assigned to
        the reference topologies (everything connected to bus 1)

        It assigns a component of a tau vector for each known "substation topology".

        It acts basically as a subset of the :func:`ProxyLeapNet._all_topo_encode` but considering not all
        topologies but only a subset of the possible ones encountered up until a certain point.

        The "all connected to bus 1" topology will be encoded by 0, as always.

        The resulting "tau" vector will count as many component as `kwargs_tau` (which should be an integer) as
        argument.

        **NB** if an object is disconnected, this method will behave as if it is connected to bus 1.

        **NB** as opposed to other methods (:func:`ProxyLeapNet._all_topo_encode` or
        :func:`ProxyLeapNet._raw_topo_vect`) but like its "big sister" :func:`ProxyLeapNet._given_list_topo_encode`
        it also search if the inverse of the topology

        Parameters
        ----------
        obs

        Returns
        -------
        the topology vector

        Examples
        --------

        Once initialized, you have the following steps:


        ..code-block:: python

            import grid2op
            import numpy as np
            from leap_net.proxy import ProxyLeapNet

            proxy = ProxyLeapNet(attr_tau=("line_status", "topo_vect",),
                                 topo_vect_to_tau="online_list",
                                 kwargs_tau=7  # as an example, any > 0 integer works
                                 )
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[0] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (1, 2, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[1] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[2] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(12, (2, 2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 0

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 2, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 0

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(0, (2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            act = env.action_space({"set_bus": {"substations_id": [(13, (2, 1, 2))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 2
            assert res[0] == 1.
            assert res[3] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[4] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 2, 1, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[5] == 1.

            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 2, 1, 2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 1
            assert res[6] == 1.

            # "overflow" in the number of topologies => like ref topology
            obs = env.reset()
            act = env.action_space({"set_bus": {"substations_id": [(1, (2, 1, 1, 2, 1, 1))]}})
            obs, reward, done, info = env.step(act)
            res = proxy.topo_vect_handler(obs)
            assert np.sum(res) == 0

        """
        res = np.zeros(self._nb_max_topo)
        # retrieve the topology
        topo_vect = 1 * obs.topo_vect

        # and now put the right number
        prev = 0
        for sub_id in range(obs.n_sub):
            # TODO there might be a way to optimize that, but what for ?
            beg_, end_ = self.subs_index[sub_id]
            nb_el = obs.sub_info[sub_id]
            this_sub_topo = topo_vect[beg_:end_]
            disco = this_sub_topo == -1
            conn = ~disco
            if np.all(this_sub_topo[conn] == 2) or np.all(this_sub_topo[conn] == 1):
                # complete / reference topology, so i don't do anything
                continue
            # so i have a different topology that the reference one
            lookup = (sub_id, tuple([el if el >= 1 else 1 for el in this_sub_topo]))
            if lookup not in self.dict_topo:
                if self._current_used_topo_max_id >= self._nb_max_topo:
                    # we can't add another topology...
                    warnings.warn(f"Already too much topologies encoded. Please consider increasing \"kwargs_tau\""
                                  f"which is currently {self._nb_max_topo} and try again. You can save the values "
                                  f"of the relevant layers in numpy arrays, create another proxy with larger tau, "
                                  f"encode all the taus in the same order as this one, and assign the value of the "
                                  f"\"new\" layer to be the old values and try again to learn. This is for now painful"
                                  f"but looks doable.")
                    sub_topo_id_ = None
                else:
                    topo_1 = lookup
                    topo_2 = (sub_id, tuple([2 if el == 1 else 1 for el in this_sub_topo]))
                    self.dict_topo[topo_1] = self._current_used_topo_max_id
                    self.dict_topo[topo_2] = self._current_used_topo_max_id
                    sub_topo_id_ = self._current_used_topo_max_id
                    self._current_used_topo_max_id += 1
            else:
                sub_topo_id_ = self.dict_topo[lookup]
            if sub_topo_id_ is not None:
                res[sub_topo_id_] = 1.
        return res
