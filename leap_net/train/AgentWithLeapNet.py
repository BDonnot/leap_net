# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import os

import numpy as np
import warnings
from grid2op.Agent import BaseAgent

import tensorflow as tf
import tensorflow.keras.optimizers as tfko
from tensorflow.keras.layers import Dense

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Input, Lambda, subtract, add
    import tensorflow.keras.backend as K

from leap_net.LtauNoAdd import LtauNoAdd


class AgentWithProxy(BaseAgent):
    """
    Add to an agent a proxy leap net (usefull to train a leap net model)

    TODO add an example of usage


    """
    def __init__(self,
                 agent_action,  # the agent that will take some actions
                 max_row_training_set=int(1e5),
                 batch_size=32,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v"),
                 attr_tau=("line_status", ),
                 sizes_enc=(20, 20, 20),
                 sizes_main=(150, 150, 150),
                 sizes_out=(100, 40),
                 lr=1e-5,
                 name="leap_net",
                 logdir="tf_logs",
                 update_tensorboard=256
                 ):
        BaseAgent.__init__(self, agent_action.action_space)
        self.agent_action = agent_action

        # to fill the dataset
        self.max_row_training_set = max_row_training_set
        self.batch_size = batch_size
        self.last_id = 0
        self.train_iter = 0
        self.attr_x = attr_x
        self.attr_y = attr_y
        self.attr_tau = attr_tau
        self.__db_full = False
        self.__is_init = False

        # sizes
        self._sz_x = None
        self._sz_y = None
        self._sz_tau = None

        # "my" training set
        self._my_x = None
        self._my_y = None
        self._my_tau = None

        # scaler
        self._m_x = None
        self._m_y = None
        self._m_tau = None
        self._sd_x = None
        self._sd_y = None
        self._sd_tau = None

         # leap net model
        self._model = None
        self.sizes_enc = sizes_enc
        self.sizes_main = sizes_main
        self.sizes_out = sizes_out
        self.dtype = np.float32

        # model optimizer
        self._lr = lr
        self._schedule_lr_model = None
        self._optimizer_model = None

        # tensorboard
        self.name = name
        if logdir is not None:
            logpath = os.path.join(logdir, self.name)
            self._tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        else:
            logpath = None
            self._tf_writer = None
        self.update_tensorboard = update_tensorboard

    def init(self, obs):
        """

        Parameters
        ----------
        obs

        Returns
        -------

        """
        self.__db_full = False
        self.__is_init = True
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
            self._m_x.append(self._get_mean(arr_, attr_nm))
            self._sd_x.append(self._get_sd(arr_, attr_nm))

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
            self._m_y.append(self._get_mean(arr_, attr_nm))
            self._sd_y.append(self._get_sd(arr_, attr_nm))

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
            self._m_tau.append(self._get_mean(arr_, attr_nm))
            self._sd_tau.append(self._get_sd(arr_, attr_nm))

        # now build the tensorflow model
        self.build_model()

    def make_optimiser(self):
        """
        helper function to create the proper optimizer (Adam) with the learning rates and its decay
        parameters.
        """
        # schedule = tfko.schedules.InverseTimeDecay(self._lr, self._lr_decay_steps, self._lr_decay_rate)
        return None, tfko.Adam(learning_rate=self._lr)

    def build_model(self):
        self._model = Sequential()
        inputs_x = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                    zip(self._sz_x, self.attr_x)]
        inputs_tau = [Input(shape=(el,), name="x_{}".format(nm_)) for el, nm_ in
                      zip(self._sz_tau, self.attr_tau)]

        # encode each data type in initial layers
        encs_out = []
        for init_val, nm_ in zip(inputs_x, self.attr_x):
            lay = init_val
            for i, size in enumerate(self.sizes_enc):
                lay = Dense(size, name="enc_{}_{}".format(nm_, i))(lay)  # TODO resnet instead of Dense
                lay = Activation("relu")(lay)
            encs_out.append(lay)

        # concatenate all that
        lay = tf.keras.layers.concatenate(encs_out)

        # i do a few layer
        for i, size in enumerate(self.sizes_main):
            lay = Dense(size, name="main_{}".format(i))(lay)  # TODO resnet instead of Dense
            lay = Activation("relu")(lay)

        # now i do the leap net to encode the state
        encoded_state = lay
        for input_tau, nm_ in zip(inputs_tau, self.attr_tau):
            tmp = LtauNoAdd(name=f"leap_{nm_}")([lay, input_tau])
            encoded_state = tf.keras.layers.add([encoded_state, tmp], name=f"adding_{nm_}")

        # i predict the full state of the grid given the "control" variables
        outputs_gm = []
        model_losses = {}
        lossWeights = {}  # TODO
        for sz_out, nm_ in zip(self._sz_y,
                               self.attr_y):
            lay = encoded_state  # carefull i need my gradients here ! (don't use self.encoded_state)
            for i, size in enumerate(self.sizes_out):
                lay = Dense(size, name="{}_{}".format(nm_, i))(lay)
                lay = Activation("relu")(lay)

            # predict now the variable
            name_output = "{}_hat".format(nm_)
            pred_ = Dense(sz_out, name=name_output)(lay)
            outputs_gm.append(pred_)
            model_losses[name_output] = "mse"

        # now create the model in keras
        self._model = Model(inputs=(inputs_x, inputs_tau),
                            outputs=outputs_gm,
                            name="model")
        # and "compile" it
        self._schedule_lr_model, self._optimizer_model = self.make_optimiser()
        self._model.compile(loss=model_losses, optimizer=self._optimizer_model)

    def _get_mean(self, arr_, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data
        """
        return np.zeros(arr_.size, dtype=self.dtype)

    def _get_sd(self, arr_, attr_nm):
        """
        For the scaler, compute the mean that will be used to scale the data
        """
        return np.ones(arr_.size, dtype=self.dtype)

    def _extract_obs(self, obs, attr_nm):
        """

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

    def store_obs(self, obs):
        """
        store the observation into the "database" for training the model.

        Notes
        -------
        Will also increment `self.last_id`

        Parameters
        ----------
        obs: `grid2op.Action.BaseObservation`
            The current observation
        """
        if not self.__is_init:
            self.init(obs)

        for attr_nm, inp in zip(self.attr_x, self._my_x):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_tau, self._my_tau):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)
        for attr_nm, inp in zip(self.attr_y, self._my_y):
            inp[self.last_id, :] = self._extract_obs(obs, attr_nm)

        self.last_id += 1
        if self.last_id >= self.max_row_training_set -1:
            self.__db_full = True
        self.last_id %= self.max_row_training_set

    def train(self):
        """
        train the leap net model

        returns the loss
        """
        if self.__db_full:
            tmp_max = self.max_row_training_set
        else:
            tmp_max = self.last_id
        indx_train = np.random.choice(np.arange(tmp_max),
                                      size=self.batch_size,
                                      replace=False)

        tmpx = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_x, self._m_x, self._sd_x)]
        tmpy = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_y, self._m_y, self._sd_y)]
        tmpt = [(arr[indx_train, :] - m_) / sd_ for arr, m_, sd_ in zip(self._my_tau, self._m_tau, self._sd_tau)]
        batch_losses = self._model.train_on_batch((tmpx, tmpt), tmpy)
        return batch_losses

    def act(self, obs, reward, done=False):
        self.store_obs(obs)
        if self.last_id % self.batch_size == 0:
            batch_losses = self.train()
            self._save_tensorboard(batch_losses)
        return self.agent_action.act(obs, reward, done)

    def _save_tensorboard(self, batch_losses):
        """save all the information needed in tensorboard."""
        if self._tf_writer is None:
            return

        # Log some useful metrics every even updates
        if self.train_iter % self.update_tensorboard == 0:
            with self._tf_writer.as_default():
                # save the losses
                for output_nm, loss in zip(self.attr_y, batch_losses):
                    tf.summary.scalar(f"{output_nm}", loss, self.train_iter,
                                      description=f"MSE for {output_nm}")
                # save total loss
                tf.summary.scalar(f"global loss",
                                  np.sum(batch_losses),
                                  self.train_iter,
                                  description="loss for the entire model")

                # TODO add the "evaluate on validation episode"


if __name__ == "__main__":
    import grid2op
    from grid2op.Parameters import Parameters
    from leap_net.generate_data.Agents import RandomN1
    from tqdm import tqdm
    from lightsim2grid.LightSimBackend import LightSimBackend
    total_train = int(2e5)

    # generate the environment
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    env = grid2op.make(param=param, backend=LightSimBackend())

    obs = env.reset()
    agent = RandomN1(env.action_space)
    agent_with_proxy = AgentWithProxy(agent)

    done = False
    reward = env.reward_range[0]
    nb_ts = 0
    with tqdm(total=total_train) as pbar:
        while not done:
            act = agent_with_proxy.act(obs, reward, done)
            obs, reward, done, info = env.step(act)
            if done:
                obs = env.reset()
                done = False
            pbar.update(1)
            nb_ts += 1
            if nb_ts > total_train:
                break

