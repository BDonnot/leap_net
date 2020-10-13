# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import os
import json

import numpy as np
import tensorflow as tf

from collections.abc import Iterable
from grid2op.Agent import BaseAgent

from leap_net.proxy.ProxyLeapNet import ProxyLeapNet


class AgentWithProxy(BaseAgent):
    """
    Add to an agent a proxy leap net (usefull to train a leap net model)

    TODO add an example of usage


    """
    def __init__(self,
                 agent_action,  # the agent that will take some actions
                 logdir="tf_logs",
                 update_tensorboard=256,  # tensorboard is updated every XXX training iterations
                 save_freq=256,  # model is saved every save_freq training iterations
                 ext="h5",  # extension of the file in which you want to save the proxy

                 name="leap_net",
                 max_row_training_set=int(1e5),
                 batch_size=32,
                 attr_x=("prod_p", "prod_v", "load_p", "load_q"),
                 attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v"),
                 attr_tau=("line_status", ),
                 sizes_enc=(20, 20, 20),
                 sizes_main=(150, 150, 150),
                 sizes_out=(100, 40),
                 lr=1e-4,
                 ):
        BaseAgent.__init__(self, agent_action.action_space)
        self.agent_action = agent_action

        # to fill the training / test dataset
        self.max_row_training_set = max_row_training_set
        self.batch_size = batch_size
        self.global_iter = 0
        self.train_iter = 0
        self.__is_init = False  # is this model initiliazed

        # proxy part
        self._proxy = ProxyLeapNet(name=name,
                                   max_row_training_set=max_row_training_set, batch_size=batch_size,
                                   attr_x=attr_x, attr_y=attr_y, attr_tau=attr_tau,
                                   sizes_enc=sizes_enc, sizes_main=sizes_main, sizes_out=sizes_out,
                                   lr=lr)

        # tensorboard (should be initialized after the proxy)
        if logdir is not None:
            logpath = os.path.join(logdir, self.get_name())
            self._tf_writer = tf.summary.create_file_writer(logpath, name=self.get_name())
        else:
            logpath = None
            self._tf_writer = None
        self.update_tensorboard = update_tensorboard
        self.save_freq = save_freq

        # save load
        self.ext = ext
        self.save_path = None

    def init(self, obs):
        """

        Parameters
        ----------
        obs

        Returns
        -------

        """
        self.__is_init = True

        # now build the poxy
        self._proxy.init(obs)
        self._proxy.build_model()

    # agent interface
    def act(self, obs, reward, done=False):
        self.store_obs(obs)
        batch_losses = self._proxy.train(tf_writer=self._tf_writer)
        if batch_losses is not None:
            self.train_iter += 1
            self._save_tensorboard(batch_losses)
            self._save_model()
        return self.agent_action.act(obs, reward, done)

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

        self._proxy.store_obs(obs)

    def train(self, env, total_training_step, save_path=None, load_path=None):
        """
        Completely train the proxy

        Parameters
        ----------
        env
        total_training_step

        Returns
        -------

        """
        obs = env.reset()
        done = False
        reward = env.reward_range[0]
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
        if load_path is not None:
            self.load(load_path)
        with tqdm(total=total_training_step) as pbar:
            # update the progress bar
            pbar.update(self.global_iter)

            # and do the "gym loop"
            while not done:
                act = self.act(obs, reward, done)
                # TODO handle multienv here
                obs, reward, done, info = env.step(act)
                if done:
                    obs = env.reset()
                    done = False
                self.global_iter += 1
                if self.global_iter > total_training_step:
                    break
                pbar.update(1)
        # save the model at the end
        self.save(self.save_path)

    def save(self, path):
        """
        Part of the l2rpn_baselines interface, this allows to save a model. Its name is used at saving time. The
        same name must be reused when loading it back.

        Parameters
        ----------
        path: ``str``
            The path where to save the agent.

        """
        if path is not None:
            tmp_me = os.path.join(path, self.get_name())
            if not os.path.exists(tmp_me):
                os.mkdir(tmp_me)
            path_model = self._get_path_nn(path, self.get_name())
            self._save_metadata(path_model)
            self._proxy.save_weights('{}.{}'.format(os.path.join(path_model, self.get_name()), self.ext))

    def load(self, path):
        if path is not None:
            path_model = self._get_path_nn(path, self.get_name())
            self._load_metadata(path_model)
            self._proxy.build_model()
            self._proxy.load_weights('{}.{}'.format(os.path.join(path_model, self.get_name()), self.ext))

    def get_name(self):
        return self._proxy.name

    # save load model
    def _get_path_nn(self, path, name):
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        return path_model

    def _save_metadata(self, path_model):
        """save the dimensions of the models and the scalers"""
        json_nm = "metadata.json"
        me = self._to_dict()
        with open(os.path.join(path_model, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=me, fp=f)

    def _load_metadata(self, path_model):
        json_nm = "metadata.json"
        with open(os.path.join(path_model, json_nm), "r", encoding="utf-8") as f:
            me = json.load(f)
        self._from_dict(me)

    def _to_dict(self):
        res = {}
        res["proxy"] = self._proxy.get_metadata()
        res["train_iter"] = int(self.train_iter)
        res["global_iter"] = int(self.global_iter)
        return res

    def _save_dict(self, li, val):
        if isinstance(val, Iterable):
            li.append([float(el) for el in val])
        else:
            li.append(float(val))

    def _from_dict(self, dict_):
        """modify self! """
        self.train_iter = int(dict_["train_iter"])
        self.global_iter = int(dict_["global_iter"])
        self._proxy.load_metadata(dict_["proxy"])

    def _save_tensorboard(self, batch_losses):
        """save all the information needed in tensorboard."""
        if self._tf_writer is None:
            return

        # Log some useful metrics every even updates
        if self.train_iter % self.update_tensorboard == 0:
            with self._tf_writer.as_default():
                # save total loss
                tf.summary.scalar(f"global loss",
                                  np.sum(batch_losses),
                                  self.train_iter,
                                  description="loss for the entire model")
                self._proxy.save_tensorboard(self._tf_writer, self.train_iter, batch_losses)

    def _save_model(self):
        if self.train_iter % self.save_freq == 0:
            self.save(self.save_path)


if __name__ == "__main__":
    import grid2op
    from grid2op.Parameters import Parameters
    from leap_net.generate_data.Agents import RandomN1, RandomNN1
    from tqdm import tqdm
    from lightsim2grid.LightSimBackend import LightSimBackend

    total_train = 11*12*int(2e5)
    total_train = 4*int(1e5)
    env_name = "l2rpn_case14_sandbox"

    # generate the environment
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    env = grid2op.make(param=param, backend=LightSimBackend())
    agent = RandomNN1(env.action_space, p=0.5)
    agent_with_proxy = AgentWithProxy(agent, name="test_refacto1", max_row_training_set=int(total_train/10))

    agent_with_proxy.train(env,
                           total_train,
                           save_path="model_saved",
                           load_path="model_saved",
                           )
