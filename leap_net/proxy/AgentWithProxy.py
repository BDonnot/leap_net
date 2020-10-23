# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import os
import json
import re

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
                 proxy,  # the proxy to train / evaluate
                 logdir=None,  # tensorboard logs
                 update_tensorboard=256,  # tensorboard is updated every XXX training iterations
                 save_freq=int(1024)*int(64),  # model is saved every save_freq training iterations
                 ext=".h5",  # extension of the file in which you want to save the proxy
                 nb_obs_init=256,  # number of observations that are sent to the proxy to be initialized
                 ):
        BaseAgent.__init__(self, agent_action.action_space)
        self.agent_action = agent_action

        # to fill the training / test dataset
        self.global_iter = 0
        self.train_iter = 0
        self.__is_init = False  # is this model initiliazed
        self.is_training = True
        self._nb_obs_init = nb_obs_init

        # proxy part
        self._proxy = proxy

        # tensorboard (should be initialized after the proxy)
        if logdir is not None:
            logpath = os.path.join(logdir, self.get_name())
            self._tf_writer = tf.summary.create_file_writer(logpath, name=self.get_name())
        else:
            self._tf_writer = None
        self.update_tensorboard = update_tensorboard
        self.save_freq = save_freq

        # save load
        if re.match(r"^\.", ext) is None:
            # add a point at the beginning of the extension
            self.ext = f".{ext}"
        else:
            self.ext = ext
        self.save_path = None

    def init(self, env):
        """

        Parameters
        ----------
        env

        Returns
        -------

        """
        # generate a few observation to init the proxy
        obss = []
        nb_obs = 0
        while nb_obs <= self._nb_obs_init:
            done = False
            reward = env.reward_range[0]
            obs = env.reset()
            while not done:
                act = self.agent_action.act(obs, reward, done)
                obs, reward, done, info = env.step(act)
                if not done:
                    nb_obs += 1
                    obss.append(obs)

        # now build the poxy
        self._proxy.init(obss)
        # build the model
        self._proxy.build_model()
        self.__is_init = True

    # agent interface
    def act(self, obs, reward, done=False):
        self.global_iter += 1
        self.store_obs(obs)
        if self.is_training:
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
        done = False
        reward = env.reward_range[0]
        self.save_path = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
        if load_path is not None:
            self.load(load_path)
        self.is_training = True
        if not self.__is_init:
            self.init(env)
        with tqdm(total=total_training_step) as pbar:
            # update the progress bar
            pbar.update(self.global_iter)
            obs = self._reboot(env)
            # and do the "gym loop"
            while not done:
                act = self.act(obs, reward, done)
                # TODO handle multienv here
                if not self._proxy.DEBUG or self.global_iter <= self._proxy.train_batch_size:
                    obs, reward, done, info = env.step(act)
                    if done:
                        obs = self._reboot(env)
                        done = False
                # TODO have  multiprocess here: this script stores the data in the database of the proxy
                # TODO the proxy reads from this database (exposed as a tensorflow dataset) to train the model
                pbar.update(1)
                if self.global_iter >= total_training_step:
                    break

        # save the model at the end
        self.save(self.save_path)

    def evaluate(self, env, total_evaluation_step, load_path, save_path=None, metrics=None, verbose=0):
        """

        Parameters
        ----------
        env
        total_evaluation_step
        load_path
        save_path
        metrics:
            dictionary of function, with keys being the metrics name, and values the function that compute
            this metric (on the whole output) that should be `metric_fun(y_true, y_pred)`
        verbose

        Returns
        -------

        """
        done = False
        reward = env.reward_range[0]
        self.is_training = False

        if load_path is not None:
            self.load(load_path)
            self.global_iter = 0
            self.save_path = None  # disable the saving of the model

        # TODO find a better approach for more general proxy that can adapt to grid of different size
        sizes = self._proxy.get_output_sizes()
        true_val = [np.zeros((total_evaluation_step, el), dtype=self._proxy.dtype) for el in sizes]
        pred_val = [np.zeros((total_evaluation_step, el), dtype=self._proxy.dtype) for el in sizes]

        if not self.__is_init:
            self.init(env)
        with tqdm(total=total_evaluation_step) as pbar:
            # update the progress bar
            pbar.update(self.global_iter)
            obs = self._reboot(env)
            # and do the "gym loop"
            while not done:
                # act (which increment global_iter and ask the actor what action to do)
                act = self.act(obs, reward, done)

                # save the predictions and the reference
                predictions = self._proxy.predict(force=self.global_iter == total_evaluation_step)
                if predictions is not None:
                    for arr_, pred_ in zip(pred_val, predictions):
                        sz = pred_.shape[0]
                        min_ = max((self.global_iter-sz), 0)
                        arr_[min_:self.global_iter, :] = pred_

                reality = self._proxy.get_true_output(obs)
                for arr_, ref_ in zip(true_val, reality):
                    arr_[self.global_iter-1, :] = ref_.reshape(-1)

                # TODO handle multienv here (this might be more complicated!)
                obs, reward, done, info = env.step(act)
                if done:
                    obs = self._reboot(env)
                    done = False
                pbar.update(1)
                if self.global_iter >= total_evaluation_step:
                    break
        # save the results and compute the metrics
        # TODO save the real x's too!
        return self._save_results(obs, save_path, metrics, pred_val, true_val, verbose)

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
            path_save = os.path.join(path, self.get_name())
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            self._save_metadata(path_save)
            self._proxy.save_weights(path=path_save, ext=self.ext)

    def load(self, path):
        if path is not None:
            # the following if is to be able to restore a file with possibly a different name...
            if self.is_training:
                path_model = self._get_path_nn(path, self.get_name())
            else:
                path_model = path
            if not os.path.exists(path_model):
                raise RuntimeError(f"You asked to load a model at \"{path_model}\" but there is nothing there.")
            self._load_metadata(path_model)
            self._proxy.build_model()
            self._proxy.load_weights(path=path, ext=self.ext)

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
                tf.summary.scalar(f"0_global_loss",
                                  batch_losses[0],
                                  self.train_iter,
                                  description="Loss of the entire model")
                self._proxy.save_tensorboard(self._tf_writer, self.train_iter, batch_losses[1:])

    def _save_model(self):
        if self.train_iter % self.save_freq == 0:
            self.save(self.save_path)

    def _save_results(self, obs, save_path, metrics, pred_val, true_val, verbose):

        # compute the metrics (if any)
        dict_metrics = {}
        dict_metrics["predict_step"] = int(self.global_iter)
        dict_metrics["predict_time"] = float(self._proxy.get_total_predict_time())
        dict_metrics["avg_pred_time_s"] = float(self._proxy.get_total_predict_time()) / float(self.global_iter)
        if metrics is not None:
            array_names = self._proxy.get_attr_output_name(obs)
            for metric_name, metric_fun in metrics.items():
                dict_metrics[metric_name] = {}
                for nm, pred_, true_ in zip(array_names, pred_val, true_val):
                    tmp = metric_fun(true_, pred_)
                    # print the results and make sure the things are json serializable
                    if isinstance(tmp, Iterable):
                        if verbose >= 2:
                            print(f"{metric_name} for {nm}: {tmp}")  # don't print not to overload the display
                        dict_metrics[metric_name][nm] = [float(el) for el in tmp]
                    else:
                        if verbose >=1:
                            print(f"{metric_name} for {nm}: {tmp:.2f}")
                        dict_metrics[metric_name][nm] = float(tmp)
                # TODO if installed, use the grid2op plotMatplot to project this data on the grid and get a pdf of
                # TODO where the errors are located

        # save the numpy arrays (if needed)
        if save_path is not None:
            # save the proxy and the meta data
            self.save(save_path)

            # now the other data
            array_names = self._proxy.get_attr_output_name(obs)
            save_path = os.path.join(save_path, self.get_name())
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            for nm, pred_, true_ in zip(array_names, pred_val, true_val):
                np.save(os.path.join(save_path, f"{nm}_pred.npy"), pred_)
                np.save(os.path.join(save_path, f"{nm}_real.npy"), true_)
            with open(os.path.join(save_path, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dict_metrics, fp=f, indent=4, sort_keys=True)
        return dict_metrics

    def _reboot(self, env):
        """when an environment is "done" this function reset it and act a first time with the agent_action"""
        # TODO skip and random start at some steps
        done = False
        reward = env.reward_range[0]
        obs = env.reset()
        obs, reward, done, info = env.step(self.agent_action.act(obs, reward, done))
        while done:
            # we restart until we find an environment that is not "game over"
            obs = env.reset()
            obs, reward, done, info = env.step(self.agent_action.act(obs, reward, done))
        return obs


def reproducible_exp(env, agent, env_seed=None, chron_id=None, agent_seed=None):
    if env_seed is not None:
        env.seed(env_seed)

    if chron_id is not None:
        # reset the environment
        env.chronics_handler.tell_id(chron_id-1)

    if agent_seed is not None:
        agent.seed(agent_seed)


if __name__ == "__main__":
    import grid2op
    from grid2op.Parameters import Parameters
    from leap_net.agents import RandomN1, RandomNN1, RandomN2
    from grid2op.Agent import DoNothingAgent
    from tqdm import tqdm
    from lightsim2grid.LightSimBackend import LightSimBackend
    from sklearn.metrics import mean_squared_error, mean_absolute_error  # mean_absolute_percentage_error
    from grid2op.Chronics import MultifolderWithCache
    from leap_net.proxy.ProxyBackend import ProxyBackend
    from leap_net.proxy.NRMSE import nrmse
    from leap_net.ResNetLayer import ResNetLayer

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    total_train = int(1024)*int(128)  # ~4 minutes  # for case 14
    total_train = int(1024)*int(1024)  # ~30 minutes [32-35 mins] # for case 14 50 mins for case 118
    total_train = int(1024)*int(1024) * int(16)  # ~10h for 14
    total_train = int(1024)*int(1024) * int(8)  # ~7h for 118
    total_evaluation_step = int(1024) * int(32)
    # env_name = "l2rpn_case14_sandbox"
    # model_name = "Anne_Onymous"
    # model_name = "realtest_13"
    env_name = "l2rpn_neurips_2020_track2_small"
    # env_name = "l2rpn_case14_sandbox"
    model_name = "118_07"
    model_name = "test_118_09"
    save_path = "model_saved"
    save_path_final_results = "model_results"
    save_path_tensorbaord = "tf_logs"
    chron_id_val = 100
    env_seed = 42
    agent_seed = 1
    layerfun = ResNetLayer

    do_train = True
    do_dc = False
    do_N1 = True
    do_N2 = True
    do_hades = True  # TODO not used yet

    # generate the environment
    param = Parameters()
    param.NO_OVERFLOW_DISCONNECTION = True
    param.NB_TIMESTEP_COOLDOWN_LINE = 0
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    param.MAX_SUB_CHANGED = 99999
    param.MAX_LINE_STATUS_CHANGED = 99999
    param.NB_TIMESTEP_COOLDOWN_SUB = 0
    if env_name == "l2rpn_case14_sandbox":
        env = grid2op.make(env_name,
                           param=param,
                           backend=LightSimBackend(),
                           chronics_class=MultifolderWithCache
                           )
        total_train = int(1024) * int(1024)  # ~30 minutes [32-35 mins] # for case 14 50 mins for case 118
        sizes_enc = (20,)
        sizes_main = (150, 150)
        sizes_out = (40,)
        val_regex = ".*99[0-9].*"
        if do_train is False:
            model_name = "realtest_13"
        li_sizes = [1, 3, 10, 30, 100, 300, 1000, 2008, 3000, 10000, 30000, 100000, 300000]
        attr_y = ("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex")
        lr = 3e-4
        scale_main_layer = None
        scale_input_dec_layer = None
        scale_input_enc_layer = None
        layer_act = None
        # I select only part of the data, for training
        # env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is None)
        # env.chronics_handler.real_data.reset()
        obs = env.reset()
    elif env_name == "l2rpn_neurips_2020_track2_small":
        multimix = grid2op.make(env_name,
                                param=param,
                                backend=LightSimBackend(),
                                chronics_class=MultifolderWithCache
                                )
        if do_train is False:
            model_name = "test_118_05"
        sizes_enc = (60, )
        sizes_main = (300, 300, 300, 300)
        sizes_out = (60, )
        val_regex_train = ".*Scenario_february_[1-9][0-9].*"  # don't include the validation set
        val_regex = ".*Scenario_february_0[0-9].*"
        env = multimix[next(iter(sorted(multimix.keys())))]
        li_sizes = [1, 3, 10, 30, 100, 300, 1000, 2304, 3000, 10000, 30000, 100000]  # , 300000]
        attr_y = ("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex")
        lr = 1e-4
        scale_main_layer = 600
        scale_input_dec_layer = None
        scale_input_enc_layer = None
        layer_act = "relu"
        # I select only part of the data, for training
        env.chronics_handler.set_filter(lambda path: re.match(val_regex_train, path) is not None)
        print("... resetting the chronics")
        env.chronics_handler.real_data.reset()
        obs = env.reset()
        print("done")
    else:
        raise RuntimeError("Unsupported environment for now")

    if do_train:
        agent = RandomNN1(env.action_space, p=0.5)
        # agent = DoNothingAgent(env.action_space)
        proxy = ProxyLeapNet(name=model_name,
                             lr=lr,
                             layer=layerfun,
                             layer_act=layer_act,
                             sizes_enc=sizes_enc,
                             sizes_main=sizes_main,
                             sizes_out=sizes_out,
                             scale_main_layer=scale_main_layer,
                             scale_input_dec_layer=scale_input_dec_layer,
                             scale_input_enc_layer=scale_input_enc_layer,
                             attr_y=attr_y)
        agent_with_proxy = AgentWithProxy(agent,
                                          proxy=proxy,
                                          logdir=save_path_tensorbaord
                                          )

        # train it
        agent_with_proxy.train(env,
                               total_train,
                               save_path=save_path
                               )

    # Now proceed with the evaluation
    # I select only part of the data, for training
    if env_name == "l2rpn_case14_sandbox":
        env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is not None)
        env.chronics_handler.real_data.reset()
        obs = env.reset()
    elif env_name == "l2rpn_neurips_2020_track2_small":
        env.chronics_handler.set_filter(lambda path: re.match(val_regex, path) is not None)
        env.chronics_handler.real_data.reset()
        obs = env.reset()
    else:
        raise RuntimeError("Unsupported environment for now")

    # evaluate a baseline
    if do_dc:
        print("#######################"
              "## DC approximation  ##"
              "#######################")
        agent_evalN1 = RandomN1(env.action_space)
        reproducible_exp(env,
                         agent=agent_evalN1,
                         env_seed=env_seed,
                         agent_seed=agent_seed,
                         chron_id=chron_id_val)
        proxy_dc = ProxyBackend(env._init_grid_path,
                                name=f"{model_name}_evalDC",
                                is_dc=True)
        agent_with_proxy_dc = AgentWithProxy(agent_evalN1,
                                             proxy=proxy_dc,
                                             logdir=None)
        agent_with_proxy_dc.init(env)
        agent_with_proxy_dc.evaluate(env,
                                     load_path=None,
                                     save_path=save_path_final_results,
                                     total_evaluation_step=total_evaluation_step,
                                     metrics={"MSE_avg": mean_squared_error,
                                              "MAE_avg": mean_absolute_error,
                                              "NRMSE_avg": nrmse,
                                              "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred,
                                                                                               multioutput="raw_values"),
                                              "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred,
                                                                                                multioutput="raw_values"),
                                              "NRMSE": lambda y_true, y_pred: nrmse(y_true, y_pred,
                                                                                    multioutput="raw_values"),
                                              }
                                     )

    # evaluate this proxy on a similar dataset
    if do_N1:
        print("#######################"
              "##     Test set      ##"
              "#######################")
        agent_evalN1 = RandomN1(env.action_space)
        max_ = np.max(li_sizes)
        for pred_batch_size in li_sizes:
            reproducible_exp(env,
                             agent=agent_evalN1,
                             env_seed=env_seed,
                             agent_seed=agent_seed,
                             chron_id=chron_id_val)
            proxy_eval = ProxyLeapNet(name=f"{model_name}_evalN1",
                                      max_row_training_set=max(total_evaluation_step, pred_batch_size),
                                      eval_batch_size=pred_batch_size,  # min(total_evaluation_step, 1024*64)
                                      layer=layerfun)
            agent_with_proxy_evalN1 = AgentWithProxy(agent_evalN1,
                                                     proxy=proxy_eval,
                                                     logdir=None)

            dict_metrics = agent_with_proxy_evalN1.evaluate(env,
                                             total_evaluation_step=pred_batch_size,
                                             load_path=os.path.join(save_path, model_name),
                                             save_path=save_path_final_results,
                                             metrics={"MSE_avg": mean_squared_error,
                                                      "MAE_avg": mean_absolute_error,
                                                      "NRMSE_avg": nrmse,
                                                      "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred,
                                                                                                       multioutput="raw_values"),
                                                      "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred,
                                                                                                        multioutput="raw_values"),
                                                      "NRMSE": lambda y_true, y_pred: nrmse(y_true, y_pred,
                                                                                            multioutput="raw_values"),
                                                      },
                                                            verbose=pred_batch_size == max_
                                             )
            total_pred_time_ms = 1000.*dict_metrics["predict_time"]
            print(f'Time to compute {pred_batch_size} powerflow: {total_pred_time_ms:.2f}ms ({total_pred_time_ms/pred_batch_size:.4f} ms/powerflow)')
            if pred_batch_size == max_:
                print(proxy_eval._model.summary())
            # import pdb
            # pdb.set_trace()

    # now evaluate this proxy on a different dataset (here we use another "actor" to sample the action and hence the state
    if do_N2:
        print("#######################"
              "##   SuperTest set   ##"
              "#######################")
        agent_evalN2 = RandomN2(env.action_space)
        reproducible_exp(env,
                         agent=agent_evalN2,
                         env_seed=env_seed,
                         agent_seed=agent_seed,
                         chron_id=chron_id_val)
        proxy_eval = ProxyLeapNet(name=f"{model_name}_evalN2",
                                  max_row_training_set=total_evaluation_step,
                                  eval_batch_size=min(total_evaluation_step, 1024*64),
                                  layer=layerfun
                                  )
        agent_with_proxy_evalN2 = AgentWithProxy(agent_evalN2,
                                                 proxy=proxy_eval,
                                                 logdir=None)
        agent_with_proxy_evalN2.evaluate(env,
                                         total_evaluation_step=total_evaluation_step,
                                         load_path=os.path.join(save_path, model_name),
                                         save_path=save_path_final_results,
                                         metrics={"MSE_avg": mean_squared_error,
                                                  "MAE_avg": mean_absolute_error,
                                                  "NRMSE_avg": nrmse,
                                                  "MSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred,
                                                                                                   multioutput="raw_values"),
                                                  "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred,
                                                                                                    multioutput="raw_values"),
                                                  "NRMSE": lambda y_true, y_pred: nrmse(y_true, y_pred,
                                                                                        multioutput="raw_values"),
                                                  },
                                         verbose=1
                                         )
