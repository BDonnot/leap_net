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
from tqdm import tqdm

from collections.abc import Iterable
from grid2op.Agent import BaseAgent

import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOT_OK = True
except ImportError:
    MATPLOT_OK = False

# TODO merge "reproducible exp" as a method of AgentWithProxy
# TODO implement a "I have gathered enough data, now let me learn without gathering more"


class PlotErrorOnGrid:
    """
    TODO move this class in this own file
    this class is used to "project" on the grid the metrics / errors of some proxies.

    it just ensure the projection, and as of creation (October, 26th 2020) it requires a development version
    of grid2op found at https://github.com/BDonnot/Grid2Op

    """
    def __init__(self, env):
        from grid2op.PlotGrid import PlotMatplot
        self.plot_helper = PlotMatplot(env.observation_space)
        self._line_attr = {"a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "v_or", "v_ex"}
        self._load_attr = {"load_p", "load_q", "load_v"}
        self._prod_attr = {"prod_p", "prod_q", "prod_v"}

    def get_fig(self, attr_nm, metrics):
        fig = None
        try:
            # only floating point values are supported at the moment
            metrics = metrics.astype(np.float)
        except Exception as exc_:
            return None

        if np.all(~np.isfinite(metrics)):
            # no need to plot a "all nan" vector
            return None

        if attr_nm in self._prod_attr:
            # deals generator attributes
            self.plot_helper.assign_gen_palette(increase_gen_size=1.5)
            fig = self.plot_helper.plot_info(gen_values=metrics, coloring="gen")
            self.plot_helper.restore_gen_palette()
        elif attr_nm in self._line_attr:
            # deals with lines attributes
            self.plot_helper.assign_line_palette()
            fig = self.plot_helper.plot_info(line_values=metrics, coloring="line")
            self.plot_helper.restore_line_palette()
        return fig


class AgentWithProxy(BaseAgent):
    """
    This class allows to easily manipulate an proxy (training et evaluating) coupled with "an actor" that allows
    for easily evaluate how a proxy deals with data coming from different distribution.

    We do not recommend to modify this class.
    """
    def __init__(self,
                 actor,  # the agent that will take some actions
                 proxy,  # the proxy to train / evaluate
                 logdir=None,  # tensorboard logs
                 update_tensorboard=256,  # tensorboard is updated every XXX training iterations
                 save_freq=int(1024)*int(64),  # model is saved every save_freq training iterations
                 ext=".h5",  # extension of the file in which you want to save the proxy
                 nb_obs_init=256,  # number of observations that are sent to the proxy to be initialized
                 ):
        BaseAgent.__init__(self, actor.action_space)
        self.actor = actor

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
        self.save_freq = int(save_freq)

        # save / load
        if re.match(r"^\.", ext) is None:
            # add a point at the beginning of the extension
            self.ext = f".{ext}"
        else:
            self.ext = ext
        self.save_path = None

    def init(self, env):
        """
        Initialize this object.

        It will first run the environment using the actor on a certain number of steps go gather some information
        and will then initialize the proxy wit these observations.

        Parameters
        ----------
        env:
            The grid2op environment
        """
        if self.__is_init:
            return
        # if the model is in "training mode" then when need to update the proxy with some observations of the
        # environment (basically to compute the scalers for the data)
        if self.is_training:
            # generate a few observation to init the proxy
            obss = []
            nb_obs = 0
            while nb_obs <= self._nb_obs_init:
                done = False
                reward = env.reward_range[0]
                obs = env.reset()
                while not done:
                    act = self.actor.act(obs, reward, done)
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
        """
        Overloading of the grid2op.BaseAgent "act" function that will first store the relevant attribute of the
        observation in the proxy.

        Then, if the proxy need to be trained, train it.

        And if the proxy need to be saved, save it.

        Parameters
        ----------
        obs: ``grid2op.Observation`
            The current grid2op observation.

        reward: ``float``
            The reward of the last action, irrelevant for this class but forwarded to the actor.

        done: ``bool``
            Whether or not there was a game over in the last action. Irrelevant for this class
             but forwarded to the actor.

        Returns
        -------
        act: `grid2op.Action`
            The action chosen by the actor (remember the proxy does not take any decisions here)
        """
        self.global_iter += 1
        self._store_obs(obs)
        if self.is_training:
            batch_losses = self._proxy.train(tf_writer=self._tf_writer)
            if batch_losses is not None:
                self.train_iter += 1
                self._save_tensorboard(batch_losses)
                self._save_model()
        return self.actor.act(obs, reward, done)

    def train(self, env, total_training_step, save_path=None, load_path=None, verbose=1):
        """
        Completely train the proxy. This run a possible entire procedure to train a proxy.

        We highly recommend to call this method to train it.

        Parameters
        ----------
        env:
            The environment

        total_training_step:
            The name is completely misleading. This is not the total number of training step, but the total
            number of data that will be see during training. For example, if the batch size is 32, then
            then `"total number of training step" = "total number of data that will be see during training" / 32`
            this number is "total number of data that will be see during training".

        save_path: ``str``
            Path where the proxy and some other data from this instance will be saved. (``None`` to deactivate it)

        load_path: ``str``
            If it is not None, data stored at the location "path" will be loaded back.

        verbose: ``int``
            Degree of verbosity. The more verbose the more information will be plotted on the command line

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
        with tqdm(total=total_training_step, disable=verbose == 0) as pbar:
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

    def evaluate(self, env, total_evaluation_step, load_path, save_path=None, metrics=None,
                 verbose=0, save_values=True):
        """
        This is a function to evaluate the performance of a proxy.

        We highly recommend to use this method if you coded a model that follows the "proxy" interface.

        Parameters
        ----------
        env:
            The grid2op environment used

        total_evaluation_step:
            Total number of states the proxy will be evaluated on

        load_path:
            Path from which the proxy will be loaded.

        metrics:
            dictionary of function, with keys being the metrics name, and values the function that compute
            this metric (on the whole output) that should be `metric_fun(y_true, y_pred)`

        save_path: ``str``
            Path where the results of the models are stored. This is not the same as the "save_path" argument
            of "train")

        verbose: ``int``
            Degrees of verbosity (the higher the more information will be printed on the command prompt)

        save_values: ``bool``
            Do you save the computed values (prediction of the proxy AND ground truth) or just the metrics

        Returns
        -------
        res: ``dict``
            The dictionary containing the values of each metrics defined in "metrics" for each output variable
            of the proxy.

        """
        if not os.path.exists(save_path):
            print(f"Creating path \"{save_path}\" to save the logs of the trained agent")
            os.mkdir(save_path)

        error_plot = None
        try:
            error_plot = PlotErrorOnGrid(env)
        except Exception as exc_:
            # plotting will not be available, but this is not a reason to crash
            pass

        done = False
        reward = env.reward_range[0]
        self.is_training = False
        self.save_path = None  # disable the saving of the model

        self.load(load_path)
        self.global_iter = 0
        self.train_iter = 0

        # TODO find a better approach for more general proxy that can adapt to grid of different size
        sizes = self._proxy.get_output_sizes()
        true_val = [np.zeros((total_evaluation_step, el), dtype=self._proxy.dtype) for el in sizes]
        pred_val = [np.zeros((total_evaluation_step, el), dtype=self._proxy.dtype) for el in sizes]

        if not self.__is_init:
            self.init(env)
        with tqdm(total=total_evaluation_step, disable=verbose == 0) as pbar:
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
        return self._save_results(obs, save_path, metrics, pred_val, true_val, verbose, save_values, error_plot)

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
            self._proxy.save_data(path=path_save, ext=self.ext)

    def load(self, path):
        """
        Load a previously stored experiments from the hard drive.

        This both load data for this class and from the proxy.

        Parameters
        ----------
        path: ``str``
            Where to load the experiment from.

        """
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
            self._proxy.load_data(path=path, ext=self.ext)

    def get_name(self):
        """get the name of this experiment, that by definition (for now) is the name given to the proxy"""
        return self._proxy.name

    ###############
    ## Utilities ##
    ##############
    def _store_obs(self, obs):
        """
        Utility function that only (for now) tell the proxy to store the observation.

        Parameters
        ----------
        obs: `grid2op.Action.BaseObservation`
            The current observation

        """
        self._proxy.store_obs(obs)

    # save load model
    def _get_path_nn(self, path, name):
        """utilities when path and file names are not formatted the same way"""
        if name is None:
            path_model = path
        else:
            path_model = os.path.join(path, name)
        return path_model

    def _save_metadata(self, path_model):
        """save the dimensions of the models and the scalers"""
        json_nm = "metadata.json"
        me = self._to_dict()
        me["proxy"] = self._proxy.get_metadata()
        with open(os.path.join(path_model, json_nm), "w", encoding="utf-8") as f:
            json.dump(obj=me, fp=f)

    def _load_metadata(self, path_model):
        """load the metadata of the experiments (both for me and for the proxy)"""
        json_nm = "metadata.json"
        with open(os.path.join(path_model, json_nm), "r", encoding="utf-8") as f:
            me = json.load(f)
        self._from_dict(me)
        self._proxy.load_metadata(me["proxy"])

    def _to_dict(self):
        """output a json serializable dictionary representing the current state of the experiment"""
        res = {}
        res["train_iter"] = int(self.train_iter)
        res["global_iter"] = int(self.global_iter)
        return res

    def _from_dict(self, dict_):
        """update my meta data only, do not affect the proxy"""
        self.train_iter = int(dict_["train_iter"])
        self.global_iter = int(dict_["global_iter"])

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
        """trigger the saving of the model"""
        if self.train_iter % self.save_freq == 0:
            self.save(self.save_path)

    def _save_results(self, obs, save_path, metrics, pred_val, true_val, verbose,
                      save_values=True, error_plot=None):
        """
        This function will save the results of the evaluation of the proxy in multiple form:

        - all the "ground truth" for all the variable that the proxy needs to predict
        - all the predicted values for all the variables that the proxy predicted
        - the timing and all metrics in a "metrics.json" file
        - the metadata of the model in the "metada.json" file
        - the data of the model (depending on the model)
        - for all metrics that have been computed, an attempt to display on the figure where the errors are located (
          it uses the grid2op.PlotGrid module)

        Parameters
        ----------
        obs
        save_path
        metrics
        pred_val
        true_val
        verbose
        save_values: ``bool``
            Do I save the arrays (of true and predicted values). If ``False`` only the json is saved
        error_plot:
            Utility to plot the error on the grid in a matplotlib figures

        Returns
        -------

        """
        # TODO save the "x" given to the proxy here too

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

                        # plot the error on the grid layout
                        if error_plot is not None and save_path is not None and MATPLOT_OK:
                            fig = error_plot.get_fig(nm, tmp)
                            if fig is not None:
                                save_path_fig = os.path.join(save_path, self.get_name())
                                if not os.path.exists(save_path_fig):
                                    os.mkdir(save_path_fig)
                                fig.savefig(os.path.join(save_path_fig, f"{metric_name}_{nm}.pdf"))
                                plt.close(fig)
                    else:
                        if verbose >=1:
                            print(f"{metric_name} for {nm}: {tmp:.2f}")
                        dict_metrics[metric_name][nm] = float(tmp)

        # save the numpy arrays (if needed)
        if save_path is not None:
            # save the proxy and the meta data
            self.save(save_path)

            # now the other data
            array_names = self._proxy.get_attr_output_name(obs)
            save_path = os.path.join(save_path, self.get_name())
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if save_values:
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
        obs, reward, done, info = env.step(self.actor.act(obs, reward, done))
        while done:
            # we restart until we find an environment that is not "game over"
            obs = env.reset()
            obs, reward, done, info = env.step(self.actor.act(obs, reward, done))
        return obs
