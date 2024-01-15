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
from datetime import datetime

from leap_net.tf_keras import Ltau, ResNetLayer
from leap_net.tf_keras.kerasutils import MultipleDasetCallBacks

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate as k_concatenate
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


def encode(inputs,
           lss=[10 for _ in range(5)],
           name=None,
           builder=Dense,
           act="relu"):
    """
    This function creates a series of layer whose size are given by lss.

    The input layer of this "series of layer" is given in `inputs`

    Parameters
    ----------
    inputs: ``keras tensor``
        The input layer

    lss: ``list``
        List of size of each layer

    name: ``str``
        Additional name to be added

    builder: ``keras contrstrutor``
        Typically "tf.keras.layers.Dense" or anything with the same behaviour

    act: ``str``
        Name of the activation function. You can put "linear" if you don't want any activation function.


    Returns
    -------
    res: ``keras layer``
        The output layer

    """
    tmp = inputs
    for i, ls in enumerate(lss):
        nm = None
        if name is not None:
            nm = "{}_layer{}".format(name, i)
        tmp = builder(ls, name=nm)(tmp)
        if name is not None:
            nm = "{}_act{}".format(name, i)
        tmp = Activation(act, name=nm)(tmp)
    return tmp


def get_model(n_gen,
              n_load,
              n_line,
              n_sub,
              dim_topo,
              dim_tau,
              lr=1e-3,
              leap=True,
              act="relu",
              builder=Dense):
    """
    Build a model from the parameters given as input.

    THis is a work in progress, but is flexible enough to code every type of neural networks used in the papers
    mentioned in the readme.

    Parameters
    ----------
    n_gen: ``int``
        Number of generator of the grid

    n_load: ``int``
        Number of loads  in the grid

    n_line: ``int``
        Numbre of powerline in the grid.

    n_sub: ``int``
        Number of substations in the grid

    dim_topo: ``int``
        Total number of objects (each ends of a powerline, a load or a generator) of the grid

    dim_tau: ``int``
        Dimention of the tau vector

    lr: ``float``
        Which learing rate to use

    leap: ``bool``
        Whether to use LEAP Net or ResNet

    act: ``str``
        Name of the activation function to use

    builder: ``keras builder``
        Typically "keras.layers.Dense". Type of layer to make.

    Returns
    -------
    model: ``keras model``
        THe compiled keras model. THis might change in the future.

    """
    facto = 3

    # encoding part
    nb_layer_enc = 0
    size_layer_enc_p = facto * 2 * n_gen
    size_layer_enc_c = facto * 2 * n_load
    size_layer_enc_t = facto * 2 * dim_tau

    # now E
    nb_layer_E = 3
    size_layer_E = facto * 3 * n_line

    # number of leap layers
    nb_leap = 3

    # now D
    nb_layer_D = 0
    size_layer_D = 25

    # regular input
    pp_ = Input(shape=(n_gen,), name="prod_p")
    pv_ = Input(shape=(n_gen,), name="prod_v")
    cp_ = Input(shape=(n_load,), name="load_p")
    cq_ = Input(shape=(n_load,), name="load_q")

    # modulator input tau
    tau_ = Input(shape=(dim_tau,), name="tau")

    # encode regular inputs
    pp_e = encode(pp_, lss=[size_layer_enc_p for _ in range(nb_layer_enc)], builder=builder)
    pv_e = encode(pv_, lss=[size_layer_enc_p for _ in range(nb_layer_enc)], builder=builder)
    cp_e = encode(cp_, lss=[size_layer_enc_c for _ in range(nb_layer_enc)], builder=builder)
    cq_e = encode(cq_, lss=[size_layer_enc_c for _ in range(nb_layer_enc)], builder=builder)

    if not leap:
        tau_e = encode(tau_, lss=[size_layer_enc_t for _ in range(nb_layer_enc)], builder=builder)

    # now concatenate everything
    li = [pp_e, pv_e, cp_e, cq_e]
    if not leap:
        li.append(tau_e)
    input_E_raw = k_concatenate(li)
    input_E_raw = Activation(act)(input_E_raw)

    # scale up to have same size of the E part between ResNet and LEAPNet
    # input_E = Dense()(input_E)
    if nb_layer_enc > 0:
        size_resnet = 2 * (size_layer_enc_p + size_layer_enc_c) + size_layer_enc_t
    else:
        size_resnet = 2 * (n_gen + n_load) + dim_tau
    input_E = Dense(size_resnet, name="rescale")(input_E_raw)
    input_E = Activation(act)(input_E)

    # and compute E
    E = encode(input_E, lss=[size_layer_E for _ in range(nb_layer_E)], builder=builder)

    # now apply Ltau
    tmp = E
    for i in range(nb_leap):
        if leap:
            tmp = Ltau(name="Ltau_{}".format(i))((tmp, tau_))
        else:
            tmp = ResNetLayer(dim_tau, name="RestBlock_{}".format(i))(tmp)
    E_modulated = tmp

    # decode it
    D = encode(E_modulated, lss=[size_layer_D for _ in range(nb_layer_D)])

    # linear output
    flow_a_hat = Dense(n_line, name="flow_a_hat")(D)
    flow_p_hat = Dense(n_line, name="flow_p_hat")(D)
    line_v_hat = Dense(n_line, name="line_v_hat")(D)

    model = Model(inputs=[pp_, pv_, cp_, cq_, tau_], outputs=[flow_a_hat, flow_p_hat, line_v_hat])

    adam_ = tf.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam_, loss='mse')
    return model


def load_dataset(path, name):
    """
    Helper to load the datasets, that are now given as numpy arrays. But it might change.
    """
    res = np.load(os.path.join(path, "{}.npy".format(name))).astype(np.float32)
    return res


def compute_loss(dict_tmp, model, Xdatasets, Ydatasets):
    """
    Computes the loss on each outputs if `model` is evaluated on inputs `Xdatasets` and the real output ares `Ydatasets`.

    `dict_tmp` is a dictionnary that is used to store the loss to be more easily extracted later.
    """
    (prod_p_test, scaler_pp), (prod_v_test, scaler_pv), (load_p_test, scaler_cp), (load_q_test, scaler_cq), tau_test = Xdatasets
    (flow_a_test, scaler_fa), (flow_p_test, scaler_fp), (line_v_test, scaler_fv) = Ydatasets
    flow_a_hat, flow_p_hat, line_v_hat = model.predict([scaler_pp.transform(prod_p_test),
                                                        scaler_pv.transform(prod_v_test),
                                                        scaler_cp.transform(load_p_test),
                                                        scaler_cq.transform(load_q_test),
                                                        tau_test])
    flow_a_hat = scaler_fa.inverse_transform(flow_a_hat)
    flow_p_hat = scaler_fp.inverse_transform(flow_p_hat)
    line_v_hat = scaler_fv.inverse_transform(line_v_hat)

    for nm_arr, arr_hat, arr_true in zip(["flow_a", "flow_p", "line_v"],
                                         [flow_a_hat, flow_p_hat, line_v_hat],
                                         [flow_a_test, flow_p_test, line_v_test]):
        nm = "{}_rmse".format(nm_arr)
        rmse_ = float(np.sqrt(mean_squared_error(arr_true, arr_hat)))
        if nm in dict_tmp:
            dict_tmp[nm].append(rmse_)
        else:
            dict_tmp[nm] = [rmse_]


def main(p,
         nb_epoch=1,
         batch_size=32,
         lr=3e-4,
         logdir="logs/",
         path_data="data"):
    """
    Main function to train the desired model (build using `get_model` and evaluate it on the 3 extra datasets:
    -  val dataset: generated with the exact same distribution as the training dataset
    - test dataset: generated with a distribution relatively close from the training dataset but consisting only of
      one change
    - super test dataset: generated with a distribution never seen in the training dataset consisting in making two
      individual changes (while only 1 change at most was made when generating the training set)

    Parameters
    ----------
    p: ``float``
        What is the probability to have "one change" in the training set.

    nb_epoch: ``int``
        Number of epoch for which to train the model

    batch_size: ``int``
        Size of the batch on which the model will be trained

    lr: ``float``
        Learning rate used for training.

    logdir: ``str``
        A path where the models outcome will be stored.

    path_data: ``str``
        Path where to look for the training / validation / test / supertest datasets

    """
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    datetime_start = "{:%Y%m%d-%H%M%S}".format(datetime.now())

    expe_summary_path = os.path.join(logdir, "expe_summary.json")
    if os.path.exists(expe_summary_path):
        with open(expe_summary_path, "r") as f:
            dict_previous_all = json.load(f)
    else:
        dict_previous_all = {}

    p_str = "{:.3f}".format(p)
    if not p_str in dict_previous_all:
        dict_previous_all[p_str] = {}
    dict_previous = dict_previous_all[p_str]

    path_data_train = os.path.join(path_data, "training_set_{:.3f}".format(p))
    path_data_test = os.path.join(path_data, "test_set")
    path_data_supertest = os.path.join(path_data, "supertest_set")
    path_data_val = os.path.join(path_data, "liketrain_set_{:.3f}".format(p))

    # load taining data
    prod_p = load_dataset(path_data_train, "prod_p")
    prod_v = load_dataset(path_data_train, "prod_v")
    load_p = load_dataset(path_data_train, "load_p")
    load_q = load_dataset(path_data_train, "load_q")
    tau = load_dataset(path_data_train, "tau")
    flow_a = load_dataset(path_data_train, "flow_a")
    flow_p = load_dataset(path_data_train, "flow_p")
    line_v = load_dataset(path_data_train, "line_v")

    # scale the data to have variance 1 and mean 0 by column (easier learning)
    scaler_pp = preprocessing.StandardScaler().fit(prod_p)
    scaler_pv = preprocessing.StandardScaler().fit(prod_v)
    scaler_cp = preprocessing.StandardScaler().fit(load_p)
    scaler_cq = preprocessing.StandardScaler().fit(load_q)
    scaler_fa = preprocessing.StandardScaler().fit(flow_a)
    scaler_fp = preprocessing.StandardScaler().fit(flow_p)
    scaler_fv = preprocessing.StandardScaler().fit(line_v)

    # load validation data (same distribution as training dataset)
    prod_p_val = load_dataset(path_data_val, "prod_p")
    prod_v_val = load_dataset(path_data_val, "prod_v")
    load_p_val = load_dataset(path_data_val, "load_p")
    load_q_val = load_dataset(path_data_val, "load_q")
    tau_val = load_dataset(path_data_val, "tau")
    flow_a_val = load_dataset(path_data_val, "flow_a")
    flow_p_val = load_dataset(path_data_val, "flow_p")
    line_v_val = load_dataset(path_data_val, "line_v")

    # load test data
    prod_p_test = load_dataset(path_data_test, "prod_p")
    prod_v_test = load_dataset(path_data_test, "prod_v")
    load_p_test = load_dataset(path_data_test, "load_p")
    load_q_test = load_dataset(path_data_test, "load_q")
    tau_test = load_dataset(path_data_test, "tau")
    flow_a_test = load_dataset(path_data_test, "flow_a")
    flow_p_test = load_dataset(path_data_test, "flow_p")
    line_v_test = load_dataset(path_data_test, "line_v")

    # load supertest data
    prod_p_supertest = load_dataset(path_data_supertest, "prod_p")
    prod_v_supertest = load_dataset(path_data_supertest, "prod_v")
    load_p_supertest = load_dataset(path_data_supertest, "load_p")
    load_q_supertest = load_dataset(path_data_supertest, "load_q")
    tau_supertest = load_dataset(path_data_supertest, "tau")
    flow_a_supertest = load_dataset(path_data_supertest, "flow_a")
    flow_p_supertest = load_dataset(path_data_supertest, "flow_p")
    line_v_supertest = load_dataset(path_data_supertest, "line_v")

    # define the values for the callbacks
    for_call_backval = ("val",
                         [scaler_pp.transform(prod_p_val),
                                                        scaler_pv.transform(prod_v_val),
                                                        scaler_cp.transform(load_p_val),
                                                        scaler_cq.transform(load_q_val),
                                                        tau_val],
                        [scaler_fa.transform(flow_a_val),
                                     scaler_fp.transform(flow_p_val),
                                     scaler_fv.transform(line_v_val)]
                         )
    for_call_backtest = ("test",
                         [scaler_pp.transform(prod_p_test),
                                                        scaler_pv.transform(prod_v_test),
                                                        scaler_cp.transform(load_p_test),
                                                        scaler_cq.transform(load_q_test),
                                                        tau_test],
                        [scaler_fa.transform(flow_a_test),
                                     scaler_fp.transform(flow_p_test),
                                     scaler_fv.transform(line_v_test)]
                         )
    for_call_backsupertest = ("supertest",
                         [scaler_pp.transform(prod_p_supertest),
                                                        scaler_pv.transform(prod_v_supertest),
                                                        scaler_cp.transform(load_p_supertest),
                                                        scaler_cq.transform(load_q_supertest),
                                                        tau_supertest],
                        [scaler_fa.transform(flow_a_supertest),
                                     scaler_fp.transform(flow_p_supertest),
                                     scaler_fv.transform(line_v_supertest)]
                         )

    # define and fit the LEAP model
    tf.keras.backend.clear_session()
    model = get_model(prod_p.shape[1], load_p.shape[1], flow_a.shape[1], None, None, tau.shape[1], lr=lr)
    logdir_leap = os.path.join(logdir, "LEAPNet_{:.3f}_{}".format(p, datetime_start))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir_leap)
    loss_callback = MultipleDasetCallBacks([for_call_backval, for_call_backtest, for_call_backsupertest],
                                             log_dir=logdir_leap
                                             )

    model.fit(x=[scaler_pp.transform(prod_p),
                 scaler_pv.transform(prod_v),
                 scaler_cp.transform(load_p),
                 scaler_cq.transform(load_q),
                 tau],
              y=[scaler_fa.transform(flow_a),
                 scaler_fp.transform(flow_p),
                 scaler_fv.transform(line_v)],
              epochs=nb_epoch,
              batch_size=batch_size,
              verbose=0,
              callbacks=[loss_callback, tensorboard_callback]
              )

    if not "LEAP" in dict_previous:
        dict_previous["LEAP"] = {}

    dict_previous["LEAP"]["nb_params"] = int(model.count_params())

    # and now make predictions and store results
    if not "val" in dict_previous["LEAP"]:
        dict_previous["LEAP"]["val"] = {}
    dict_tmp = dict_previous["LEAP"]["val"]
    Xdatasets = (prod_p_val, scaler_pp), (prod_v_val, scaler_pv), (load_p_val, scaler_cp), (load_q_val, scaler_cq), tau_val
    Ydatasets = (flow_a_val, scaler_fa), (flow_p_val, scaler_fp), (line_v_val, scaler_fv)
    compute_loss(dict_tmp, model, Xdatasets, Ydatasets)

    if not "test" in dict_previous["LEAP"]:
        dict_previous["LEAP"]["test"] = {}
    dict_tmp = dict_previous["LEAP"]["test"]
    Xdatasets = (prod_p_test, scaler_pp), (prod_v_test, scaler_pv), (load_p_test, scaler_cp), (load_q_test, scaler_cq), tau_test
    Ydatasets = (flow_a_test, scaler_fa), (flow_p_test, scaler_fp), (line_v_test, scaler_fv)
    compute_loss(dict_tmp, model, Xdatasets, Ydatasets)

    if not "supertest" in dict_previous["LEAP"]:
        dict_previous["LEAP"]["supertest"] = {}
    dict_tmp = dict_previous["LEAP"]["supertest"]
    Xdatasets = (prod_p_supertest, scaler_pp), (prod_v_supertest, scaler_pv), (load_p_supertest, scaler_cp), (load_q_supertest, scaler_cq), tau_supertest
    Ydatasets = (flow_a_supertest, scaler_fa), (flow_p_supertest, scaler_fp), (line_v_supertest, scaler_fv)
    compute_loss(dict_tmp, model, Xdatasets, Ydatasets)

    with open(expe_summary_path, "w", encoding="utf-8") as f:
        json.dump(obj=dict_previous_all, fp=f, sort_keys=True, indent=4)

    tf.keras.backend.clear_session()
    model_resnet = get_model(prod_p.shape[1], load_p.shape[1], flow_a.shape[1], None, None, tau.shape[1], leap=False, lr=lr)
    logdir_resnet = os.path.join(logdir, "ResNet_{:.3f}_{}".format(p, datetime_start))
    tensorboard_callback_resnet = keras.callbacks.TensorBoard(log_dir=logdir_resnet)
    loss_callback_resnet = MultipleDasetCallBacks([for_call_backval, for_call_backtest, for_call_backsupertest],
                                                  log_dir=logdir_resnet
                                                    )
    model_resnet.fit(x=[scaler_pp.transform(prod_p),
                        scaler_pv.transform(prod_v),
                        scaler_cp.transform(load_p),
                        scaler_cq.transform(load_q),
                        tau],
                     y=[scaler_fa.transform(flow_a),
                        scaler_fp.transform(flow_p),
                        scaler_fv.transform(line_v)],
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     verbose=0,
                     callbacks=[loss_callback_resnet, tensorboard_callback_resnet]
                     )

    if not "ResNet" in dict_previous:
        dict_previous["ResNet"] = {}

    dict_previous["ResNet"]["nb_params"] = int(model.count_params())
    #print("LEAP Net model has {} parameters".format(model.count_params()))

    # and now make predictions and store results
    if not "val" in dict_previous["ResNet"]:
        dict_previous["ResNet"]["val"] = {}
    dict_tmp = dict_previous["ResNet"]["val"]
    Xdatasets = (prod_p_val, scaler_pp), (prod_v_val, scaler_pv), (load_p_val, scaler_cp), (load_q_val, scaler_cq), tau_val
    Ydatasets = (flow_a_val, scaler_fa), (flow_p_val, scaler_fp), (line_v_val, scaler_fv)
    compute_loss(dict_tmp, model_resnet, Xdatasets, Ydatasets)

    if not "test" in dict_previous["ResNet"]:
        dict_previous["ResNet"]["test"] = {}
    dict_tmp = dict_previous["ResNet"]["test"]
    Xdatasets = (prod_p_test, scaler_pp), (prod_v_test, scaler_pv), (load_p_test, scaler_cp), (load_q_test, scaler_cq), tau_test
    Ydatasets = (flow_a_test, scaler_fa), (flow_p_test, scaler_fp), (line_v_test, scaler_fv)
    compute_loss(dict_tmp, model_resnet, Xdatasets, Ydatasets)

    if not "supertest" in dict_previous["ResNet"]:
        dict_previous["ResNet"]["supertest"] = {}
    dict_tmp = dict_previous["ResNet"]["supertest"]
    Xdatasets = (prod_p_supertest, scaler_pp), (prod_v_supertest, scaler_pv), (load_p_supertest, scaler_cp), (load_q_supertest, scaler_cq), tau_supertest
    Ydatasets = (flow_a_supertest, scaler_fa), (flow_p_supertest, scaler_fp), (line_v_supertest, scaler_fv)
    compute_loss(dict_tmp, model_resnet, Xdatasets, Ydatasets)

    with open(expe_summary_path, "w", encoding="utf-8") as f:
        json.dump(obj=dict_previous_all, fp=f, sort_keys=True, indent=4)


if __name__ == "__main__":
    nb_epoch = 20
    batch_size = 32
    lr = 3e-4
    logdir = "logs_10epoch/"
    for i in range(10):
        for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.5]:
            main(p=p, nb_epoch=nb_epoch, batch_size=batch_size, lr=lr, logdir=logdir)