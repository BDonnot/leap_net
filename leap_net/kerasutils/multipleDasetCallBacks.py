# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import os

from tensorflow import keras
import tensorflow as tf


class MultipleDasetCallBacks(keras.callbacks.Callback):
    """
    This class allows to save the loss (in tensorboard) for multiple dataset during training.

    It uses 1 saver per dataset,
    Examples
    --------
    # training dataset
    Xtrain = np.load(...)
    Ytrain = np.load(...)

    # validation dataset
    Xval = np.load(...)
    Yval = np.load(...)

    # test dataset 1
    Xtest1 = np.load(...)
    Ytest1 = np.load(...)

    # test dataset 2
    Xtest2 = np.load(...)
    Ytest2 = np.load(...)

    # define your model
    model = ...

    # and when you fit it
    loss_callback = MultipleDasetCallBacks((["test_1", Xtest1, Ytest1], ["test_2", Xtest2, Ytest2]), logdir="logs")
    model.fit(x=Xtrain,
              y=Ytrain,
              validation_data=(Xval, Yval),
              ...,  # other argument pass to model.fit
              callbacks=[loss_callback]
              )

    """
    def __init__(self, validation_sets, log_dir, verbose=0, savecsv=True, batch_size=None):
        """
        thank @https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

        :param validation_sets:
        a list of 3-tuples (validation_set_name, validation_data, validation_targets)
        or 4-tuples (validation_set_name, validation_data, validation_targets, sample_weights)

        :param verbose:
        verbosity mode, 1 or 0

        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(MultipleDasetCallBacks, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

        self.tf_writer = {dsn: tf.summary.create_file_writer(os.path.join(log_dir, dsn), name=dsn)
                          for dsn, *_ in validation_sets}
        self.name = os.path.split(log_dir)[-1]
        self.log_dir = log_dir
        self.savecsv = savecsv
        self.res_csv = {dsn: {"epoch": []} for dsn, *_ in validation_sets}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for id_, validation_set in enumerate(self.validation_sets):
            if len(validation_set) == 3:
                validation_set_name, validation_data, validation_targets = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_set_name, validation_data, validation_targets, sample_weights = validation_set
            else:
                raise ValueError()

            results = self.model.evaluate(x=validation_data,
                                          y=validation_targets,
                                          verbose=self.verbose,
                                          sample_weight=sample_weights,
                                          batch_size=self.batch_size)

            for i, result in enumerate(results):
                if i == 0:
                    valuename = 'epoch_loss'
                    colname_csv = 'loss'
                else:
                    valuename = 'epoch_{}_loss'.format(self.model.output_names[i-1])
                    colname_csv = '{}_loss'.format(self.model.output_names[i-1])
                self.history.setdefault(valuename, []).append(result)

                with self.tf_writer[validation_set_name].as_default():
                    tf.summary.scalar(valuename, data=result, step=epoch)

                # save the CSV
                if i == 0:
                    self.res_csv[validation_set_name]["epoch"].append(epoch)
                if not colname_csv in self.res_csv[validation_set_name]:
                    self.res_csv[validation_set_name][colname_csv] = []
                self.res_csv[validation_set_name][colname_csv].append(result)

    def _save_csv(self):
        try:
            import pandas as pd
        except ImportError:
            return

        if self.savecsv:
            for tab_name, json in self.res_csv.items():
                df = pd.DataFrame(json)
                df.to_csv(os.path.join(self.log_dir, "{}.csv".format(tab_name)), index=False, sep=";")

    def on_train_end(self, logs=None):
        self._save_csv()

