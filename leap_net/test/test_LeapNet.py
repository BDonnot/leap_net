import logging
import os
import numpy as np
import unittest
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from leap_net import Ltau
import pdb


class Test(unittest.TestCase):
    def setUp(self):
        self.tol = 3e-6  # use to compare results that should be strictly equal, up to numerical error
        self.tol_learn = 1e-2  # use to compare results from a test set
        
        # to have "reproducible" results
        np.random.seed(1)
        tf.random.set_seed(1)

    def test_ok_tau0(self):
        dim_x = 10
        n_elem = 5
        dim_tau = 1

        x = Input(shape=(dim_x,), name="x")
        tau = Input(shape=(dim_tau,), name="tau")

        res_model = Ltau()((x, tau))
        model = Model(inputs=[x, tau], outputs=[res_model])

        X_train = np.random.normal(size=(n_elem, dim_x)).astype(np.float32)
        TAU_train = np.zeros(shape=(n_elem, dim_tau), dtype=np.float32)
        res = model.predict([X_train, TAU_train])
        assert np.all(res == X_train)

    def test_ok_tau1(self):
        dim_x = 10
        n_elem = 100
        dim_tau = 1
        X_train = np.random.normal(size=(n_elem, dim_x)).astype(np.float32)
        TAU_train = np.ones(shape=(n_elem, dim_tau), dtype=np.float32)

        # the keras model
        x = Input(shape=(dim_x,), name="x")
        tau = Input(shape=(dim_tau,), name="tau")
        res_model = Ltau(initializer='ones', use_bias=False)((x, tau))
        model = Model(inputs=[x, tau], outputs=[res_model])

        # make predictions
        res = model.predict([X_train, TAU_train])

        # LEAP Net implementation in numpy in case tau is not 0
        res_th = np.matmul(X_train, np.ones((dim_x, dim_tau), dtype=np.float32))
        res_th = np.multiply(res_th, TAU_train)
        res_th = np.matmul(res_th, np.ones((dim_tau, dim_x), dtype=np.float32))
        res_th += X_train
        assert np.mean(np.abs(res - res_th)) <= self.tol, "problem with l1"
        assert np.max(np.abs(res - res_th)) <= self.tol, "problem with linf"

    def test_ok_tau_rand(self):
        dim_x = 10
        n_elem = 100
        dim_tau = 20

        X_train = np.random.normal(size=(n_elem, dim_x)).astype(np.float32)
        TAU_train = np.random.normal(size=(n_elem, dim_tau)).astype(np.float32)

        # the keras model
        x = Input(shape=(dim_x,), name="x")
        tau = Input(shape=(dim_tau,), name="tau")
        res_model = Ltau(initializer='ones', use_bias=False)((x, tau))
        model = Model(inputs=[x, tau], outputs=[res_model])

        # make predictions
        res = model.predict([X_train, TAU_train])

        # LEAP Net implementation in numpy in case tau is not 0
        res_th = np.matmul(X_train, np.ones((dim_x, dim_tau), dtype=np.float32))
        res_th = np.multiply(res_th, TAU_train)
        res_th = np.matmul(res_th, np.ones((dim_tau, dim_x), dtype=np.float32))
        res_th += X_train
        assert np.mean(np.abs(res - res_th)) <= self.tol, "problem with l1"
        assert np.max(np.abs(res - res_th)) <= self.tol, "problem with linf"

    def test_can_learn(self):
        dim_x = 30
        n_elem = 32*32
        dim_tau = 5

        X_train = np.random.normal(size=(n_elem, dim_x)).astype(np.float32)
        TAU_train = np.random.normal(size=(n_elem, dim_tau)).astype(np.float32)

        e = np.random.normal(size=(dim_x, dim_tau)).astype(np.float32)
        d = np.random.normal(size=(dim_tau, dim_x)).astype(np.float32)

        Y_train = np.matmul(X_train, e)
        Y_train = np.multiply(Y_train, TAU_train)
        Y_train = np.matmul(Y_train, d)
        Y_train += X_train

        # the keras model
        x = Input(shape=(dim_x,), name="x")
        tau = Input(shape=(dim_tau,), name="tau")
        res_model = Ltau()((x, tau))
        model = Model(inputs=[x, tau], outputs=[res_model])

        adam_ = tf.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam_, loss='mse')
        ## train it
        model.fit(x=[X_train, TAU_train], y=[Y_train], epochs=200, batch_size=32, verbose=False)

        # test it has learn something relevant
        X_test = np.random.normal(size=(n_elem, dim_x)).astype(np.float32)
        TAU_test = np.random.normal(size=(n_elem, dim_tau)).astype(np.float32)
        Y_test = np.matmul(X_test, e)
        Y_test = np.multiply(Y_test, TAU_test)
        Y_test = np.matmul(Y_test, d)
        Y_test += X_test
        res = model.predict([X_test, TAU_test])
        assert np.mean(np.abs(res - Y_test)) <= self.tol_learn, "problem with l1"
        assert np.max(np.abs(res - Y_test)) <= self.tol_learn, "problem with linf"


if __name__ == "__main__":
    unittest.main()