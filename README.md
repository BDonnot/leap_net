# leap_net
This repository implements the necessary algorithm and data generation process to reproduce the results published around LEAP Net.

## What is the leap net

### Brief introduction
Suppose you have a "system" `S` that generates data `y` from input data `x`. Suppose also that the response `y` of this
system can be modulated depending on some known setpoint `τ`. 

In our experiments, `S` was a powergrid, we were interested in predicting `y` the vector representing the flows
on each powerline of this grid. These flows are determined by the power injected at each "bus" (a bus is the 
word in the power system community close to meaning "nodes") of the `x` (these injections can be both positive if power 
is injected, typically when there is a production unit, or negative when there power is consumed). The vector `τ`
encodes for variation of the topology of the powergrid, typically "is a powerline connected or disconnected" and 
"is this powerline connected to this other powerline".

In summary we suppose a generation process `y = S(x, τ)`. We also suppose that we have some dataset
`{(x_i, τ_i, y_i)}` that was generated using this model with input data coming from a distribution `D = Dist(x, τ)`. 
The LEAP net is a "novel" neural network architecture that is 
able to predict some response `ŷ_i` from  `x_i` and `τ_i` with the following properties
 
- it is fully trainable by stochastic gradient descent like any neural network
- its implementation (given here in keras) is really simple
- on data `(x, τ)` drawn from the same distribution `D` than the one use for training `ŷ` is good approximation
- most importantly, under some circumstances, even if `(x, τ)` is **NOT** drawn from the distribution used to train it.

We call this last property "*super generalization*". This is somewhat related to transfer learning and *zero shot* /
*few shots* learning. We explored this super-generalization properties with discrete modulation `τ` in the case
where, for example, the neural network is learned when the system `S` has **zero** or **one** 
disconnected powerline but it's still able to perform accurate prediction even when **two** powerlines are disconnected
at the same time.

Internally, we also made some experiments on load forecast, where the input `x` included the past realized loads and
the weather forecast for example. The modulating
variable `τ` included the properties of the day to predict *eg* is it a monday or a sunday ? It is during bank holiday ?
Is there lots of road traffic (indicating possible start of holidays) this day ? etc. On another topic we also studied 
this model in the context of generative models (cVAE or GANs) where `x` was noise, `y` MNIST images and the modulator 
`τ` included the color or rotation of the generated digits.

LEAP Net gave also pretty good results on all these tasks that we didn't had time to polish for publishing. This makes
us believe that the leap net model is suited in different context than powergrid related application and usable
for modulator `τ` both discrete and continuous.

### References
To know more about the leap Net, you can have a look at the
[LEAP nets for power grid perturbations](https://arxiv.org/pdf/1908.08314.pdf) paper available on arxiv publish at the
ESANN conference.

It has been my main focus during my PhD titled
[Deep learning methods for predicting flows in power grids : novel architectures and algorithms
](https://tel.archives-ouvertes.fr/tel-02045873/document)
also available online.

More recently, some analytical proofs and further development where published in the paper 
[LEAP Nets for System Identification and Application to Power Systems
](https://www.sciencedirect.com/science/article/abs/pii/S0925231220305051)

## Use the leap net

### Reproducing results of the neuro computing paper.
The repository [neurocomputing_paper](./neurocomputing_paper) contains the necessary material to reproduce the figures
presented in the paper. **NB** as of writing, a commercial solver was used to compute the powerflows. We are trying to 
port the code to use the [Grid2Op](https://github.com/rte-france/Grid2Op) framework instead.

### Use the LEAP Net

#### Setting up
##### Quick and dirty way
Of course, this way of doing is absolutely not recommended. By doing it you need to make sure the license of your
own code is compatible with the license of this specific package etc. You have more information on this topic in the
[LICENSE](LICENSE) file.

The most simple way to use the LEAP Net, and especially the Ltau module is to define this class in your project:
```python
# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import add as tfk_add
from tensorflow.keras.layers import multiply as tfk_multiply


class Ltau(Layer):
    """
    This layer implements the Ltau layer.

    This kind of leap net layer computes, from their input `x`: `d.(e.x * tau)` where `.` denotes the
    matrix multiplication and `*` the elementwise multiplication.

    """
    def __init__(self, initializer='glorot_uniform', use_bias=True, trainable=True, name=None, **kwargs):
        super(Ltau, self).__init__(trainable=trainable, name=name, **kwargs)
        self.initializer = initializer
        self.use_bias=use_bias
        self.e = None
        self.d = None

    def build(self, input_shape):
        is_x, is_tau = input_shape
        nm_e = None
        nm_d = None
        if self.name is not None:
            nm_e = '{}_e'.format(self.name)
            nm_d = '{}_d'.format(self.name)
        self.e = Dense(is_tau[-1],
                       kernel_initializer=self.initializer,
                       use_bias=self.use_bias,
                       trainable=self.trainable,
                       name=nm_e)
        self.d = Dense(is_x[-1],
                       kernel_initializer=self.initializer,
                       use_bias=False,
                       trainable=self.trainable,
                       name=nm_d)

    def call(self, inputs):
        x, tau = inputs
        tmp = self.e(x)
        tmp = tfk_multiply([tau, tmp])  # element wise multiplication
        tmp = self.d(tmp)
        res = tfk_add([x, tmp])
        return res
```
This is the complete code of the Ltau module that you can use as any keras layer.


##### Clean installation (from source)
We also provide a simple implement of the LEAP Net that can be use as a any `tf.keras` layer. First you have to 
download this github repository:
```bash
git clone https://github.com/BDonnot/leap_net.git
cd leap_net
```
Then you need to install it (we strongly encourage to install it in a virtual envrionment):
```bash
pip install -U .
```
Then, **as all python packages installed from source** you need to change the current working directory to use this
module:
```bash
cd ..
rm -rf leap_net  # optionnally you can also delete the repository
```
In the future, to ease the installation process, we might provide a version of this package on pypi soon, 
but haven't done that at the moment. If you would like this feature, write us an issue on github.

#### LeapNet usage
Once installed, this package provide a keras-compatible of the `Ltau` block defined in the cited papers. Supposes you 
have at your disposal:
- a `X` matrix of dimension (nb_row, dim_x)
- a `T` matrix of dimension (nb_row, dim_tau)
- a `Y` matrix of dimentsion (nb_row, dim_x)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from leap_net import Ltau  # this import might change if you use the "quick and dirty way".

# create the keras model
x = Input(shape=(dim_x,), name="x")
tau = Input(shape=(dim_tau,), name="tau")
res_Ltau = Ltau()((x, tau))
model = Model(inputs=[x, tau], outputs=[res_Ltau])

# "compile" the model with a given optimizer
adam_ = tf.optimizers.Adam(lr=1e-3)
model.compile(optimizer=adam_, loss='mse')
# train it
model.fit(x=[X, T], y=[Y], epochs=200, batch_size=32, verbose=False)

# make prediction out of it
y_hat = model.predict([X, T])
```

Of course, it is more than recommended to first encode your input data `X` with an encore denoted by `E` on the paper
and then decode them with a "decoder" denoted by `D` in the papers. An example of such a model is:
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Dense
from tensorflow.keras.models import Model
from leap_net import Ltau  # this import might change if you use the "quick and dirty way".

# create the keras model
x = Input(shape=(dim_x,), name="x")
tau = Input(shape=(dim_tau,), name="tau")

## create E, for example with 2 layers of size "layer_size"
layer1 = Dense(layer_size)(x)
layer1 = Activation("relu")(layer1)

layer2 = Dense(layer_size)(x)
layer2 = Activation("relu")(layer1)
# layer2 is the output of E.

## this is Ltau
res_Ltau = Ltau()((layer2, tau))

## now create D, in this case hidden layer, for example
layer4 = Dense(layer_size)(res_Ltau)
layer4 = Activation("relu")(layer4)

# and make the standard (if you do a regression) linear layer for the output
output = Dense(dim_y)(layer4)

model = Model(inputs=[x, tau], outputs=[output])

# "compile" the model with a given optimizer
adam_ = tf.optimizers.Adam(lr=1e-3)
model.compile(optimizer=adam_, loss='mse')
# train it
model.fit(x=[X, T], y=[Y], epochs=200, batch_size=32, verbose=False)

# make prediction out of it
y_hat = model.predict([X, T])
```

**NB** We think the variable we use above are transparent, and we let the user of this work fine tune the learning
rate, the optimizer, the number of epochs the even the size of the batch to suit their purpose. 

**NB** To use this model easily, we suppose you already format your dataset to have the shape `{(x_i, τ_i, y_i)}` and
in particular that you have a pre-defined encoding of your modulator `τ` in the form of a vector. The performance of
the LEAP Net can vary depending on the encoding you choose for `τ`. More information will be provided in the near 
future when we will release a port of the code we used to get our results for the neurcomputing paper. We remind
that this port of the code will not be strictly equivalent to the original implementation of the paper that uses a 
proprietary powerflow as this code will use the open source [Grid2Op](https://github.com/rte-france/Grid2Op) framework, 
that as not available when the paper was first submitted.

## Cite this work
If you use this work please cite:
```
@article{DONON2020,
title = "LEAP nets for system identification and application to power systems",
journal = "Neurocomputing",
year = "2020",
issn = "0925-2312",
doi = "https://doi.org/10.1016/j.neucom.2019.12.135",
url = "http://www.sciencedirect.com/science/article/pii/S0925231220305051",
author = "B. Donon and B. Donnot and I. Guyon and Z. Liu and A. Marot and P. Panciatici and M. Schoenauer",
keywords = "System identification, Latent space, Residual networks, LEAP Net, Power systems",
abstract = "Using neural network modeling, we address the problem of system identification for continuous multivariate systems, whose structures vary around an operating point. Structural changes in the system are of combinatorial nature, and some of them may be very rare; they may be actionable for the purpose of controlling the system. Although our ultimate goal is both system identification and control, we only address the problem of identification in this paper. We propose and study a novel neural network architecture called LEAP net, for Latent Encoding of Atypical Perturbation. Our method maps system structure changes to neural net structure changes, using structural actionable variables. We demonstrate empirically that LEAP nets can be trained with a natural observational distribution, very concentrated around a “reference” operating point of the system, and yet generalize to rare (or unseen) structural changes. We validate the generalization properties of LEAP nets theoretically in particular cases. We apply our technique to power transmission grids, in which high voltage lines are disconnected and re-connected with one-another from time to time, either accidentally or willfully. We discuss extensions of our approach to actionable variables, which are continuous (instead of discrete, in the case of our application) and make connections between our problem setting, transfer learning, causal inference, and reinforcement learning."
}
```