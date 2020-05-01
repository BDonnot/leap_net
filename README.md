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
We also provide a simple implement of the LEAP Net that can be use as a any `tf.keras` in the following way:

TODO

## Cite this work
TODO