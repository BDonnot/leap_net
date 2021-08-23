# Training and evaluating proxies

In this part of the package we show some examples on how you can leverage the grid2op framework to train
and evaluate the performance of some "proxies".

## What is a proxy
By *proxy* here we mean "something that can approximate the results of a powerflow", "something that can simulate
quite rapidly the effect of an action on a grid", "a (statistical) predictive model that give the state of the grid" and
it can also be called "augmented simulator" in some context.

## General concept

### What is an AgentWithProxy  
In this module, we suppose some agent takes some actions on the grid. This "agent" is fully autonomous: it does not
depends on anything.

To this agent (let's call it the "*actor*") we add a *proxy*. Thanks to a *recorder* (represented by the class
`AgentWithProxy`), the actor will be able to take some actions, and at the same time, the *proxy* will be given
informations about the current state of the grid.

Conceptually, the *recorder* is implemented something like:

```python
from grid2op.Agent import BaseAgent


class AgentWithProxy(BaseAgent):
    def __init__(self, actor, proxy):
        self.proxy = proxy
        self.actor = actor
    
    def act(self, obs, reward, done=False):
        # inform the proxy of the current state of the grid
        self.proxy.current_obs(obs)

        # perform the action of the actor
        action = self.actor.act(obs, reward, done)
        return action            
```
**NB** We emphasize that this is NOT the current implementation. The current implementation is more flexible and allows
for more control of what is going on and how to use the proxy.

In this setting, each time the `AgentWithProxy` (*ie* the *recorder*) sees an action, it gives this information to the
proxy and then do the action of the actor.

### Why an AgentWithProxy ?
Why having this type of specific agent ?

This specific architecture allows to distinguish on one end the "proxy" whose objective is to predict flows, or some
other power system related quantities denoted `y`, that are consequences of injections `x` and topology denoted by `τ`.

One of the important thing we consider in the leap net paper is how a model (*ie* a "proxy") learned from data coming 
from a given distribution of `(x, τ) ~ Dtrain` can generalize to some other distribution `Dtest`.

Using the proposed architecture it's fairly easy to do. You train with a given *actor_train*, which is an agent that will
takes actions. These actions will determine the training set `(x, τ)` for the proxy.

Then once the model is trained, at test time, we can use a different *actor_test* that will produce another distribution
`(x, τ) ~ Dtest` on which the proxy will make predictions.

This makes the testing on a different distribution easy to code and do not require any modification to anything.

### Other usage
That being said, we can also simply use the flexibility offered by this code to train some proxy and test them, 
regardless of the difference in the training distribution.

This module indeed bring all kind of methods to store and retrieve predictions made by the proxy, the state of the
grid etc.


## The "proxy" interface

In this section I will introduce what is the "interface" the proxy must implement in order to be used in this
"framework". The proxy must inherit from [BaseProxy](baseProxy.py) (see the documentation for more information). 

### Creating a custom proxy class
When you want to create a custom proxy, you should implement the following method

- `build_model()`: is used to create the proxy (for a neural network this creates the neural network with the 
  meta parameters given as input for example. Sometimes it does not require to do anything, like for the DC 
  approximation for example.)
- `_make_predictions(data)`: is used when the proxy is asked to make some predictions from the data given as input.
- `_train_model(data)` : [optional in general, but required if your proxy need to be trained]. Not all proxy are
  "trainable" (for example the dc approximation, or a proxy based on a backend is not trainable). But if your
  proxy is made from a "machine learning" algorithm you should implement this methods. **NB** we don't recommend
  to implement the `train` method of the `BaseProxy` class. In case you want a model that is trainable, we highly
  recommend to inherit from [`BaseNNProxy`](baseNNProxy.py) and only implement the "_train_model" method as stated
  here.

And optionally, the "most common" methods that can be implemented are:

- `init(obss)`: obss is a list of observations. This method is used at the beginning (before the proxy is being used)
  to perform initialized some information (*eg* the mean and standard deviation if you want to scale the data for 
  example)
- `load_metadata(dict_)`: in this context "meta data" refers to everything that the proxy needs to be built properly.
  for example if the proxy is a neural network, we can imagine the "meta data" being the number of layers, size of
  each layers, activation function per layers, scaling coefficients etc. 
  This function must, given a dictionary representing valid "metadata" load it properly and initialize the current
  instance with it
- `get_metadata()`: as opposed to the `load_metadata` function, this one should output a valid dictionary that 
  represents the metadata of the proxy (see example in the bullet point above)
- `load_data(path)`: load the data stored at the location "path" (data includes for example the weights of the 
  neural network)
- `save_data(path)`: save the data of the proxy (*eg* the weights of the neural network for example)

And this is it. Nothing else is required.

For examples of usage, there are currently 2 implemented proxies:
- `ProxyLeapNet` a proxy based on a neural network (with a leap net architecture)
- `ProxyBackend` a proxy based on a grid2op backend. Can be used for example to test how precise is the DC 
  approximation

### Initializing  proxy
Once the class is created, a proxy can be created with:

```python

proxy = MyAweSomeProxyClass(name,  # name you give to your proxy, for example "leapnet_on_case14"
                            # number of examples present in the "cache" dataset
                            max_row_training_set=int(1e5),  
                            # how many examples are predicted "at once" / "by batch" during evaluation
                            eval_batch_size=1024,  
                            # what the proxy is expected to use as input
                            attr_x=("prod_p", "prod_v", "load_p", "load_q", "topo_vect"),  
                            # what the proxy is required to predict
                            attr_y=("a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex", "prod_q", "load_v", "v_or", "v_ex"),
                            # other attributes specific to your proxy
                            ...
                            )
```
More information are given in the documentation of [BaseProxy](baseProxy.py).

### Input and output data
The "proxy" is learned from data stored "on the fly" generate by running grid2op environments. These data are stored
in database called `self._my_x` (for the input data) and `self._my_y` for the output data. 

Technically, these database are created once with a given size (key word argument `max_row_training_set` you should 
pass when creating a proxy), so keep in mind this dataset might not be entirely filled when you want to learn directly
from these datasets.

In a common usage, you don't have to worry about anything. **The handling of these "custom datasets filled on the fly" is
taken care __automatically__ by the BaseProxy class**. This is why you just need to create functions to train 
your agent (`_train_model`) on given data and to make predictions with it (`_make_predictions`). 

## Train and evaluate a proxy
After having exposed how to create a class representing a proxy in the previous section, in this section we explain
how to first train a proxy, and then how to evaluate its performance (and what information can be saved).

Once a proxy is created, it's possible to train it with a standard database, by loading data from the hard drive, and
use a standard method to train it. The advantage of this module is that it lets you train a model on data gathered 
"on the fly" using grid2op. For that, the class `AgentWithProxy` has been developed to automate the training /
evaluation process of "proxies". To use it, it supposes that "proxies" given as input implements the 
"proxy interface" described above.

### Training a proxy
Not all proxies require training, for example a proxy based on the DC approximation do not require any training
at all. 

But for proxies that requires it (*eg* proxy based on machine learning method) there is a specific method to do it.

To train a proxy, you can simply run the following code:

```python

# defines where the model will be stored and how long you will train it
total_training_step = int(1024) * int(1024)
save_path = "a/local/path/where/proxy/is/stored"  # a path where you want to save your proxy 

# create the grid2op environment
env = ... 

actor = ...  # create your "actor" here
proxy = ...  # create your proxy here

# now create the AgentWithProxy
agent_with_proxy = AgentWithProxy(actor,
                                  proxy=proxy)

# train the proxy
agent_with_proxy.train(env,
                       total_training_step=total_training_step,
                       save_path=save_path
                       )
```
Of course more option are available if you want to customize the training process. They are (TODO) described in the
documentation.

For a concrete example, you can have a look at the [`train_proxy_case14`](./train_proxy_case14.py) file.

### Evaluating a proxy
Once the proxy is trained, we also made easy the possibility to evaluate a proxy. This includes loading it, restoring
the meta data (if this is a neural network, these are also called "meta parameters" for example number of layers,
size of each layers, activation functions etc.) and the data of the model.

Once this is done, the method "evaluate" will perform the evaluation for you.

```python

from leap_net.proxy import reproducible_exp, AgentWithProxy, DEFAULT_METRICS

# for reproducible experiments (optional)
env_seed = ... # seed used to seed the environment
agent_seed = ... # seed used to seed the actor NOT the proxy
chron_id_start = ...  # id of the grid2op chronics to start

# number of evaluation step
total_evaluation_step = int(1024)
load_path = "a/local/path/where/proxy/is/stored"  # a path where you have saved your proxy
# load_path should corresponds to a "save_path" of the train function (see above)
save_path = "a/local/path/where/evaluation/are/stored"  # a path where the results of the proxy evaluations
# are being stored

# create the grid2op environment
env = ... 

actor = ...  # create your "actor" here (this might be different from the one used for training)
proxy = ...  # create your proxy here

reproducible_exp(env,
                 agent=actor,
                 env_seed=env_seed,
                 agent_seed=agent_seed,
                 chron_id_start=chron_id_start)

agent_with_proxy = AgentWithProxy(actor,
                                  proxy=proxy,
                                  logdir=None)

dict_metrics = agent_with_proxy.evaluate(env,
                                       total_evaluation_step=total_evaluation_step,
                                       load_path=load_path,
                                       save_path=save_path,
                                       metrics=DEFAULT_METRICS,
                                       )
```
Of course more option are available if you want to customize the training process. They are (TODO) described in the
documentation.

For a concrete example, you can have a look at the [`evaluate_proxy_case_14`](./evaluate_proxy_case_14.py) file.

