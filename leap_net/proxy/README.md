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
"framework". The proxy must inherit from `BaseProxy` (see the documentation for more information).

TODO 

And it must absolutely implement the following methods:

- `build_model()`: is used to create the proxy
- `_make_predictions(data)`: is used when the proxy is asked to make some predictions from the data given as input.

ANd optionnally, the "most common" methods that can be implemented are:

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
- `store_obs(obs)`: store a single observation in the database. We are working on simplifiying this part.

TODO : add load_data and save_data

And this is it. Nothing else is required.

For examples of usage, there are currently 2 implemented proxies:
- `ProxyLeapNet` a proxy based on a neural network (with a leap net architecture)
- `ProxyBackend` a proxy based on a grid2op backend. Can be used for example to test how precise is the DC 
  approximation

## Train and evaluate a proxy
After having exposed how to create a class representing a proxy in the previous section, in this section we explain
how to first train a proxy, and then how to evaluate its performance (and what informations can be saved).

TODO 


