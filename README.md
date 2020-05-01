# leap_net
This repository implements the necessary algorithm and data generation process to reproduce the results published around LEAP Net.

## What is the leap net

### Brief introduction
Suppose you have a "system" $S$ that generates data $y$ from input data $x$. Suppose also that the reponse $y$ of this
system can be modulated depending on some known setpoint $\tau$. 

In our experiments, $S$ was a powergrid, we were interested in predicting $y$ the vector representing the flows
on each powerline of this grid. These flows are determined by the power injected at each "bus" (a bus is the 
word in the power system community close to meaning "nodes") of the $x$ (these injections can be both positive if power 
is injected, typically when there is a production unit, or negative when there power is consumed).

In summary we have : $y = S(x, \tau)$

### References
Leap net has been introduced 
[LEAP nets for power grid perturbations](https://arxiv.org/pdf/1908.08314.pdf) paper available on arxiv publish at the
ESANN conference.

It has been my main focus during my PhD titled
[Deep learning methods for predicting flows in power grids : novel architectures and algorithms](https://tel.archives-ouvertes.fr/tel-02045873/document)
also available online.

More recently, some analytical proofs and further developement where published in the paper 
[LEAP Nets for System Identification and Application to Power Systems](https://www.sciencedirect.com/science/article/abs/pii/S0925231220305051)



## Cite this work
TODO