Change Log
===========

[0.1.1] - 2024-02-21
--------------------
- [FIXED] Broken tests
- [ADDED] test of both keras v3 and tf_keras implementation (when appropriate)
  (*eg* not for python 3.8 where keras v3 is not available)
- [ADDED] automatic upload on pypi on new version

[0.1.0] - 2024-01-15
----------------------
- [BREAKING] refactoring of the code to use keras >= 3.0 (compatible with 
  tensorflow, pytorch and jax) instead of the now deprecated "tensroflow.keras"
  It means that if you did `from leap_net import Ltau` you don't have to change anything.
  But if you did `from leap_net.ltau import Ltau` then you need to either do `from leap_net import Ltau`
  to use the version using keras >= 3 or `from leap_net.tf_keras import Ltau` for using the 
  oldest (and deprecated) version of the leap_net using tensorflow keras.
- [BREAKING] This is the same for the "kerasutils" module. Now you need to do `from leap_net.tf_keras.kerasutils import XXX`
  instead of `from leap_net.kerasutils import XXX`. This module has not been ported to keras >= 3.0 yet but will be in the short
  term.
- [BREAKING] drop support of python 3.6 and 3.7

[0.0.5] - 2022-06-28
----------------------
- [IMPROVED] now the topo_vect_to_tau=from_list is also able to handle some
  topology when some elements are disconnected under some circumstances

[0.0.4] - 2022-06-23
--------------------
- [FIXED] a bug preventing the symmetry to be properly taken into account in `ProxyLeapNet`
  when "topo_vect_to_tau" was not "raw"
- [FIXED] missing a copy when the "thermal_limit" was in the observation and used by the proxy 
  (it was not copied and this caused some error with recent grid2op versions)


[0.0.3] - 2021-08-23
----------------------
- [BREAKING] refactoring the script names to be lower case (class names are still upper case, as in PEP convention)
- [FIXED] The env parameters in "generate_data.py"
- [ADDED] Different way to compute the `tau` vector in the ProxyLeapNet
- [ADDED] `agents` that can change the topology (one substation, or 2 substations)
- [ADDED] the `mape_quantile` loss function that computes the mape on only the highest value (in absolute value)
  of each columns.
