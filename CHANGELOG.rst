Change Log
===========

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
