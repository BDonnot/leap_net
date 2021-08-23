Change Log
===========

[0.0.4] - 2021-xx-yy
---------------------
- [ADDED] the `mape_quantile` loss function that computes the mape on only the highest value (in absolute value)
  of each columns.

[0.0.3] - 2021-07-20
----------------------
- [BREAKING] refactoring the script names to be lower case (class names are still upper case, as in PEP convention)
- [FIXED] The env parameters in "generate_data.py"
- [ADDED] Different way to compute the `tau` vector in the ProxyLeapNet
- [ADDED] `agents` that can change the topology (one substation, or 2 substations)
