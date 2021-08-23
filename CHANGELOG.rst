Change Log
===========

[0.0.3] - 2021-08-23
----------------------
- [BREAKING] refactoring the script names to be lower case (class names are still upper case, as in PEP convention)
- [FIXED] The env parameters in "generate_data.py"
- [ADDED] Different way to compute the `tau` vector in the ProxyLeapNet
- [ADDED] `agents` that can change the topology (one substation, or 2 substations)
- [ADDED] the `mape_quantile` loss function that computes the mape on only the highest value (in absolute value)
  of each columns.
