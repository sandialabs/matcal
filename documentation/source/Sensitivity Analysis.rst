********************
Sensitivity Analysis
********************

Sensitivity Analysis Theory
===========================

LHS
---

Halton
------

The Halton sequence is a deterministic method for generating points in space. Because Halton sequences
have low-discrepancy, meaning they approximate a uniform distribution well, points from the squence are
categorized as quasi-random. The Halton sequence is constructed using a mathematical function called the
radical inverse function to produce numbers on the unit hypercube :math:`[0, 1]^{d}`, where :math:`d` is the dimensionality
of the parameter space. Parameter bounds defined in the ParameterCollection are used to scale the generated
Halton sequence from the unit hypercube to the bounding region defining the parameter space.


Sensitivity Analysis Implementation
===================================


LHS
---

Halton
------
The SciPy :cite:`scipy` implementation of the Halton sequence (scipy.stats.qmc.Halton) is used in MatCal. Scrambling of the
Halton sequence is supported with the "scramble" (bool) keyword (default False) in order to improve the
statistical properties of the sequence, introducing a controlled randomness while preserving the low-discrepancy
structure. The "rng" (int) keyword (default None) enables reproduciblity by allowing users to pass a random generator.


Examples
========





