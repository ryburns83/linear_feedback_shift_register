[NOTEBOOKS]:
cracking_lfsrs_with_keras_part1.ipynb         (a good starting point)

[MODULES]:
galois_tools.py
lfsr_network_viz.py
m_sequence_viz.py
ml_utils.py

[DESCRIPTION]:
This directory is devoted to the hobbyist study of linear feedback shift registers (LFSRs)
from a machine learning, cryptanalytic perspective. It is well understood that LFSRs themselves
serve as very poor stream ciphers, with the pseudorandom sequences / ciphertext produced by these
linear finite field recursions being vulnerable to correlation attack, among other forms of
cryptanalysis. In this directory, we define software tools for the Pythonic implementation and study
of LFSRs. Also defined are Tensorflow/Keras models for prediction of LFSR states and recursion
formulae, for example, from the binary m-sequences these recursions produce. These approaches
treat N-length windows of bits as N-length register state vectors. Neural networks can use these
observations to infer properties of the LFSR. For example, in cracking_lfsrs_with_keras_part1.ipynb,
we use the n'th LFSR state as a basis for prediction of the (n + 1)'st state by a 2-layer feedforward
binary neural network. This is an example of a simple and shallow ML model implictly learning the
finite field functions underlying the LFSR's recurrence relation. This code has been proven to work
for all primitive polynomials of degree 10 in ./primitive_polynomials/10.txt, but is suspected to
generalize to most other degrees 4-32, for which coefficients lists are stored as hex values locally.

For better documentation of the components of this repository, have a look around. Much of the
documentation is inline. This is just for fun. Thanks for visiting!

[DATA]:
All data is generated with finite field arithmetic. The coefficients for the shift registers which
generate this data can be found for primitive polynomial degrees 4-32 at ./primitive_polynomials/.

[AUTHOR]:
Ryan Burns