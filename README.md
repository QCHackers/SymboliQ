![Continuous Integration](https://github.com/SupertechLabs/SupermarQ/actions/workflows/ci.yml/badge.svg)


# SymboliQ: A python framework for Symbolic Quantum computation

A 2022 QIP talk titled “Software of QIP, by QIP, and for QIP” by Dave Bacon asked “What would a symbolic library for quantum computing look like?”. Even though it’s natural for most scientists to perform calculations with symbols, popular quantum programming languages like Cirq and Qiskit don’t emphasize symbolic derivations of expressions such as manipulating quantum state amplitudes symbolically instead of assuming finite precision complex numbers. Sympy: A Python framework for Symbolic Computing contains the capabilities for symbolically manipulating quantum mechanical equations but it doesn't provide an easy way to import existing algorithms and protocols in order to create new ones. SymboliQ builds off of Sympy to create a robust library for expressing quantum algorithms and protocols symbolically.


There is a gap between the notation that's written in most quantum computing
textbooks  exsiting programming languages. The existing programming languages talk
about things in terms on circuits. But you can also expresss quantum things in terms of equations. 
For example, quantum teleportation can be expressed in terms of quantum circuits

There are other ways to describe algorithms that are not just gates. For example in Deutch algorithm you can express
everything in braket notation and sums.

[SupermarQ](https://arxiv.org/abs/2202.11045)


FAQ

Why symobliq computation: Even though the symbolic expression is small, the equivalent numerical expression might be 
very large or even infinite-dimensional. - https://scirate.com/arxiv/2008.06467


Symbolic expressions are also good because as the number of qubits increases, the matrix dimension grows 
exponetially and the computation becomes intractable - https://scirate.com/arxiv/2005.11023

This is beause with symboliq computation, no multiplication is required at all. For example, see how the X gate 
is applied to a state - https://scirate.com/arxiv/2005.11023


Sympy does matrix multiplication to compute the states. It actually doesn't use dirac notation in the background.
SymboliQ uses Dirac notation that's been done in Coq but no one uses Coq, they need to import sympy


## Installation

The SupermarQ package is available via `pip` and can be installed in your current Python environment with the command.

```
pip install supermarq
```

## Using SupermarQ

The benchmarks are defined as classes within `supermarq/benchmarks/`. Each application
defines two methods; `circuit` and `score`. These methods are used to generate the benchmarking circuit and evaluate its performance
after execution on hardware.

The quantum benchmarks within SupermarQ are designed to be scalable, meaning that the benchmarks can be
instantiated and generated for a wide range of circuit sizes and depths.

The [`examples/ghz_example.py`](examples/ghz_example.py) file contains an end-to-end example of how to execute the GHZ benchmark
using [SuperstaQ](https://superstaq.super.tech/). The general workflow is as follows:

```python
import supermarq

ghz = supermarq.benchmarks.ghz.GHZ(num_qubits=3)
ghz_circuit = ghz.circuit()
counts = execute_circuit_on_quantum_hardware(ghz_circuit) # For example, via AWS Braket, IBM Qiskit, or SuperstaQ
score = ghz.score(counts)
```


TODO:
Multi-qubit gates
GUI
Basic algorithms
Maunal applications of rules
