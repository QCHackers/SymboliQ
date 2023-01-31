
# SymboliQ: A python framework for Symbolic Quantum computation
SymboliQ is an extension of [Sympy's Quantum Mechanics subpackage](https://docs.sympy.org/latest/modules/physics/quantum/index.html).
It's the only Python package we know of that allows for simulation of quantum circuits via 
[Dirac Notation](https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation).
See the [poster](https://github.com/vtomole/SymboliQ/blob/main/examples/poster.pdf)
for an in-depth explanation.

## Installation

The SymboliQ package is available via `pip` and can be installed in your current Python environment with the command:

```
pip install symboliq
```

## Getting started

```python
from sympy.physics.quantum import TensorProduct
from symboliq.dirac_notation import DiracNotation, cx, h, i, x, ket_0,  ket_1

print(DiracNotation(ket_0))
# prints
#|0>

print(DiracNotation(x * ket_1))
# prints
# (|0><1| + |1><0|)*|0>

bell_state = DiracNotation(
        cx * TensorProduct(h, i) * TensorProduct(ket_0, ket_0)
    )
print(bell_state.operate_reduce())
# prints
# sqrt(2)*|0>x|0>/2 + sqrt(2)*|1>x|1>/2
```

### Feature requests / Bugs / Questions
If you have questions, feature requests or you found a bug, [please file them on Github](https://github.com/vtomole/SymboliQ/issues).
