import sympy
from sympy.physics.quantum.gate import UGate


def rx(qubit: int, symbol: sympy.Symbol) -> UGate:
    matrix = sympy.ImmutableMatrix([[sympy.cos(symbol / 2), -1j * sympy.sin(symbol / 2)],
                                    [-1j * sympy.sin(symbol / 2), sympy.cos(symbol / 2)]])
    return UGate((qubit,), matrix)


def ry(qubit: int, symbol: sympy.Symbol) -> UGate:
    matrix = sympy.ImmutableMatrix(
        [
            [sympy.cos(symbol / 2), -1 * sympy.sin(symbol / 2)],
            [sympy.sin(symbol / 2), sympy.cos(symbol / 2)],
        ]
    )
    return UGate((qubit,), matrix)


def rz(qubit: int, symbol: sympy.Symbol) -> UGate:
    matrix = sympy.ImmutableMatrix(
        [
            [sympy.cos(symbol / 2), -1 * sympy.sin(symbol / 2)],
            [sympy.sin(symbol / 2), sympy.cos(symbol / 2)],
        ]
    )
    return UGate((qubit,), matrix)
