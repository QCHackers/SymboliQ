from sympy import sqrt
from sympy.physics.quantum import Ket, TensorProduct

from symboliq.dirac_notation import (
    b_3,
    ket_0,
    ket_1,
    X,
    I_2,
    H,
    bra_0,
    CX,
    b_1,
    b_2,
    DiracNotation,
    B0,
    Ket0,
)


def test_base_reduce():
    dirac_notation = DiracNotation()
    # B_0 * |0> = |0>
    assert dirac_notation.multiple_operations(B0 * Ket0) == Ket0
    # B_0 * |1> = 0
    assert dirac_notation.multiple_operations(ket_0 * bra_0 * ket_1) == 0
    # B_1 * |0> = 0
    assert dirac_notation.multiple_operations(b_1 * ket_0) == 0
    # B_1 * |1> = |0>
    assert dirac_notation.multiple_operations(b_1 * ket_1) == Ket(0)
    # B_2 * |0> = |1>
    assert dirac_notation.multiple_operations(b_2 * ket_0) == Ket(1)
    # B_2 * |1> = 0
    assert dirac_notation.multiple_operations(b_2 * ket_1) == 0
    # B_3 * |0> = 0
    assert dirac_notation.multiple_operations(b_3 * ket_0) == 0
    # B_3 * |1> = |1>
    assert dirac_notation.multiple_operations(b_3 * ket_1) == Ket(1)
    # B_3 * |1> = |1>
    assert dirac_notation.multiple_operations(b_3 * ket_1) == Ket(1)


def test_single_qubit_gate_reduce():
    dirac_notation = DiracNotation()
    # X * |0> = |1>
    assert dirac_notation.multiple_operations(X * ket_0) == Ket(1)
    # X * |1> = |0>
    assert dirac_notation.multiple_operations(X * ket_1) == Ket(0)
    # X * |0> = |1>
    assert dirac_notation.multiple_operations(X * ket_0) == Ket(1)
    # X * |1> = |0>
    assert dirac_notation.multiple_operations(X * ket_1) == Ket(0)
    # X * I * |1> = |1>
    assert dirac_notation.multiple_operations((X * I_2) * ket_0) == Ket(1)
    # H * |0> = 1/sqrt(2)|0> + 1/sqrt(2)|1>
    assert dirac_notation.multiple_operations(H * ket_0) == 1 / sqrt(2) * Ket(0) + 1 / sqrt(
        2
    ) * Ket(1)
    # H * |1> = 1/sqrt(2)|0> - 1/sqrt(2)|1>
    assert dirac_notation.multiple_operations(H * ket_1) == 1 / sqrt(2) * Ket(0) - 1 / sqrt(
        2
    ) * Ket(1)
    # B_0 x B_0 * |00> = |00>
    assert dirac_notation.multiple_operations(ket_0 * bra_0 * ket_0) == Ket(0)


def test_two_qubit_gate_reduce():
    dirac_notation = DiracNotation()
    # CX * |00> = |00>
    assert dirac_notation.multiple_operations(CX * TensorProduct(ket_0, ket_0)) == TensorProduct(
        ket_0, ket_0
    )
    # CX * |01> = |01>
    assert dirac_notation.multiple_operations(CX * TensorProduct(ket_0, ket_1)) == TensorProduct(
        ket_0, ket_1
    )
    # CX * |10> = |11>
    assert dirac_notation.multiple_operations(CX * TensorProduct(ket_1, ket_0)) == TensorProduct(
        ket_1, ket_1
    )
    # CX * |11> = |10>
    assert dirac_notation.multiple_operations(CX * TensorProduct(ket_1, ket_1)) == TensorProduct(
        ket_1, ket_0
    )


def test_circuit_reduce():
    dirac_notation = DiracNotation()
    assert dirac_notation.multiple_operations(I_2 * X * X * Ket0) == Ket0

    assert dirac_notation.multiple_operations(
        CX * TensorProduct(X, I_2) * TensorProduct(ket_0, ket_0)
    ) == TensorProduct(Ket(1), Ket(1))

    assert dirac_notation.multiple_operations(
        TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0)
    ) == sqrt(2) / 2 * TensorProduct(Ket(0), Ket(0)) + sqrt(2) / 2 * TensorProduct(Ket(1), Ket(0))
    assert dirac_notation.multiple_operations(
        CX * TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0)
    ) == sqrt(2) / 2 * TensorProduct(Ket(0), Ket(0)) + sqrt(2) / 2 * TensorProduct(Ket(1), Ket(1))
