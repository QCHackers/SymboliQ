from sympy import sqrt
from sympy.physics.quantum import Ket, TensorProduct

from symboliq.dirac_notation import (
    B0,
    B1,
    CX,
    I_2,
    DiracNotation,
    H,
    Ket0,
    X,
    b_1,
    b_2,
    b_3,
    bra_0,
    ket_0,
    ket_1,
)


def test_base_reduce() -> None:
    # B_0 * |0> = |0>
    assert DiracNotation(B0 * Ket0).operate_reduce() == Ket0
    # B_0 * |1> = 0
    assert DiracNotation(ket_0 * bra_0 * ket_1).operate_reduce() == 0
    # B_1 * |0> = 0
    assert DiracNotation(B1 * ket_0).operate_reduce() == 0
    # B_1 * |1> = |0>
    assert DiracNotation(b_1 * ket_1).operate_reduce() == Ket(0)
    # B_2 * |0> = |1>
    assert DiracNotation(b_2 * ket_0).operate_reduce() == Ket(1)
    # B_2 * |1> = 0
    assert DiracNotation(b_2 * ket_1).operate_reduce() == 0
    # B_3 * |0> = 0
    assert DiracNotation(b_3 * ket_0).operate_reduce() == 0
    # B_3 * |1> = |1>
    assert DiracNotation(b_3 * ket_1).operate_reduce() == Ket(1)
    # B_3 * |1> = |1>
    assert DiracNotation(b_3 * ket_1).operate_reduce() == Ket(1)


def test_single_qubit_gate_reduce() -> None:
    # X * |0> = |1>
    assert DiracNotation(X * ket_0).operate_reduce() == Ket(1)
    # X * |1> = |0>
    assert DiracNotation(X * ket_1).operate_reduce() == Ket(0)
    # X * |0> = |1>
    assert DiracNotation(X * ket_0).operate_reduce() == Ket(1)
    # X * |1> = |0>
    assert DiracNotation(X * ket_1).operate_reduce() == Ket(0)
    # X * I * |1> = |1>
    assert DiracNotation((X * I_2) * ket_0).operate_reduce() == Ket(1)
    # H * |0> = 1/sqrt(2)|0> + 1/sqrt(2)|1>
    assert DiracNotation(H * ket_0).operate_reduce() == 1 / sqrt(2) * Ket(0) + 1 / sqrt(2) * Ket(1)
    # H * |1> = 1/sqrt(2)|0> - 1/sqrt(2)|1>
    assert DiracNotation(H * ket_1).operate_reduce() == 1 / sqrt(2) * Ket(0) - 1 / sqrt(2) * Ket(1)
    # B_0 x B_0 * |00> = |00>
    assert DiracNotation(ket_0 * bra_0 * ket_0).operate_reduce() == Ket(0)


def test_two_qubit_gate_reduce() -> None:
    # CX * |00> = |00>
    assert DiracNotation(CX * TensorProduct(ket_0, ket_0)).operate_reduce() == TensorProduct(
        ket_0, ket_0
    )
    # CX * |01> = |01>
    assert DiracNotation(CX * TensorProduct(ket_0, ket_1)).operate_reduce() == TensorProduct(
        ket_0, ket_1
    )
    # CX * |10> = |11>
    assert DiracNotation(CX * TensorProduct(ket_1, ket_0)).operate_reduce() == TensorProduct(
        ket_1, ket_1
    )
    # CX * |11> = |10>
    assert DiracNotation(CX * TensorProduct(ket_1, ket_1)).operate_reduce() == TensorProduct(
        ket_1, ket_0
    )


def test_circuit_reduce() -> None:
    assert DiracNotation(I_2 * X * X * Ket0).operate_reduce() == Ket0

    assert DiracNotation(
        CX * TensorProduct(X, I_2) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == TensorProduct(Ket(1), Ket(1))

    assert DiracNotation(
        TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == sqrt(2) / 2 * TensorProduct(Ket(0), Ket(0)) + sqrt(2) / 2 * TensorProduct(
        Ket(1), Ket(0)
    )
    assert DiracNotation(
        CX * TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == sqrt(2) / 2 * TensorProduct(Ket(0), Ket(0)) + sqrt(2) / 2 * TensorProduct(
        Ket(1), Ket(1)
    )


def test_get_steps_latex() -> None:
    # B_0 * |0> = |0>
    assert (
        DiracNotation(B0 * Ket0).get_steps_latex()
        == r"(0) \quad B_{0} {\left|0\right\rangle } \\(1) \quad \left\langle 0 \right. "
        r"{\left|0\right\rangle } {\left|0\right\rangle } \\(2) \quad {\left|0\right\rangle } \\"
    )
