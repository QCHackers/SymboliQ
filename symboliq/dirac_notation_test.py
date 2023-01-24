from sympy import sqrt
from sympy.physics.quantum import TensorProduct

from symboliq.dirac_notation import (
    B0,
    B1,
    DiracNotation,
    b_1,
    b_2,
    b_3,
    bra_0,
    cx,
    h,
    i,
    ket_0,
    ket_1,
    x,
)


def test_base_reduce() -> None:
    # B_0 * |0> = |0>
    assert DiracNotation(B0 * ket_0).operate_reduce() == ket_0
    # B_0 * |1> = 0
    assert DiracNotation(ket_0 * bra_0 * ket_1).operate_reduce() == 0
    # B_1 * |0> = 0
    assert DiracNotation(B1 * ket_0).operate_reduce() == 0
    # B_1 * |1> = |0>
    assert DiracNotation(b_1 * ket_1).operate_reduce() == ket_0
    # B_2 * |0> = |1>
    assert DiracNotation(b_2 * ket_0).operate_reduce() == ket_1
    # B_2 * |1> = 0
    assert DiracNotation(b_2 * ket_1).operate_reduce() == 0
    # B_3 * |0> = 0
    assert DiracNotation(b_3 * ket_0).operate_reduce() == 0
    # B_3 * |1> = |1>
    assert DiracNotation(b_3 * ket_1).operate_reduce() == ket_1
    # B_3 * |1> = |1>
    assert DiracNotation(b_3 * ket_1).operate_reduce() == ket_1


def test_single_qubit_gate_reduce() -> None:
    # X * |0> = |1>
    assert DiracNotation(x * ket_0).operate_reduce() == ket_1
    # X * |1> = |0>
    assert DiracNotation(x * ket_1).operate_reduce() == ket_0
    # X * |0> = |1>
    assert DiracNotation(x * ket_0).operate_reduce() == ket_1
    # X * |1> = |0>
    assert DiracNotation(x * ket_1).operate_reduce() == ket_0
    # X * I * |1> = |1>
    assert DiracNotation((x * i) * ket_0).operate_reduce() == ket_1
    # H * |0> = 1/sqrt(2)|0> + 1/sqrt(2)|1>
    assert DiracNotation(h * ket_0).operate_reduce() == 1 / sqrt(2) * ket_0 + 1 / sqrt(2) * ket_1
    # H * |1> = 1/sqrt(2)|0> - 1/sqrt(2)|1>
    assert DiracNotation(h * ket_1).operate_reduce() == 1 / sqrt(2) * ket_0 - 1 / sqrt(2) * ket_1
    # B_0 x B_0 * |00> = |00>
    assert DiracNotation(ket_0 * bra_0 * ket_0).operate_reduce() == ket_0


def test_two_qubit_gate_reduce() -> None:
    # CX * |00> = |00>
    assert DiracNotation(cx * TensorProduct(ket_0, ket_0)).operate_reduce() == TensorProduct(
        ket_0, ket_0
    )
    # CX * |01> = |01>
    assert DiracNotation(cx * TensorProduct(ket_0, ket_1)).operate_reduce() == TensorProduct(
        ket_0, ket_1
    )
    # CX * |10> = |11>
    assert DiracNotation(cx * TensorProduct(ket_1, ket_0)).operate_reduce() == TensorProduct(
        ket_1, ket_1
    )
    # CX * |11> = |10>
    assert DiracNotation(cx * TensorProduct(ket_1, ket_1)).operate_reduce() == TensorProduct(
        ket_1, ket_0
    )


def test_circuit_reduce() -> None:
    assert DiracNotation(i * x * x * ket_0).operate_reduce() == ket_0

    assert DiracNotation(
        cx * TensorProduct(x, i) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == TensorProduct(ket_1, ket_1)

    assert DiracNotation(
        TensorProduct(h, i) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == sqrt(2) / 2 * TensorProduct(ket_0, ket_0) + sqrt(2) / 2 * TensorProduct(
        ket_1, ket_0
    )
    assert DiracNotation(
        cx * TensorProduct(h, i) * TensorProduct(ket_0, ket_0)
    ).operate_reduce() == sqrt(2) / 2 * TensorProduct(ket_0, ket_0) + sqrt(2) / 2 * TensorProduct(
        ket_1, ket_1
    )


def test_get_steps_latex() -> None:
    # B_0 * |0> = |0>
    assert (
        DiracNotation(B0 * ket_0).get_steps_latex()
        == r"(0) \quad B_{0} {\left|0\right\rangle } \\(1) \quad \left\langle 0 \right. "
        r"{\left|0\right\rangle } {\left|0\right\rangle } \\(2) \quad {\left|0\right\rangle } \\"
    )
