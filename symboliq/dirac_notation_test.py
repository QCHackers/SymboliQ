from sympy import sqrt
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.gate import HadamardGate, IdentityGate, XGate, YGate, ZGate
from sympy.physics.quantum.qubit import Qubit

import symboliq
from symboliq.dirac_notation import (
    B_0,
    B_1,
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


def test_str() -> None:
    assert str(DiracNotation(ket_0)) == "|0>"


def test_repr() -> None:
    assert repr(DiracNotation(B_0 * ket_0)) == "Mul(Symbol('B_{0}'), Ket(Integer(0)))"


def test_qapply() -> None:
    assert symboliq.qapply(IdentityGate(0) * Qubit("0")) == ket_0
    assert symboliq.qapply(XGate(0) * Qubit("0")) == ket_1
    assert symboliq.qapply(YGate(0) * Qubit("1")) == ket_0
    assert symboliq.qapply(ZGate(0) * Qubit("1")) == -ket_1
    assert (
        symboliq.qapply(HadamardGate(0) * Qubit("0")) == 1 / sqrt(2) * ket_0 + 1 / sqrt(2) * ket_1
    )


def test_get_simp_steps() -> None:
    assert (
        symboliq.get_simp_steps(XGate(0) * Qubit("0"))
        == """(0) X(0)*|0>
(1) |0><1|*|0> + |1><0|*|0>
(2) |1>
"""
    )


def test_base_reduce() -> None:
    # B_0 * |0> = |0>
    assert DiracNotation(B_0 * ket_0).operate_reduce() == ket_0
    # B_0 * |1> = 0
    assert DiracNotation(ket_0 * bra_0 * ket_1).operate_reduce() == 0
    # B_1 * |0> = 0
    assert DiracNotation(B_1 * ket_0).operate_reduce() == 0
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


def test_get_steps() -> None:
    assert (
        DiracNotation(x * ket_1).get_steps()
        == """(0) (|0><1| + |1><0|)*|1>
(1) |0><1|*|1> + |1><0|*|1>
(2) |0>
"""
    )


def test_get_steps_latex() -> None:
    # B_0 * |0> = |0>
    assert (
        DiracNotation(B_0 * ket_0).get_steps_latex()
        == r"(0) \quad B_{0} {\left|0\right\rangle } \\(1) \quad \left\langle 0 \right. "
        r"{\left|0\right\rangle } {\left|0\right\rangle } \\(2) \quad {\left|0\right\rangle } \\"
    )
