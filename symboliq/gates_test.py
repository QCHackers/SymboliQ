import symboliq
import sympy
from sympy.physics.quantum import qapply
from sympy.physics.quantum.qubit import Qubit


def test_rx_gate() -> None:
    psi = Qubit('00')
    theta = sympy.Symbol('θ')
    assert str(qapply(symboliq.gates.rx(0, theta) * psi)) == "-1.0*I*sin(θ/2)*|01> + cos(θ/2)*|00>"


def test_ry_gate() -> None:
    psi = Qubit('00')
    theta = sympy.Symbol('θ')
    assert str(qapply(symboliq.gates.ry(0, theta) * psi)) == "sin(θ/2)*|01> + cos(θ/2)*|00>"


def test_rz_gate() -> None:
    psi = Qubit('00')
    theta = sympy.Symbol('θ')
    assert str(qapply(symboliq.gates.ry(1, theta) * psi)) == "sin(θ/2)*|10> + cos(θ/2)*|00>"
