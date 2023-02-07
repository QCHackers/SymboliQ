from typing import Any, List

import sympy
from sympy.core.numbers import Half
from sympy.physics.quantum import (
    Bra,
    Dagger,
    InnerProduct,
    Ket,
    OuterProduct,
    TensorProduct,
    tensor_product_simp,
)
from sympy.physics.quantum.gate import HadamardGate, IdentityGate, XGate, YGate, ZGate
from sympy.physics.quantum.qubit import Qubit

ket_0 = Ket(0)
bra_0 = Dagger(ket_0)
ket_1 = Ket(1)
bra_1 = Dagger(ket_1)
b_0 = ket_0 * bra_0
b_1 = ket_0 * bra_1
b_2 = ket_1 * bra_0
b_3 = ket_1 * bra_1
B_0 = sympy.Symbol("B_{0}")
B_1 = sympy.Symbol("B_{1}")
B_2 = sympy.Symbol("B_{2}")
B_3 = sympy.Symbol("B_{3}")
x = b_1 + b_2
y = -1j * b_1 + 1j * b_2
z = b_0 - b_3
h = (
    1 / sympy.sqrt(2) * b_0
    + 1 / sympy.sqrt(2) * b_1
    + 1 / sympy.sqrt(2) * b_2
    + (-1 / sympy.sqrt(2)) * b_3
)
i = b_0 + b_3
cx = TensorProduct(b_0, i) + TensorProduct(b_3, x)


def qapply(expr: sympy.Expr) -> sympy.Expr:
    return DiracNotation(expr).operate_reduce()


def get_simp_steps(expr: sympy.Expr) -> str:
    return DiracNotation(expr).get_steps()


class DiracNotation:
    steps: List[sympy.Expr] = []
    base_symbols = [B_0, B_1, B_2, B_3]

    def __init__(self, expr: sympy.Expr):
        self._expr = expr
        self._num_qubits = _count_number_of_tensor_products(expr) + 1

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return sympy.srepr(self._expr)

    def _get_steps_as_list(self) -> List[sympy.Expr]:
        self.steps = []
        self.steps.append(self._expr)
        self.operate_reduce()
        expr_list = []
        for i in self.steps:
            expr_list.append(i)

        return expr_list

    def get_steps(self) -> str:
        """Returns the steps the program went through to simplify
        a given expression
            Returns:
                    The steps as a string
        """
        expr_list = self._get_steps_as_list()
        plain_str: str = ""
        for k, l in enumerate(expr_list):
            plain_str = plain_str + f"({k}) {l}\n"
        return plain_str

    def get_steps_latex(self) -> str:
        """Returns the steps the program went through to simplify
        a given expression in latex representation that is capable of
        being printed in for example
            >> from IPython.display import Math
            >> dirac_notation = DiracNotation(x * ket_0)
            >> Math(dirac_notation.get_steps_latex())


            Returns:
                    The steps represented in latex notation as a string
        """
        expr_list = self._get_steps_as_list()
        latex_str: str = ""
        for k, l in enumerate(expr_list):
            latex_str = latex_str + rf"({k}) \quad {sympy.latex(l)} \\"
        return latex_str

    def _mul_reduce(self, arg: sympy.Expr, add_step: bool) -> sympy.Expr:
        brakets = []
        constants = []
        new_term = sympy.Integer(1)
        args = arg.args
        for term in args:
            if isinstance(term, (OuterProduct, Ket, Bra, Qubit)):
                brakets.append(term)
            elif isinstance(term, InnerProduct):
                if add_step:
                    self.steps.append(arg)
                new_term = new_term * self._gate_reduce(term, add_step) * args[1]
                if add_step:
                    self.steps.append(new_term)
                return new_term
            elif isinstance(term, (Half, sympy.Pow, sympy.Rational)):
                constants.append(term)

        calc = self._gate_reduce(brakets[0].args[0] * (brakets[0].args[1] * brakets[1]), add_step)
        for i in constants:
            calc = calc * i
        assert isinstance(calc, sympy.Expr)
        return calc

    def _add_reduce(self, arg: sympy.Expr, add_step: bool) -> sympy.Expr:
        added_term: sympy.Expr = sympy.Integer(0)
        for i in arg.args:
            assert isinstance(i, sympy.Expr)
            added_term = added_term + self._gate_reduce(i, add_step)
        return added_term

    def _tensor_reduce(self, arg: sympy.Expr, add_step: bool) -> sympy.Expr:
        final_state_vec = sympy.Integer(0)
        mes = tensor_product_simp(arg.expand(tensorproduct=True))
        mes = _factor_tensor(mes, self._num_qubits)
        for i in mes:
            tansors = []
            for j in i:
                tansors.append(self._gate_reduce(j, add_step))
            final_state_vec = final_state_vec + TensorProduct(*tansors)
        assert isinstance(final_state_vec, sympy.Expr)
        return final_state_vec

    def _gate_reduce(self, arg: sympy.Expr, add_step: bool) -> sympy.Expr:
        if isinstance(arg, InnerProduct):
            return _base_reduce(arg)

        elif any(item in self.base_symbols for item in list(arg.args)):
            arg = arg.subs([(B_0, b_0), (B_1, b_1), (B_2, b_2)])
            return self._gate_reduce(arg, add_step)

        elif _check_pauli_hadamard(arg):

            arg = _sub_pauli_hadamard(arg)

            # Distributivity of matrix multiplication over addition
            arg = arg.expand()
            self.steps.append(arg)
            arg = self._gate_reduce(arg, add_step=False)
            self.steps.append(arg)
            return arg

        elif (
            isinstance(arg, sympy.Mul)
            and _count_number_of_kets_and_bras(arg) == 2
            and any(isinstance(item, Ket) for item in list(arg.args))
        ):
            return self._mul_reduce(arg, add_step)
        elif arg.func == sympy.Add:
            return self._add_reduce(arg, add_step)
        return self._tensor_reduce(arg, add_step)

    def operate_reduce(self) -> sympy.Expr:
        """Iterates through an expression and simplifies incrementally while
        keeping track of the steps that are taken to simplify it

            Returns:
                The simplified expression
        """
        assert isinstance(self._expr, sympy.Mul)
        rev_args = self._expr.args[::-1]
        state = rev_args[0]
        for i in range(1, len(rev_args)):
            rev_args_by_index = rev_args[i]
            if isinstance(rev_args_by_index, sympy.Pow):
                exp = rev_args_by_index.exp
                base = rev_args_by_index.base
                for _ in range(exp):
                    state = self._gate_reduce(base * state, True)

            else:
                state = self._gate_reduce(rev_args_by_index * state, True)
        return state


def _check_pauli_hadamard(expr: sympy.Expr) -> bool:
    """Tells you whether an expression contains at least one Pauli or Hadamard gate
    Args:
        expr: The expression
    Returns:
        True if expression contains at least one  Pauli or Hadamard gate
         False otherwise
    """
    pauli_and_hadamard_dirac = [i, x, z, h]
    pauli_and_hadamard_gates = (IdentityGate, XGate, YGate, ZGate, HadamardGate)
    return any(item in pauli_and_hadamard_dirac for item in list(expr.args)) or any(
        isinstance(item, pauli_and_hadamard_gates) for item in list(expr.args)
    )


def _sub_pauli_hadamard(arg: sympy.Expr) -> sympy.Expr:
    """Looks through an expression and replaced pauli and hadamard sympy gates with their
    dirac notation equivalents
        Args:
            Any sympy expression
        Returns:
            Another sympy expression
    """
    for a in arg.args:
        if isinstance(a, IdentityGate):
            arg = arg.subs(a, i)
        elif isinstance(a, XGate):
            arg = arg.subs(a, x)
        elif isinstance(a, YGate):
            arg = arg.subs(a, y)
        elif isinstance(a, ZGate):
            arg = arg.subs(a, z)
        elif isinstance(a, HadamardGate):
            arg = arg.subs(a, h)
    return arg


def _count_number_of_kets_and_bras(arg: sympy.Mul) -> int:
    args = arg.args
    count = 0
    for i in args:
        if isinstance(i, (OuterProduct, InnerProduct, Ket, Bra, Qubit)):
            count = count + 1
    return count


def _count_number_of_tensor_products(arg: sympy.Expr) -> int:
    args = arg.args
    count = 0
    for i in args:
        if isinstance(i, TensorProduct):
            count = count + 1
    return count


def _base_reduce(inner_product: sympy.physics.quantum.InnerProduct) -> sympy.Integer:
    """Expressions like Bra(0) * Ket(0).doit()) evaluate to <0|0> when they
    should be evaluating to 1. This function bypasses that limitation
               Args:
                   inner_product: has to be a <b|k> where b and k are either 1 or 0
               Returns:
                   The evaluated inner product
    """
    same_bras_and_kets = [Bra(0) * Ket(0), Bra(0) * Qubit(0), Bra(1) * Ket(1), Bra(1) * Qubit(1)]
    different_bras_and_kets = [
        Bra(0) * Ket(1),
        Bra(1) * Ket(0),
        Bra(0) * Qubit(1),
        Bra(1) * Qubit(0),
    ]
    assert inner_product in same_bras_and_kets + different_bras_and_kets

    if inner_product in same_bras_and_kets:
        return sympy.Integer(1)
    else:
        return sympy.Integer(0)


def _factor_tensor(expr: Any, state_space: int) -> Any:
    """For example, given sqrt(2)*(|0><0|*|0>)x((|0><0| + |1><1|)*|0>)/2,
    returns [sqrt(2)*(|0><0|*|0>), ((|0><0| + |1><1|)*|0>)/2]
        Args: The expression containing the tensors to factor
        Return: A list of the tensored factors


    """
    tensors = []
    constant = []
    additions = []

    if expr.func == sympy.Add:
        for ar in expr.args:
            additions.append(_factor_tensor(ar, state_space))
        return additions

    if expr.func == sympy.Mul:
        for ar in expr.args:
            if ar.func == TensorProduct:
                tensors = list(ar.args)
            else:
                constant.append(ar)
        for i in constant:
            tensors[0] = tensors[0] * i
    if expr.func == TensorProduct:
        for ar in expr.args:
            tensors.append(ar)
    return tensors
