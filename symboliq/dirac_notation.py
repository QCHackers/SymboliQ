import sympy.physics.quantum
from sympy import *
from sympy.physics.quantum import (
    qapply,
    Ket,
    Bra,
    Dagger,
    OuterProduct,
    InnerProduct,
    TensorProduct,
    tensor_product_simp,
)

ket_0 = Ket(0)
ket_0 = Ket(0)
Ket0 = ket_0
bra_0 = Dagger(ket_0)
ket_1 = Ket(1)
bra_1 = Dagger(ket_1)
b_0 = ket_0 * bra_0
b_1 = ket_0 * bra_1
b_2 = ket_1 * bra_0
b_3 = ket_1 * bra_1
B0 = Symbol("B_{0}")
B1 = Symbol("B_{1}")
B2 = Symbol("B_{2}")
B3 = Symbol("B_{3}")
XSymbol = Symbol("X")
X = b_1 + b_2
H = 1 / sqrt(2) * b_0 + 1 / sqrt(2) * b_1 + 1 / sqrt(2) * b_2 + (-1 / sqrt(2)) * b_3
I_2 = b_0 + b_3
CX = TensorProduct(b_0, I_2) + TensorProduct(b_3, X)

bases_matrices = [B0, B1, B2, B3]
base_states = [ket_0, ket_1]
pauli_and_hadamard_gates = [X, I_2, H]


def count_number_of_kets_and_bras(arg):
    args = arg.args
    count = 0
    for i in args:
        if isinstance(i, (OuterProduct, InnerProduct, Ket, Bra)):
            count = count + 1
    return count


def factor_tensor(expr, state_space):
    """Given sqrt(2)*(|0><0|*|0>)x((|0><0| + |1><1|)*|0>)/2,
    returns

    [sqrt(2)*(|0><0|*|0>), ((|0><0| + |1><1|)*|0>)/2]

    """
    tensors = []
    constant = []
    additions = []

    if expr.func == Add:
        for ar in expr.args:
            additions.append(factor_tensor(ar, state_space))
        return additions

    if expr.func == Mul:
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


class DiracNotation:
    steps = []

    def __init__(self):
        pass

    def get_steps_latex(self, expr):
        self.steps = []
        self.steps.append(expr)
        self.my_simplify_3(expr)
        expr_list = []
        for i in self.steps:
            expr_list.append(latex(i))
        latex_str = ""
        for i, j in enumerate(expr_list):
            latex_str = latex_str + rf"({i}) \quad {j} \\"
        return latex_str

    def my_simplify_3(self, arg):
        if arg.func == InnerProduct:
            if arg == Bra(0) * Ket(0) or arg == Bra(1) * Ket(1):
                return 1
            elif arg == Bra(1) * Ket(0) or arg == Bra(0) * Ket(1):
                return 0
            else:
                raise ValueError("Unhandled")
        if any(item in bases_matrices for item in list(arg.args)):
            arg = arg.subs([(B0, b_0), (B1, b_1), (B2, b_2)])
            return self.my_simplify_3(arg)

        if any(item in pauli_and_hadamard_gates for item in list(arg.args)):
            arg = arg.subs([(XSymbol, X)])
            # Distributivity of matrix multiplication over addition
            return self.my_simplify_3(arg.expand())

        elif (
            arg.func == Mul
            and count_number_of_kets_and_bras(arg) == 2
            and any(item in base_states for item in list(arg.args))
        ):
            brakets = []
            constants = []
            new_term = 1
            args = arg.args
            for term in args:
                if isinstance(term, (OuterProduct, Ket, Bra)):
                    brakets.append(term)
                elif isinstance(term, InnerProduct):
                    self.steps.append(arg)
                    new_term = new_term * self.my_simplify_3(term) * args[1]
                    self.steps.append(new_term)
                    return new_term
                elif isinstance(term, (sympy.core.numbers.Half, Pow, Rational)):
                    constants.append(term)
            if arg.has(OuterProduct):
                calc = self.my_simplify_3(brakets[0].args[0] * (brakets[0].args[1] * brakets[1]))
                for i in constants:
                    calc = calc * i
                return calc
            else:
                return new_term
        elif arg.func == Add:
            added_term = 0
            for i in arg.args:
                added_term = added_term + self.my_simplify_3(i)
            return added_term
        elif arg.func == Mul and arg.has(TensorProduct):
            final_state_vec = 0
            mes = tensor_product_simp(arg.expand(tensorproduct=True))
            mes = factor_tensor(mes, 2)
            for i in mes:
                tansors = []
                for j in i:
                    tansors.append(self.my_simplify_3(j))
                final_state_vec = final_state_vec + TensorProduct(*tansors)
            return final_state_vec
        elif arg.func == TensorProduct:
            tensors = []
            for i in arg.args:
                tensors.append(self.my_simplify_3(i))

            return TensorProduct(*tensors)

    def multiple_operations(self, arg):
        rev_args = arg.args[::-1]
        state = rev_args[0]
        for i in range(1, len(rev_args)):
            rev_args_by_index = rev_args[i]
            if rev_args_by_index.func == Pow:
                exp = rev_args_by_index.exp
                base = rev_args_by_index.base
                for _ in range(exp):
                    state = self.my_simplify_3(base * state)
            else:
                state = self.my_simplify_3(rev_args_by_index * state)

        return state
