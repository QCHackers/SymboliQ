import sympy.physics.quantum
from sympy import *
from sympy.physics.quantum import qapply, Ket, Bra, Dagger, OuterProduct, InnerProduct


def my_simpify(expr):
    if isinstance(expr, sympy.physics.quantum.InnerProduct):
        if Dagger(expr.ket) == expr.bra:
            return 1
        else:
            return 0


ket_0 = Ket(0)
bra_0 = Dagger(ket_0)
ket_1 = Ket(1)
bra_1 = Dagger(ket_1)
b_0 = ket_0 * bra_0
b_1 = ket_0 * bra_1
b_2 = ket_1 * bra_0
b_3 = ket_1 * bra_1
X = b_1 + b_2
H = 1 / sqrt(2) * b_0 + 1 / sqrt(2) * b_1 + 1 / sqrt(2) * b_2 + (-1 / sqrt(2)) * b_3

assert my_simpify(bra_0 * ket_0) == 1
assert my_simpify(bra_1 * ket_1) == 1
assert my_simpify(bra_0 * ket_1) == 0
assert my_simpify(bra_1 * ket_0) == 0

x_0 = b_0 * ket_0


def my_simplify2(c):
    for arg in preorder_traversal(c):
        if arg.has(Mul) and not arg.has(Add):
            # Associative property
            args = arg.args
            ket = args[1]
            outer_product = args[0]

            constants = []
            brakets = []

            res = 1

            for term in args:
                if isinstance(term, (OuterProduct, Ket)):
                    brakets.append(term)
                else:
                    constants.append(term)


            new_brakets = brakets[0].args[0] * (brakets[0].args[1] * brakets[1])

            res = Mul(new_brakets * sympy.prod(constants))
            res = res.replace(bra_1 * ket_1, 1)
            res = res.replace(bra_0 * ket_0, 1)
            res = res.replace(bra_0 * ket_1, 0)
            res = res.replace(bra_1 * ket_0, 0)
            return res

        elif arg.has(Add):
            # Distributive property
            args = arg.args
            ket = args[1]
            add = args[0]
            res = 0
            for term in add.args:
                res = res + my_simplify2(term * ket)
            return res

            # b = Mul(Add(add.args[0] * ket, add.args[1] * ket))
            # return my_simplify2(b.args[0]) + my_simplify2(b.args[1])


#
# B_0 * |0> = |0>
assert my_simplify2(ket_0 * bra_0 * ket_0) == Ket(0)
# B_0 * |1> = |1>
assert my_simplify2(ket_0 * bra_0 * ket_1) == 0
# B_1 * |0> = 0
assert my_simplify2(b_1 * ket_0) == 0
# B_1 * |1> = |0>
assert my_simplify2(b_1 * ket_1) == Ket(0)
# B_2 * |0> = |0>
assert my_simplify2(b_2 * ket_0) == Ket(1)
# B_2 * |1> = 0
assert my_simplify2(b_2 * ket_1) == 0
# B_3 * |0> = 0
assert my_simplify2(b_3 * ket_0) == 0
# B_3 * |1> = 0
assert my_simplify2(b_3 * ket_1) == Ket(1)
# B_3 * |1> = 0
assert my_simplify2(b_3 * ket_1) == Ket(1)
assert my_simplify2(X * ket_0) == Ket(1)
assert my_simplify2(X * ket_1) == Ket(0)
assert my_simplify2(H * ket_0) == 1/sqrt(2) * Ket(0) + 1/sqrt(2) * Ket(1)
