import sympy.physics.quantum
from sympy import *
from sympy.physics.quantum import qapply, Ket, Bra, Dagger, OuterProduct, InnerProduct, TensorProduct, \
    tensor_product_simp


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
I_2 = b_0 + b_3
CX = TensorProduct(b_0, I_2) + TensorProduct(b_3, X)

assert my_simpify(bra_0 * ket_0) == 1
assert my_simpify(bra_1 * ket_1) == 1
assert my_simpify(bra_0 * ket_1) == 0
assert my_simpify(bra_1 * ket_0) == 0

x_0 = b_0 * ket_0


def multiple_operations(arg, state_space):
    state = arg.args[len(arg.args) - 1]
    for i in reversed(range(len(arg.args) - 1)):
        state = my_simplify2(arg.args[i] * state, state_space=state_space)
    return state


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def check_integer_pows(pows):
    if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):
        return False
    return True


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


def my_simplify2(c, multiple_op: bool = False, state_space=1):
    if multiple_op:
        return multiple_operations(c, state_space)
    for arg in preorder_traversal(c):
        if arg.has(Pow) and check_integer_pows(list(arg.atoms(Pow))):
            pows = list(arg.atoms(Pow))
            base = pows[0].base
            exponent = pows[0].exp
            state = arg.args[len(arg.args) - 1]

            for i in range(exponent):
                state = my_simplify2(base * state)
            return state
        if arg.has(Mul) and not arg.has(Add) and not arg.has(TensorProduct):
            # Associative property
            # print(arg)
            args = arg.args
            constants = []
            brakets = []
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

        elif arg.has(Add) and not arg.has(TensorProduct):
            # Distributive property
            args = arg.args
            ket = args[1]
            add = args[0]
            res = 0
            for term in add.args:
                res = res + my_simplify2(term * ket)
            return res
        elif arg.has(TensorProduct) and arg.has(Add):
            last_list1 = []
            final_state_vec = 0
            res = tensor_product_simp(arg.expand(tensorproduct=True))
            res = factor_tensor(res, 2)


            for i in res:
                tansors = []
                for j in i:
                    tansors.append(my_simplify2(j))
                final_state_vec = final_state_vec + TensorProduct(*tansors)
            return final_state_vec
            #     print(type(ar))
            #     for sub in ar.args:
            #         if isinstance(sub, (Rational, Pow)):
            #             pass
            #         else:
            #             #print(f"Sub {sub}")
            #             #  print(f"My simpify sub {my_simplify2(sub)}")
            #             last_list1.append(my_simplify2(sub))
            # last_list1 = chunks(last_list1, state_space)
            # for i in last_list1:
            #     #print(f"The result {i}")
            #     final_state_vec = final_state_vec + TensorProduct(*i)
        else:
            print(f"Arg {arg}")
            return arg


#B_0 * |0> = |0>
assert my_simplify2(ket_0 * bra_0 * ket_0) == Ket(0)
# B_0 * |1> = 0
assert my_simplify2(ket_0 * bra_0 * ket_1) == 0
# B_1 * |0> = 0
assert my_simplify2(b_1 * ket_0) == 0
# B_1 * |1> = |0>
assert my_simplify2(b_1 * ket_1) == Ket(0)
# B_2 * |0> = |1>
assert my_simplify2(b_2 * ket_0) == Ket(1)
# B_2 * |1> = 0
assert my_simplify2(b_2 * ket_1) == 0
# B_3 * |0> = 0
assert my_simplify2(b_3 * ket_0) == 0
# B_3 * |1> = |1>
assert my_simplify2(b_3 * ket_1) == Ket(1)
# B_3 * |1> = |1>
assert my_simplify2(b_3 * ket_1) == Ket(1)
# X * |0> = |1>
assert my_simplify2(X * ket_0) == Ket(1)
# X * |1> = |0>
assert my_simplify2(X * ket_1) == Ket(0)
# X * |0> = |1>
assert my_simplify2(X * ket_0) == Ket(1)
# X * |1> = |0>
assert my_simplify2(X * ket_1) == Ket(0)
# X * I * |1> = |1>
assert my_simplify2((X * I_2) * ket_0, True) == Ket(1)
# H * |0> = 1/sqrt(2)|0> + 1/sqrt(2)|1>
assert my_simplify2(H * ket_0) == 1 / sqrt(2) * Ket(0) + 1 / sqrt(2) * Ket(1)
# H * |1> = 1/sqrt(2)|0> - 1/sqrt(2)|1>
assert my_simplify2(H * ket_1) == 1 / sqrt(2) * Ket(0) - 1 / sqrt(2) * Ket(1)
# B_0 x B_0 * |00> = |00>
assert my_simplify2(ket_0 * bra_0 * ket_0) == Ket(0)
# CX * |00> = |00>
assert my_simplify2(CX * TensorProduct(ket_0, ket_0), state_space=2) == TensorProduct(ket_0, ket_0)
# CX * |01> = |01>
assert my_simplify2(CX * TensorProduct(ket_0, ket_1), state_space=2) == TensorProduct(ket_0, ket_1)
# CX * |10> = |11>
assert my_simplify2(CX * TensorProduct(ket_1, ket_0), state_space=2) == TensorProduct(ket_1, ket_1)
# CX * |11> = |10>
assert my_simplify2(CX * TensorProduct(ket_1, ket_1), state_space=2) == TensorProduct(ket_1, ket_0)
assert my_simplify2(X * X * ket_0, multiple_op=True) == Ket(0)
assert my_simplify2(CX * TensorProduct(X, I_2) * TensorProduct(ket_0, ket_0), state_space=2, multiple_op=True) == TensorProduct(Ket(1), Ket(1))
assert my_simplify2(TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0), state_space=2) == sqrt(2)/2 * TensorProduct(
    Ket(0), Ket(0)) + sqrt(2)/2 * TensorProduct(Ket(1), Ket(0))



# print(TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0))
res = my_simplify2(TensorProduct(H, I_2) * TensorProduct(ket_0, ket_0), state_space=2)
# print(res)

# print(my_simplify2(CX * TensorProduct(ket_0, ket_0), state_space=2) == TensorProduct(ket_0, ket_0))
# print(ras)

