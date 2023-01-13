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
assert my_simpify(bra_0 * ket_0) == 1
assert my_simpify(bra_1 * ket_1) == 1
assert my_simpify(bra_0 * ket_1) == 0
assert my_simpify(bra_1 * ket_0) == 0

x_0 = b_0 * ket_0

def simplify_associative_mul(c):
    d = Mul(OuterProduct(Ket(Integer(0)),Bra(Integer(0))), Ket(Integer(0)))
    print(d.args)

    for arg in preorder_traversal(c):
        print(srepr(arg))
    #print(srepr(c))
def my_simplify2(c):
    mul_arg = None
    inner_product_arg = None
    for arg in preorder_traversal(c):
        #print(arg)
        if isinstance(arg, sympy.Mul):
            mul_arg = arg
            #simplify_associative_mul(arg)

            if arg == Mul(OuterProduct(Ket(Integer(0)),Bra(Integer(0))), Ket(Integer(0))):
                in_prod_arg = InnerProduct(Bra(Integer(0)),Ket(Integer(0)))
                new_arg = Mul(in_prod_arg, Ket(Integer(0)))
                result = my_simpify(in_prod_arg)
                return new_arg.subs(in_prod_arg, result)

        if isinstance(arg, sympy.physics.quantum.InnerProduct):
            #print("sulfsdui")
            inner_product_arg = arg
            result = my_simpify(inner_product_arg)
            break

    multiplied = mul_arg.subs(inner_product_arg, result)
    return multiplied


c = ket_0 * bra_0 * ket_0
print(my_simplify2(c))

