# pragma: no cover
from typing import List, Union

from sympy import Add, Basic, Mul, Pow
from sympy.physics.quantum import AntiCommutator, Commutator, OuterProduct
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.tensorproduct import (
    TensorProduct,
    tensor_product_simp,
    tensor_product_simp_Pow,
)


def _break_up_tensor_products(lst: List[Union[OuterProduct, TensorProduct]]) -> List[Basic]:
    new_list = []
    for i in lst:
        print(type(i))
        if isinstance(i, TensorProduct):
            new_list.append(list(i.args))
        flat_list = [item for sublist in new_list for item in sublist]
        flat_list.append(i)
    return flat_list


def tensor_product_simp_Mul_fork(e):  # type: ignore  # noqa: C901
    """Simplify a Mul with TensorProducts.

    Current the main use of this is to simplify a ``Mul`` of ``TensorProduct``s
    to a ``TensorProduct`` of ``Muls``. It currently only works for relatively
    simple cases where the initial ``Mul`` only has scalars and raw
    ``TensorProduct``s, not ``Add``, ``Pow``, ``Commutator``s of
    ``TensorProduct``s.

    Parameters
    ==========

    e : Expr
        A ``Mul`` of ``TensorProduct``s to be simplified.

    Returns
    =======

    e : Expr
        A ``TensorProduct`` of ``Mul``s.

    Examples
    ========

    This is an example of the type of simplification that this function
    performs::

        >>> from sympy.physics.quantum.tensorproduct import \
                    tensor_product_simp_Mul, TensorProduct
        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> C = Symbol('C',commutative=False)
        >>> D = Symbol('D',commutative=False)
        >>> e = TensorProduct(A,B)*TensorProduct(C,D)
        >>> e
        AxB*CxD
        >>> tensor_product_simp_Mul(e)
        (A*C)x(B*D)

    """
    # TODO: This won't work with Muls that have other composites of
    # TensorProducts, like an Add, Commutator, etc.
    # TODO: This only works for the equivalent of single Qbit gates.

    if not isinstance(e, Mul):
        return e
    c_part, nc_part = e.args_cnc()
    n_nc = len(nc_part)
    if n_nc == 0:
        return e
    elif n_nc == 1:
        if isinstance(nc_part[0], Pow):
            return Mul(*c_part) * tensor_product_simp_Pow(nc_part[0])
        return e
    elif e.has(TensorProduct):
        current = nc_part[0]
        if not isinstance(current, TensorProduct):
            if isinstance(current, Pow):
                if isinstance(current.base, TensorProduct):
                    current = tensor_product_simp_Pow(current)
            else:
                raise TypeError("TensorProduct expected, got: %r" % current)

        new_args = list(current.args)
        for next in nc_part[1:]:  # pylint: disable=redefined-builtin
            # TODO: check the hilbert spaces of next and current here.
            if isinstance(next, TensorProduct):
                for i in new_args:
                    if isinstance(i, TensorProduct):
                        new_args = _break_up_tensor_products(new_args)
                n_terms = len(new_args)
                if n_terms != len(next.args):
                    raise QuantumError(
                        "TensorProducts of different lengths: %r and %r" % (current, next)
                    )
                for i in range(len(new_args)):
                    new_args[i] = new_args[i] * next.args[i]
            else:
                if isinstance(next, Pow):
                    if isinstance(next.base, TensorProduct):
                        new_tp = tensor_product_simp_Pow(next)
                        for i in range(len(new_args)):
                            new_args[i] = new_args[i] * new_tp.args[i]
                    else:
                        raise TypeError("TensorProduct expected, got: %r" % next)
                else:
                    raise TypeError("TensorProduct expected, got: %r" % next)
            current = next
        return Mul(*c_part) * TensorProduct(*new_args)
    elif e.has(Pow):
        new_args = [tensor_product_simp_Pow(nc) for nc in nc_part]
        return tensor_product_simp_Mul_fork(Mul(*c_part) * TensorProduct(*new_args))
    else:
        return e


def tensor_product_simp_fork(e, **hints):  # type: ignore
    """Try to simplify and combine TensorProducts.

    In general this will try to pull expressions inside of ``TensorProducts``.
    It currently only works for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``. It is best to see what it does by showing examples.

    Examples
    ========

    >>> from sympy.physics.quantum import tensor_product_simp
    >>> from sympy.physics.quantum import TensorProduct
    >>> from sympy import Symbol
    >>> A = Symbol('A',commutative=False)
    >>> B = Symbol('B',commutative=False)
    >>> C = Symbol('C',commutative=False)
    >>> D = Symbol('D',commutative=False)

    First see what happens to products of tensor products:

    >>> e = TensorProduct(A,B)*TensorProduct(C,D)
    >>> e
    AxB*CxD
    >>> tensor_product_simp(e)
    (A*C)x(B*D)

    This is the core logic of this function, and it works inside, powers, sums,
    commutators and anticommutators as well:

    >>> tensor_product_simp(e**2)
    (A*C)x(B*D)**2

    """
    if isinstance(e, Add):
        return Add(*[tensor_product_simp_fork(arg) for arg in e.args])
    elif isinstance(e, Pow):
        if isinstance(e.base, TensorProduct):
            return tensor_product_simp_Pow(e)
        else:
            return tensor_product_simp_fork(e.base) ** e.exp
    elif isinstance(e, Mul):
        return tensor_product_simp_Mul_fork(e)
    elif isinstance(e, Commutator):
        return Commutator(*[tensor_product_simp(arg) for arg in e.args])
    elif isinstance(e, AntiCommutator):
        return AntiCommutator(*[tensor_product_simp(arg) for arg in e.args])
    else:
        return e
