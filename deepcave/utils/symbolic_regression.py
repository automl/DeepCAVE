#  noqa: D400
"""
# Symbolic Regression

This module provides utilities for running symbolic regression with gplearn.
"""

from typing import List, Optional, Union

import numpy as np
import sympy
from gplearn import functions
from gplearn.functions import _Function, make_function
from gplearn.genetic import SymbolicRegressor

from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


def exp(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Get a safe exp function with a maximum value of 100000 to avoid overflow.

    Parameters
    ----------
    x : Union[float, np.ndarray]
        The value to calculate the exponential of.

    Returns
    -------
    Union[float, np.ndarray]
        The safe exponential of x.
    """
    with np.errstate(all="ignore"):
        max_value: Union[float, np.ndarray]
        if isinstance(x, np.ndarray):
            max_value = np.full(shape=x.shape, fill_value=100000)
        else:
            max_value = 100000.0
        return np.minimum(np.exp(x), max_value)


def get_function_set() -> List[Union[str, _Function]]:
    """
    Get a function set for symbolic regression with gplearn.

    Returns
    -------
    List[Union[str, _Function]]
        List of functions to use in symbolic regression.
    """
    exp_func = make_function(function=exp, arity=1, name="exp")

    function_set = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "abs", exp_func]

    return function_set


def convert_symb(
    symb: SymbolicRegressor, n_decimals: Optional[int] = None, hp_names: Optional[List[str]] = None
) -> str:
    """
    Convert a fitted symbolic regression to a simplified and rounded mathematical expression.

    Warning: eval is used in this function, thus it should not be used on unsanitized input (see
    https://docs.sympy.org/latest/modules/core.html?highlight=eval#module-sympy.core.sympify).

    Parameters
    ----------
    symb: SymbolicRegressor
        Fitted symbolic regressor to find a simplified expression for.
    n_decimals: Optional[int]
        If set, round floats in the expression to this number of decimals.
    hp_names: Optional[List[str]]
        If set, replace X0 and X1 in the expression by the names given.

    Returns
    -------
    str
        Converted mathematical expression.
    """
    # sqrt is protected function in gplearn, always returning sqrt(abs(x))
    sqrt_pos = []
    prev_sqrt_inserts = 0
    for i, f in enumerate(symb._program.program):
        if isinstance(f, functions._Function) and f.name == "sqrt":
            sqrt_pos.append(i)
    for i in sqrt_pos:
        symb._program.program.insert(i + prev_sqrt_inserts + 1, functions.abs1)
        prev_sqrt_inserts += 1

    # log is protected function in gplearn, always returning sqrt(abs(x))
    log_pos = []
    prev_log_inserts = 0
    for i, f in enumerate(symb._program.program):
        if isinstance(f, functions._Function) and f.name == "log":
            log_pos.append(i)
    for i in log_pos:
        symb._program.program.insert(i + prev_log_inserts + 1, functions.abs1)
        prev_log_inserts += 1

    symb_str = str(symb._program)

    converter = {
        "sub": lambda x, y: x - y,
        "div": lambda x, y: x / y,
        "mul": lambda x, y: x * y,
        "add": lambda x, y: x + y,
        "neg": lambda x: -x,
        "pow": lambda x, y: x**y,
    }

    # Abort conversion for very long programs, as they take too much time or do not finish at all.
    if symb._program.length_ > 300:
        return symb_str

    # Convert formula string to SymPy object
    symb_conv = sympy.sympify(symb_str.replace("[", "").replace("]", ""), locals=converter)

    # Replace variable names in formula by hyperparameter names
    if hp_names is not None:
        if len(hp_names) == 1:
            X0, hp0 = sympy.symbols(f"X0 {hp_names[0]}")
            symb_conv = symb_conv.subs(X0, hp0)
        elif len(hp_names) == 2:
            X0, hp0, X1, hp1 = sympy.symbols(f"X0 {hp_names[0]} X1 {hp_names[1]}")
            symb_conv = symb_conv.subs(X0, hp0)
            symb_conv = symb_conv.subs(X1, hp1)
        else:
            raise ValueError(
                "Numer of hyperparameters to be explained by symbolic explanations must not "
                "be larger than 2"
            )

    # Simplify the formula
    try:
        # Simplification can fail in some cases. If so, use the unsimplified version.
        symb_simpl = sympy.simplify(symb_conv)
    except Exception as e:
        logger.debug(
            f"Simplification of symbolic regression failed, use unsimplified expression "
            f"instead: {e}"
        )
        symb_simpl = symb_conv

    # Round floats to n_decimals
    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_simpl):
            if isinstance(a, sympy.core.numbers.Float):
                symb_simpl = symb_simpl.subs(a, round(a, n_decimals))

    return str(symb_simpl)
