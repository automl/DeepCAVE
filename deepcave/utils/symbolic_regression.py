import numpy as np
import sympy
from gplearn import functions
from gplearn.functions import make_function

from deepcave.utils.logs import get_logger

logger = get_logger(__name__)


# Create a safe exp function which does not cause problems
def exp(x):
    with np.errstate(all="ignore"):
        max_value = np.full(shape=x.shape, fill_value=100000)
        return np.minimum(np.exp(x), max_value)


def get_function_set():
    exp_func = make_function(function=exp, arity=1, name="exp")

    function_set = ["add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "abs", exp_func]

    return function_set


def convert_symb(symb, n_decimals: int = None, hp_names: list = None) -> sympy.core.expr:
    """
    Convert a fitted symbolic regression to a simplified and potentially rounded mathematical expression.
    Warning: eval is used in this function, thus it should not be used on unsanitized input (see
    https://docs.sympy.org/latest/modules/core.html?highlight=eval#module-sympy.core.sympify).

    Parameters
    ----------
    symb: Fitted symbolic regressor to find a simplified expression for.
    n_decimals: If set, round floats in the expression to this number of decimals.
    hp_names: If set, replace X0 and X1 in the expression by the names given.

    Returns
    -------
    symb_conv: Converted mathematical expression.
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

    if symb._program.length_ > 300:
        print(
            f"Expression of length {symb._program._length} too long to convert, return raw string."
        )
        return symb_str

    symb_conv = sympy.sympify(symb_str.replace("[", "").replace("]", ""), locals=converter)
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
                f"Numer of hyperparameters to be explained by symbolic explanations must not "
                f"be larger than 2"
            )

    logger.debug("Start to simplify the expression with Sympy.")
    symb_simpl = sympy.simplify(symb_conv)

    if n_decimals:
        # Make sure also floats deeper in the expression tree are rounded
        for a in sympy.preorder_traversal(symb_simpl):
            if isinstance(a, sympy.core.numbers.Float):
                symb_simpl = symb_simpl.subs(a, round(a, n_decimals))

    return symb_simpl
