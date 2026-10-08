"""AST parsing + feature extraction (one feature vector per function)."""
import ast

FEATURES = ["loc", "n_params", "cyclomatic", "max_nesting", "n_try", "bare_except",
            "broad_except", "except_pass", "mutable_default", "open_no_with",
            "n_returns", "uses_global"]

_NEST = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
_BRANCH = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp,
           ast.comprehension, ast.Assert)


def _depth(node, d=0):
    best = d
    for child in ast.iter_child_nodes(node):
        best = max(best, _depth(child, d + 1 if isinstance(child, _NEST) else d))
    return best


def _features(fn):
    nodes = list(ast.walk(fn))
    handlers = [n for n in nodes if isinstance(n, ast.ExceptHandler)]
    with_calls = {id(i.context_expr) for n in nodes
                  if isinstance(n, (ast.With, ast.AsyncWith)) for i in n.items}
    opens = [n for n in nodes if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "open"
             and id(n) not in with_calls]
    defaults = fn.args.defaults + [d for d in fn.args.kw_defaults if d]
    a = fn.args
    return {
        "loc": (fn.end_lineno or fn.lineno) - fn.lineno + 1,
        "n_params": len(a.args) + len(a.kwonlyargs) + len(a.posonlyargs),
        "cyclomatic": 1 + sum(isinstance(n, _BRANCH) for n in nodes)
                      + sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp)),
        "max_nesting": _depth(fn),
        "n_try": sum(isinstance(n, ast.Try) for n in nodes),
        "bare_except": sum(h.type is None for h in handlers),
        "broad_except": sum(isinstance(h.type, ast.Name) and h.type.id in ("Exception", "BaseException")
                            for h in handlers),
        "except_pass": sum(len(h.body) == 1 and isinstance(h.body[0], ast.Pass) for h in handlers),
        "mutable_default": sum(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults),
        "open_no_with": len(opens),
        "n_returns": sum(isinstance(n, ast.Return) for n in nodes),
        "uses_global": sum(isinstance(n, (ast.Global, ast.Nonlocal)) for n in nodes),
    }


def extract(code: str):
    """Returns [{name, line, end_line, features}]. Raises SyntaxError on bad code."""
    tree = ast.parse(code)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append({"name": n.name, "line": n.lineno, "end_line": n.end_lineno or n.lineno,
                        "features": _features(n)})
    return sorted(out, key=lambda f: f["line"])
