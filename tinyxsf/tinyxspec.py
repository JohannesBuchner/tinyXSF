"""Set up an XSPEC-style model, manage per-parameter priors and plot prior-predictive spectra.

Run as::

    tinyxspec "ztbabs*(zpowerlw + pexmon)"

or ``python -m tinyxsf.tinyxspec "ztbabs*(zpowerlw + pexmon)"``.

The model expression is parsed into a tree of components combined by
additions, multiplications and convolutions (``convmodel{expression}``,
as in XSPEC). Component occurrences are numbered breadth-first following the
XSPEC convention: the first occurrence keeps its name, later ones of the same
name get a _2, _3, ... suffix (e.g. "gaussian" and "gaussian_2").

The model expression is stored in the first line of ``parameters.txt``,
followed by one ``<component>:<parameter> <prior>`` line per model parameter.
The prior specification is a comma-separated list ``min,max,distribution``
with optional distribution arguments:

* ``min,max,uniform`` -- uniform between min and max
* ``min,max,loguniform`` -- log-uniform between min and max
* ``min,max,normal,mean,sigma`` -- normal distribution (min, max are recorded bounds)
* ``min,max,lognormal,mean,sigma`` -- log-normal, 10**normal(mean, sigma)
* ``min,max,fixed,mean`` -- fixed at the value mean
* ``min,max,linked,othercomponent:otherparameter`` -- takes the value of another parameter

The file is user-editable: when the script is re-run, priors of parameters in
the current model are reused, lines of parameters not in the current model are
commented out (and reactivated with their prior if the parameter re-appears
later), and missing parameters are added with a default prior.

Finally, 20 points are sampled from the priors, the model is evaluated
following the tree and the spectra are plotted.
"""
import argparse
import ast
import difflib
import os
import re

import numpy as np

from .model import x, xvec

PARAM_LINE_RE = re.compile(r"^\s*(?P<key>[A-Za-z_]\w*:[A-Za-z_]\w*)\s+(?P<prior>\S.*?)\s*$")
COMMENTED_PARAM_LINE_RE = re.compile(r"^#\s*(?P<key>[A-Za-z_]\w*:[A-Za-z_]\w*)\s+(?P<prior>\S.*?)\s*$")
KEY_RE = re.compile(r"[A-Za-z_]\w*:[A-Za-z_]\w*")


def parse_model_expression(expression):
    """Parse an XSPEC-style model expression into a tree.

    Components are combined by additions, multiplications and
    convolutions, with the usual precedence (multiplication before
    addition) and parentheses. A convolution model wraps the expression
    it convolves in braces, e.g. ``gsmooth{powerlw}``.

    The expression is parsed with the python ast module, after
    converting the convolution braces to call parentheses.

    Parameters
    ----------
    expression: str
        model expression, e.g. "ztbabs*(zpowerlw + pexmon)"

    Returns
    -------
    tree: tuple
        nested nodes: ("comp", name), ("conv", name, subtree) and
        (operator, left, right) with operator one of "+", "-", "*"

    Raises
    ------
    ValueError
        if the expression contains unsupported characters or cannot be parsed
    """
    if not re.fullmatch(r"[A-Za-z0-9_+\-*(){}\s]*", expression or ""):
        raise ValueError(f"Unsupported characters in model expression: {expression!r}")
    python_expression = (expression or "").replace("{", "(").replace("}", ")").strip()
    try:
        module = ast.parse(python_expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Cannot parse model expression {expression!r}: {e}") from e
    return build_tree(module.body, expression)


def build_tree(node, expression):
    """Convert a python ast node into a model tree node.

    Parameters
    ----------
    node: ast.AST
        python ast node of the parsed model expression
    expression: str
        original model expression, for error messages

    Returns
    -------
    tree: tuple
        nested nodes: ("comp", name), ("conv", name, subtree) and
        (operator, left, right) with operator one of "+", "-", "*"
    """
    if isinstance(node, ast.Name):
        return ("comp", node.id)
    if isinstance(node, ast.BinOp):
        operator = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported operator in model expression {expression!r}")
        # chains of the same operator become one n-ary node
        children = []

        def collect(child):
            if isinstance(child, ast.BinOp) and \
                    {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}.get(type(child.op)) == operator:
                collect(child.left)
                collect(child.right)
            else:
                children.append(build_tree(child, expression))

        collect(node)
        return (operator, children)
    if isinstance(node, ast.Call):
        # convolution model wrapping an expression: conv{expr}
        if not isinstance(node.func, ast.Name) or len(node.args) != 1 or node.keywords:
            raise ValueError(f"Cannot parse convolution in model expression {expression!r}")
        return ("conv", node.func.id, build_tree(node.args[0], expression))
    raise ValueError(f"Cannot parse model expression {expression!r}: unexpected syntax")


def resolve_components(components):
    """Look up component names in the XSPEC model database.

    Names are matched case-insensitively; unique prefixes are accepted as
    abbreviations, like in XSPEC (e.g. "gauss" for "gaussian").

    Parameters
    ----------
    components: list
        component names as written in the model expression

    Returns
    -------
    canonical: list
        XSPEC model names in correct capitalisation

    Raises
    ------
    ValueError
        if a component is not a known XSPEC model
    """
    models = x.list_models()
    lookup = {name.lower(): name for name in models}
    canonical = []
    for name in components:
        lowered = name.lower()
        if lowered in lookup:
            canonical.append(lookup[lowered])
            continue
        matches = sorted(set(model for model in models if model.lower().startswith(lowered)))
        if len(matches) == 1:
            canonical.append(matches[0])
            continue
        if len(matches) > 1:
            raise ValueError(f"Ambiguous model component '{name}'. Matches: {', '.join(matches)}")
        close = difflib.get_close_matches(lowered, list(lookup), n=3, cutoff=0.4)
        suggestions = ", ".join(lookup[key] for key in close)
        if suggestions:
            raise ValueError(f"Unknown model component '{name}'. Closest XSPEC models: {suggestions}")
        raise ValueError(f"Unknown model component '{name}'.")
    return canonical


def resolve_model_tree(tree):
    """Resolve the component names of a parsed model tree.

    The tree is traversed breadth-first. Component occurrences are numbered
    in XSPEC style: the first occurrence of a name keeps it, later ones of
    the same name get a _2, _3, ... suffix (e.g. "gaussian", "gaussian_2").

    Parameters
    ----------
    tree: tuple
        see parse_model_expression

    Returns
    -------
    occurrences: list
        one dict per component occurrence (breadth-first order) with keys
        "compkey", "name" (canonical model name), "modeltype", "language",
        "params" (list of (name, default, softmin, softmax, frozen) including
        the normalisation for non-convolution models), "col_start", "nparams"
        and "has_norm"
    node_keys: dict
        maps id(node) -> compkey for component and convolution nodes
    npar: int
        total number of parameter columns

    Raises
    ------
    ValueError
        if a component is not a known XSPEC model, or a convolution model is
        not used with braces (or a non-convolution model is)
    """
    queue = [tree]
    nodes = []
    names = []
    while queue:
        current = queue.pop(0)
        kind = current[0]
        if kind in ("comp", "conv"):
            nodes.append(current)
            names.append(current[1])
        if kind == "conv":
            queue.append(current[2])
        elif kind in ("+", "-", "*"):
            queue.extend(current[1])
    canonical = resolve_components(names)
    counts = {}
    compkeys = []
    for name in canonical:
        counts[name] = counts.get(name, 0) + 1
        compkeys.append(name if counts[name] == 1 else "%s_%d" % (name, counts[name]))
    node_keys = {}
    for comp_node, compkey in zip(nodes, compkeys):
        node_keys[id(comp_node)] = compkey
    infos = {}
    occurrences = []
    col = 0
    for comp_node, compkey, name in zip(nodes, compkeys, canonical):
        if name not in infos:
            infos[name] = x.info(name)
        info = infos[name]
        params = [(p.name, p.default, p.softmin, p.softmax, p.frozen) for p in info.parameters]
        has_norm = info.modeltype != x.ModelType.Con
        if has_norm:
            params = params + [("norm", 1.0, 0.0, 10.0, False)]
        if comp_node[0] == "comp" and not has_norm:
            raise ValueError(f"Convolution model '{name}' must be used as '{name}{{...}}'")
        if comp_node[0] == "conv" and has_norm:
            raise ValueError(
                f"Only convolution models can wrap an expression, but '{name}' is a "
                f"{info.modeltype} model")
        occurrences.append(dict(
            compkey=compkey, name=name, modeltype=info.modeltype, language=info.language,
            params=params, col_start=col, nparams=len(params), has_norm=has_norm))
        col += len(params)
    return occurrences, node_keys, col


def get_parameter_keys(occurrences):
    """Return the ordered parameter keys and their default priors.

    Parameters
    ----------
    occurrences: list
        see resolve_model_tree

    Returns
    -------
    keys: list
        parameter keys in column order, e.g. "zTBabs:nH"
    default_priors: dict
        key -> default prior specification
    """
    keys = []
    default_priors = {}
    for occ in occurrences:
        for param in occ["params"]:
            key = "%s:%s" % (occ["compkey"], param[0])
            keys.append(key)
            default_priors[key] = default_prior(occ["modeltype"], param)
    return keys, default_priors


def default_prior(modeltype, param):
    """Return the default prior specification for a model parameter.

    Frozen parameters get a fixed prior at their default value,
    normalisations get a wide log-uniform (additive models) or
    uniform (multiplicative models) prior, and other parameters
    get a uniform prior over their soft range.

    Parameters
    ----------
    modeltype: xspec_models_cxc.ModelType
        whether the component is additive, multiplicative or convolution
    param: tuple
        (name, default, softmin, softmax, frozen) parameter specification

    Returns
    -------
    prior: str
        prior specification, e.g. "0,10,uniform"
    """
    name, default, softmin, softmax, frozen = param
    if frozen:
        return f"{softmin},{softmax},fixed,{default}"
    if name == "norm":
        if modeltype == x.ModelType.Add:
            return "1e-10,10,loguniform"
        return "0.5,2,uniform"
    return f"{softmin},{softmax},uniform"


def update_parameter_file(filename, model_expression, keys, default_priors):
    """Create or update the parameter file.

    The first line of the file is the model expression, followed by one
    ``<component>:<parameter> <prior>`` line per model parameter.

    Priors of parameters in the current model are reused. Lines of parameters
    not in the current model are commented out. Missing parameters are added
    with a default prior, and commented-out entries are reactivated with their
    previous prior if the parameter re-appears in the current model. The
    previous model line is preserved as a comment.

    Parameters
    ----------
    filename: str
        file to create or update, e.g. "parameters.txt"
    model_expression: str
        model expression, e.g. "ztbabs*(zpowerlw + pexmon)"
    keys: list
        parameter keys of the current model in column order
    default_priors: dict
        key -> default prior specification for missing parameters

    Returns
    -------
    priors: dict
        parameter key -> prior specification string
    stats: dict
        with keys "created" (bool), "commented_out", "reactivated", "added"
        (lists of parameter keys)
    """
    lines = []
    if os.path.exists(filename):
        with open(filename) as f:
            lines = f.read().splitlines()

    # previous model line: first non-empty, non-comment, non-parameter line
    old_model = None
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and PARAM_LINE_RE.match(line) is None:
            old_model = stripped
            break

    current_keys = set(keys)
    present = set()
    priors = {}
    commented_out = []
    reactivated = []
    new_lines = [model_expression.strip()]
    if old_model is not None and old_model != model_expression.strip():
        new_lines.append(f"# {old_model}  # previous model")
    for line in lines:
        m = PARAM_LINE_RE.match(line)
        if m is not None:
            key, prior = m.group("key"), m.group("prior")
            if key in current_keys:
                new_lines.append(f"{key} {prior}")
                present.add(key)
                priors[key] = prior
            else:
                new_lines.append(f"# {key} {prior}")
                commented_out.append(key)
            continue
        m = COMMENTED_PARAM_LINE_RE.match(line)
        if m is not None:
            key, prior = m.group("key"), m.group("prior")
            if key in current_keys and key not in present:
                new_lines.append(f"{key} {prior}")
                present.add(key)
                priors[key] = prior
                reactivated.append(key)
            else:
                new_lines.append(line)
            continue
        if stripped == old_model:
            continue
        new_lines.append(line)

    added = []
    for key in keys:
        if key not in present:
            prior = default_priors[key]
            new_lines.append(f"{key} {prior}")
            priors[key] = prior
            added.append(key)

    with open(filename, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    stats = dict(
        created=not lines,
        commented_out=commented_out,
        reactivated=reactivated,
        added=added,
    )
    return priors, stats


def parse_prior(spec, context=""):
    """Parse a prior specification.

    The specification is a comma-separated list ``min,max,distribution``
    with optional distribution arguments:

    * ``min,max,uniform``
    * ``min,max,loguniform``
    * ``min,max,normal,mean,sigma``
    * ``min,max,lognormal,mean,sigma``
    * ``min,max,fixed,mean`` (mean is the fixed value)
    * ``min,max,linked,othercomponent:otherparameter``

    Parameters
    ----------
    spec: str
        prior specification, e.g. "0,10,uniform"
    context: str
        extra text for the error message, e.g. which file line failed

    Returns
    -------
    prior: dict
        with "kind" and the distribution arguments ("min"/"max",
        "mean"/"sigma", "value" or "target")

    Raises
    ------
    ValueError
        if the prior is not a supported distribution or its arguments are invalid
    """
    fields = [field.strip() for field in (spec or "").split(",")]
    if len(fields) < 3 or not fields[2]:
        raise ValueError(
            f"Cannot parse prior {spec!r}{context}. Expected 'min,max,distribution[,arguments]' "
            "with distribution one of: uniform, loguniform, normal, lognormal, fixed, linked")
    try:
        first, second = float(fields[0]), float(fields[1])
    except ValueError:
        raise ValueError(
            f"Cannot parse prior bounds in {spec!r}{context}: expected 'min,max,distribution[,arguments]'")
    kind = fields[2].lower()
    if kind in ("uniform", "loguniform"):
        if len(fields) != 3:
            raise ValueError(f"Prior {spec!r}{context} takes no extra arguments")
        if not second > first:
            raise ValueError(f"Prior {spec!r}{context}: need max > min")
        if kind == "loguniform" and first <= 0:
            raise ValueError(f"Prior {spec!r}{context}: need 0 < min")
        return dict(kind=kind, min=first, max=second)
    if kind in ("normal", "lognormal"):
        if len(fields) != 5:
            raise ValueError(f"Prior {spec!r}{context}: expected 'min,max,{kind},mean,sigma'")
        try:
            mean, sigma = float(fields[3]), float(fields[4])
        except ValueError:
            raise ValueError(f"Cannot parse prior arguments in {spec!r}{context}")
        if not sigma > 0:
            raise ValueError(f"Prior {spec!r}{context}: need sigma > 0")
        return dict(kind=kind, mean=mean, sigma=sigma, min=first, max=second)
    if kind == "fixed":
        if len(fields) != 4:
            raise ValueError(f"Prior {spec!r}{context}: expected 'min,max,fixed,mean'")
        try:
            value = float(fields[3])
        except ValueError:
            raise ValueError(f"Cannot parse prior arguments in {spec!r}{context}")
        return dict(kind="fixed", value=value, min=first, max=second)
    if kind == "linked":
        if len(fields) != 4 or KEY_RE.fullmatch(fields[3]) is None:
            raise ValueError(
                f"Prior {spec!r}{context}: expected 'min,max,linked,othercomponent:otherparameter'")
        return dict(kind="linked", target=fields[3], min=first, max=second)
    raise ValueError(
        f"Unknown prior distribution '{fields[2]}'{context}. Supported: "
        "uniform, loguniform, normal, lognormal, fixed, linked")


def sample_prior(prior, rng, n):
    """Sample values from a parsed prior specification.

    Linked priors are not handled here; assign the referenced parameter's
    values instead (see sample_parameters).

    Parameters
    ----------
    prior: dict
        see parse_prior
    rng: numpy.random.Generator
        random number generator to use
    n: int
        number of values to sample

    Returns
    -------
    values: array
        n sampled values
    """
    if prior["kind"] == "uniform":
        return rng.uniform(prior["min"], prior["max"], n)
    if prior["kind"] == "loguniform":
        return 10.0 ** rng.uniform(np.log10(prior["min"]), np.log10(prior["max"]), n)
    if prior["kind"] == "normal":
        return rng.normal(prior["mean"], prior["sigma"], n)
    if prior["kind"] == "lognormal":
        return 10.0 ** rng.normal(prior["mean"], prior["sigma"], n)
    return np.full(n, prior["value"])


def sample_parameters(keys, priors, rng, n):
    """Sample a parameter matrix from the prior specifications.

    Linked parameters take the values of the parameter they reference;
    chains of links are followed, circular links raise an error.

    Parameters
    ----------
    keys: list
        parameter keys in column order
    priors: dict
        parameter key -> prior specification string
    rng: numpy.random.Generator
        random number generator to use
    n: int
        number of samples to draw

    Returns
    -------
    samples: array
        parameter values, shape (n, len(keys))
    """
    parsed = {}
    for key in keys:
        parsed[key] = parse_prior(priors[key], context=f" for '{key}'")
    index = {key: i for i, key in enumerate(keys)}
    samples = np.empty((n, len(keys)))
    resolved = set()

    def assign(key, trail=()):
        if key in resolved:
            return
        if key in trail:
            raise ValueError(f"Circular linked priors: {' -> '.join(trail + (key,))}")
        spec = parsed[key]
        if spec["kind"] == "linked":
            target = spec["target"]
            if target not in parsed:
                raise ValueError(
                    f"Linked prior for '{key}' references '{target}', "
                    "which is not a parameter of the current model")
            assign(target, trail + (key,))
            samples[:, index[key]] = samples[:, index[target]]
        else:
            samples[:, index[key]] = sample_prior(spec, rng, n)
        resolved.add(key)

    for key in keys:
        assign(key)
    return samples


def evaluate_model(tree, occurrences, node_keys, param_matrix, energies):
    """Evaluate the model tree for a matrix of parameter values.

    The tree is traversed depth-first. Component spectra are combined
    following the tree structure; convolution models modify the spectrum
    of the expression they wrap (in-place, like in XSPEC).

    Parameters
    ----------
    tree: tuple
        see parse_model_expression
    occurrences: list
        see resolve_model_tree; the normalisation (last parameter of
        non-convolution components) is applied outside the model function
    node_keys: dict
        see resolve_model_tree
    param_matrix: array
        parameter values, shape (n, npar), ordered like the parameter file
    energies: array
        N+1 bin edges in keV where to evaluate the model

    Returns
    -------
    spectra: array
        photon spectra per energy bin, shape (n, N)
    """
    param_matrix = np.asarray(param_matrix, dtype=float)
    lookup = {occ["compkey"]: occ for occ in occurrences}

    def eval_node(current):
        kind = current[0]
        if kind in ("comp", "conv"):
            occ = lookup[node_keys[id(current)]]
            model_func = getattr(x, occ["name"], None)
            if model_func is None:
                raise ValueError(
                    f"Model '{occ['name']}' is not available as a python function -- "
                    "is the compiled XSPEC model library importable?")
            pars = param_matrix[:, occ["col_start"]:occ["col_start"] + occ["nparams"]]
            if kind == "comp":
                model_pars = pars[:, :-1] if occ["has_norm"] else pars
                spec = xvec(model_func, energies=energies, pars=model_pars)
                if occ["has_norm"]:
                    spec = spec * param_matrix[:, occ["col_start"] + occ["nparams"] - 1][:, None]
                return spec
            # convolution models only support the in-place variant and always
            # over-write the model argument; its dtype must match the model
            # language (float32 for FORTRAN, float64 otherwise) or the update
            # is silently skipped
            inner = eval_node(current[2])
            dtype = np.float32 if occ["language"].name.startswith("F77") else np.float64
            result = np.ascontiguousarray(inner, dtype=dtype)
            for i in range(len(result)):
                model_func(energies=energies, pars=pars[i], model=result[i])
            return result
        spectra = [eval_node(child) for child in current[1]]
        if kind == "+":
            return np.sum(spectra, axis=0)
        if kind == "-":
            return spectra[0] - np.sum(spectra[1:], axis=0)
        return np.prod(spectra, axis=0)

    return eval_node(tree)


def plot_spectra(energies, spectra, model_expression, plotfile, nsamples):
    """Plot prior-predictive spectra to a file.

    Parameters
    ----------
    energies: array
        N+1 bin edges in keV
    spectra: array
        photon spectra per energy bin, shape (n, N)
    model_expression: str
        model expression, used as plot title
    plotfile: str
        file to save the plot to, e.g. "spectra.pdf"
    nsamples: int
        number of prior samples plotted
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_mid = 0.5 * (energies[:-1] + energies[1:])
    delta_e = energies[1:] - energies[:-1]
    y = spectra * e_mid**2 / delta_e
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(e_mid, y.T, color="0.7", lw=0.8)
    ax.plot(e_mid, np.median(y, axis=0), color="k", lw=1.5,
            label=f"median of {nsamples} prior samples")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy [keV]")
    ax.set_ylabel(r"$E^2\,\mathrm{d}N/\mathrm{d}E$ [keV ph/cm$^2$/s]")
    ax.set_title(model_expression.strip(), fontsize=10)
    ax.legend()
    fig.savefig(plotfile, bbox_inches="tight")
    plt.close(fig)


def main(args=None):
    """Run the tinyxspec command line interface.

    Parses the model expression, creates/updates the parameter file,
    samples the priors, evaluates the model and plots the spectra.
    """
    parser = argparse.ArgumentParser(
        prog="tinyxspec",
        description="Set up an XSPEC-style model, manage per-parameter priors in a text file, "
                    "sample the priors and plot the resulting spectra.")
    parser.add_argument("model", help='XSPEC-style model expression, e.g. "ztbabs*(zpowerlw + pexmon)"')
    parser.add_argument("--parameters-file", default="parameters.txt",
                        help="file holding the model (first line) and one "
                             "'<component>:<parameter> <prior>' line per parameter (default: %(default)s)")
    parser.add_argument("--plot", default="spectra.pdf", help="output plot file (default: %(default)s)")
    parser.add_argument("--emin", type=float, default=0.1, help="lowest energy of the grid in keV (default: %(default)s)")
    parser.add_argument("--emax", type=float, default=50.0, help="highest energy of the grid in keV (default: %(default)s)")
    parser.add_argument("--nbins", type=int, default=500, help="number of energy bins (default: %(default)s)")
    parser.add_argument("--samples", type=int, default=20, help="number of prior samples to draw and plot (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible samples")
    options = parser.parse_args(args)

    try:
        x.chatter(0)
    except AttributeError:
        pass

    tree = parse_model_expression(options.model)
    occurrences, node_keys, npar = resolve_model_tree(tree)
    keys, default_priors = get_parameter_keys(occurrences)
    priors, stats = update_parameter_file(options.parameters_file, options.model, keys, default_priors)
    if stats["created"]:
        print(f"created {options.parameters_file}")
    else:
        message = f"updated {options.parameters_file}"
        for action, listed in [("commented out", stats["commented_out"]),
                               ("reactivated", stats["reactivated"]),
                               ("added", stats["added"])]:
            if listed:
                message += f", {action} {len(listed)}"
        print(message)
    for key in keys:
        print(f"  {key}: {priors[key]}")

    samples = sample_parameters(keys, priors, np.random.default_rng(options.seed), options.samples)
    energies = np.geomspace(options.emin, options.emax, options.nbins + 1)
    spectra = evaluate_model(tree, occurrences, node_keys, samples, energies)

    plot_spectra(energies, spectra, options.model, options.plot, options.samples)
    print(f"wrote {options.plot} ({options.samples} prior samples)")
    return 0


if __name__ == "__main__":
    main()
