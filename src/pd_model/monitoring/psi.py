import numpy as np


def make_distribution(values, bins=10, categorical=False):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("Пустой reference")
    if categorical:
        categories = sorted(set(str(value) for value in values))
        counts = [
            int(sum(str(value) == category for value in values))
            for category in categories
        ]
        return {"kind": "categorical", "categories": categories, "counts": counts + [0]}
    numeric = values.astype(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Reference должен содержать конечные значения")
    if np.min(numeric) == np.max(numeric):
        return {
            "kind": "constant",
            "value": float(numeric[0]),
            "counts": [0, len(values), 0],
        }
    interior = np.unique(np.quantile(numeric, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    counts = np.histogram(numeric, bins=edges)[0].tolist()
    return {"kind": "numeric", "interior_edges": interior.tolist(), "counts": counts}


def new_counts(reference, values):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("Пустой новый батч")
    if reference["kind"] == "categorical":
        categories = reference["categories"]
        counts = [
            sum(str(value) == category for value in values) for category in categories
        ]
        counts += [sum(str(value) not in categories for value in values)]
        return np.asarray(counts, dtype=float)
    numeric = values.astype(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Новый батч содержит NaN/inf")
    if reference["kind"] == "constant":
        point = reference["value"]
        return np.asarray(
            [sum(numeric < point), sum(numeric == point), sum(numeric > point)],
            dtype=float,
        )
    edges = [-np.inf] + reference["interior_edges"] + [np.inf]
    return np.histogram(numeric, bins=edges)[0].astype(float)


def population_stability_index(reference, values, epsilon=1e-6):
    if epsilon <= 0:
        raise ValueError("epsilon должен быть положительным")
    train = np.asarray(reference["counts"], dtype=float)
    current = new_counts(reference, values)
    if train.sum() <= 0:
        raise ValueError("Пустой reference")
    p = train / train.sum() + epsilon
    q = current / current.sum() + epsilon
    p, q = p / p.sum(), q / q.sum()
    score = float(np.sum((q - p) * np.log(q / p)))
    return {"psi": score, "reference_share": p.tolist(), "current_share": q.tolist()}
