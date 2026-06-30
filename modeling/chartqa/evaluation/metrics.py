from chartqa.constants import RELAXED_TOLERANCE


def _to_number(text: str):
    """Parse a numeric answer, tolerating '%' and thousands separators."""
    cleaned = text.strip().rstrip("%").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def relaxed_match(pred: str, gold: str, tol: float = RELAXED_TOLERANCE) -> bool:
    """ChartQA relaxed correctness: tolerance for numbers, exact match for text."""
    pred, gold = pred.strip(), gold.strip()
    p, g = _to_number(pred), _to_number(gold)
    if p is not None and g is not None:
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= tol
    return pred.lower() == gold.lower()


def exact_match(pred: str, gold: str) -> bool:
    """Strict (case-sensitive) string equality, ignoring surrounding whitespace."""
    return pred.strip() == gold.strip()


# Available scoring functions, selectable via --metric.
METRICS = {"relaxed": relaxed_match, "exact": exact_match}
