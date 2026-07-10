"""Drift check for the vendored Qwen3-VL wrapper.

`backend/qwen_vl_chat.py` is an intentional *vendored copy* of the source of truth
`modeling/chartqa/models/qwen_vl_chat.py`, so the backend Docker image builds
without the modeling tree or its training-only deps. A vendored copy is only safe
if CI catches drift: this test strips the intentionally-divergent regions (marked
in BOTH files with `# --- vendor-sync:ignore-start/end ---`) and asserts the
remaining shared logic is byte-identical. Any edit to the shared wrapper in one
file that is not mirrored in the other fails here.

The intentional regions are: the vendoring/provenance header (differs by design)
and the modeling copy's `__main__` demo (which pulls in the training-only
`datasets` dep). Everything else must match.

The test SKIPS cleanly when the modeling counterpart is absent, so it never breaks
the backend image build (which ships without the modeling tree).
"""

import difflib
import pathlib

import pytest

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

BACKEND_COPY = BACKEND_DIR / "qwen_vl_chat.py"
MODELING_COPY = REPO_ROOT / "modeling" / "chartqa" / "models" / "qwen_vl_chat.py"

IGNORE_START = "# --- vendor-sync:ignore-start ---"
IGNORE_END = "# --- vendor-sync:ignore-end ---"


def _shared_lines(path: pathlib.Path) -> list[str]:
    """Return the file's lines with every vendor-sync:ignore region removed.

    Uses `splitlines()` so the comparison is agnostic to CRLF/LF line endings.
    Asserts the markers are balanced and correctly ordered, so a malformed
    marker pair fails loudly instead of silently hiding a divergence.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    skipping = False
    starts = ends = 0
    for line in lines:
        stripped = line.strip()
        if stripped == IGNORE_START:
            assert not skipping, f"nested vendor-sync:ignore-start in {path}"
            skipping = True
            starts += 1
            continue
        if stripped == IGNORE_END:
            assert skipping, f"vendor-sync:ignore-end without a start in {path}"
            skipping = False
            ends += 1
            continue
        if not skipping:
            kept.append(line)
    assert not skipping, f"unterminated vendor-sync:ignore region in {path}"
    assert starts == ends, f"unbalanced vendor-sync markers in {path}"
    assert starts > 0, f"no vendor-sync:ignore markers found in {path}"
    return kept


def test_backend_copy_matches_modeling_source():
    if not MODELING_COPY.exists():
        pytest.skip(
            "modeling counterpart absent (backend image builds without the modeling "
            "tree); drift check runs in dev/CI where modeling/ is present."
        )

    backend_shared = _shared_lines(BACKEND_COPY)
    modeling_shared = _shared_lines(MODELING_COPY)

    if backend_shared != modeling_shared:
        diff = "\n".join(
            difflib.unified_diff(
                modeling_shared,
                backend_shared,
                fromfile=str(MODELING_COPY.relative_to(REPO_ROOT)),
                tofile=str(BACKEND_COPY.relative_to(REPO_ROOT)),
                lineterm="",
            )
        )
        pytest.fail(
            "Vendored backend/qwen_vl_chat.py has drifted from its source of truth "
            "modeling/chartqa/models/qwen_vl_chat.py. Mirror the shared-logic edit "
            "into both files (or wrap a genuinely-intentional difference in "
            f"vendor-sync:ignore markers).\n\n{diff}"
        )
