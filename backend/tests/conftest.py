"""Make the flat backend modules (app, inference) importable from tests."""

import pathlib
import sys

# backend/ is the parent of this tests/ dir.
BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
