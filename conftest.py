# conftest.py  (project root)
# ─────────────────────────────────────────────────────────────────────────────
# Ensures the project root is on sys.path so that imports like
#   from workload.llm.inference_workload import VLLMServer
#   from collectors.dcgm_collector import TelemetryCollector
#   from validators.run_validators import validate_utilization
# work correctly regardless of where pytest is invoked from.
#
# This file is intentionally minimal — test fixtures live in tests/conftest.py.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Insert the project root (the directory containing this file) at the front
# of sys.path. This is idempotent — inserting a path that is already present
# has no effect.
ROOT = str(Path(__file__).parent.resolve())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)