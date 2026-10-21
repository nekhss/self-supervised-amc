"""Read-only structural audit of the RadioML 2016.10a pickle.

Reports keys, classes, SNRs, shapes, dtypes, counts, finite-value status and
discrepancies from the expected layout. Never prints IQ values.

Example:
  C:\\venvs\\dlproj\\Scripts\\python.exe -I tools\\inspect_rml2016.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import rml2016  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default=str(rml2016.DEFAULT_PKL_PATH))
    args = parser.parse_args()

    path = Path(args.pkl)
    print(f"File: {path.name}  bytes={path.stat().st_size:,}")
    print(f"SHA-256: {rml2016.sha256_file(path)}")
    data = rml2016.load_raw_dict(path)
    print(json.dumps(rml2016.audit_dict(data), indent=2))
    try:
        rml2016.validate_dict(data)
        print("Layout check: PASS (matches expected RML2016.10a layout)")
    except ValueError as exc:
        print(f"Layout check: FAIL - {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
