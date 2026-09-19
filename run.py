#!/usr/bin/env python3
"""Start SpikeNet Lab.

    pip install -r requirements.txt
    python run.py

Everything lives in this one directory. There is no package to install and no
subfolder to change into.

The two checks below exist because the failure modes they catch are both common
and both produce errors that point at the wrong thing. A missing dependency
gives you an ImportError deep inside Brian2; a missing source file gives you
"No module named 'builder'", which sounds like a broken install rather than an
incomplete download.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Put this directory first on the import path so `python run.py` works from any
# working directory, e.g. `python ~/code/spikenet-lab/run.py`.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

#: Every file the app needs at runtime. Tests and the demo are not listed:
#: the server runs fine without them.
REQUIRED_FILES = [
    "server.py",
    "runner.py",
    "builder.py",
    "validation.py",
    "schema.py",
    "safe_units.py",
    "analysis.py",
    "sweep.py",
    "index.html",
]

REQUIRED_PACKAGES = [
    ("brian2", "brian2"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pydantic", "pydantic"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]


def check_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (HERE / name).is_file()]
    if not missing:
        return
    print(f"\n  These files are missing from {HERE}:\n")
    for name in missing:
        print(f"      {name}")
    print(
        "\n  This usually means only some files were downloaded. Every file has\n"
        "  to sit together in one folder. Re-extract the whole archive and run\n"
        "  this from inside the folder it creates.\n"
    )
    print(f"  Files currently here: {', '.join(sorted(p.name for p in HERE.iterdir()))}\n")
    raise SystemExit(1)


def check_packages() -> None:
    missing = []
    for module, package in REQUIRED_PACKAGES:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return
    print(f"\n  Missing Python packages: {', '.join(missing)}")
    print("\n  Install them with:\n")
    print("      pip install -r requirements.txt\n")
    raise SystemExit(1)


if __name__ == "__main__":
    check_files()
    check_packages()

    from server import main

    main()
