#!/usr/bin/env python3
"""
check_reqs.py — verify that all packages in a requirements.txt are installed
with the correct versions.
"""

import sys
from importlib import metadata

def load_requirements(path):
    reqs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '==' in line:
                pkg, ver = line.split('==', 1)
                pkg = pkg.strip()
                ver = ver.strip()
                reqs.append((pkg, ver))
            else:
                print(f"⚠️  Skipping unrecognized line: {line}")
    return reqs

def main(req_file):
    reqs = load_requirements(req_file)
    print(f"Checking {len(reqs)} packages from {req_file}...\n")

    ok = 0
    missing = 0
    wrong_ver = 0

    for pkg, expected in reqs:
        try:
            installed = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            print(f"❌ {pkg:40s} not installed")
            missing += 1
            continue

        if installed == expected:
            print(f"✅ {pkg:40s} {installed}")
            ok += 1
        else:
            print(f"⚠️  {pkg:40s} installed {installed!r}, expected {expected!r}")
            wrong_ver += 1

    print("\nSummary:")
    print(f"✅ OK:         {ok:3}")
    print(f"❌ Missing:    {missing:3}")
    print(f"⚠️  Wrong ver:  {wrong_ver:3}")

    if missing or wrong_ver:
        sys.exit(1)  # non-zero exit for CI/pipelines

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_reqs.py path/to/requirements.txt")
        sys.exit(2)
    main(sys.argv[1])
