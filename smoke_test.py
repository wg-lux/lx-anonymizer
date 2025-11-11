#!/usr/bin/env python3
"""
Runtime smoke test for lx_anonymizer.
Ensures that every module can be imported cleanly
without triggering ImportErrors or SyntaxErrors.
"""

import pkgutil
import importlib
import traceback
import sys

PACKAGE_NAME = "lx_anonymizer"

def main():
    print(f"🚀 Running runtime smoke test for package: {PACKAGE_NAME}\n")
    success, failed = [], []

    try:
        package = importlib.import_module(PACKAGE_NAME)
    except Exception as e:
        print(f"❌ Cannot import root package {PACKAGE_NAME}: {e}")
        traceback.print_exc()
        sys.exit(1)

    for mod_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module_name = mod_info.name
        try:
            importlib.import_module(module_name)
            success.append(module_name)
        except Exception as e:
            failed.append((module_name, e))
            print(f"❌ Failed to import {module_name}: {e.__class__.__name__}: {e}")
            traceback.print_exc(limit=1)

    print("\n✅ Successfully imported:", len(success))
    print("❌ Failed imports:", len(failed))

    if failed:
        print("\n--- Failed module summary ---")
        for name, e in failed:
            print(f"{name}: {e.__class__.__name__} — {e}")

        sys.exit(1)  # fail CI if needed
    else:
        print("\n🎉 All modules imported successfully!")


if __name__ == "__main__":
    main()
