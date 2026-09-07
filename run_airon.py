#!/usr/bin/env python3
"""Entry point: python run_airon.py [--windowed] [--no-depth] [--no-mirror]"""

from airon.app import main

if __name__ == "__main__":
    raise SystemExit(main())
