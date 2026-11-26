#!/usr/bin/env bash
# Start the presentation demo on http://127.0.0.1:8765 (extra args, e.g. --allow-live, are passed on).
cd "$(dirname "$0")/.." && exec .venv/bin/python -m demo.server "$@"
