#!/bin/bash
cd /workspace || exit 1
pip install -e .
exec "$@"
