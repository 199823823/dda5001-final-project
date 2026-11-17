#!/usr/bin/env bash
# Classroom Ubuntu entry point. Reuse the preinstalled course environment.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $(uname -s) != Linux ]]; then
    echo 'Run this script in the classroom Ubuntu terminal.' >&2
    exit 1
fi
if [[ -f /opt/dda5001/venv/bin/activate ]]; then
    source /opt/dda5001/venv/bin/activate
else
    echo 'Course environment not found at /opt/dda5001/venv.' >&2
    echo 'Ask the TA for its location; do not install over the classroom environment.' >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    set -- verify --gpu
fi
exec python "$ROOT/lab.py" "$@"
