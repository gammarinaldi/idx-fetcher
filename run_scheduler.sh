#!/bin/bash
cd "$(dirname "$0")" || exit 1

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "No venv found (.venv/bin or venv/bin)" >&2
    exit 1
fi

python scheduler.py "$@"
