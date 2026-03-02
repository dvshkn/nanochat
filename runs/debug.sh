#!/bin/bash

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup (skip with SKIP_SETUP=1)
if [ -z "$SKIP_SETUP" ]; then
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate
else
    source .venv/bin/activate
fi

uv run -m nanochat.debug
