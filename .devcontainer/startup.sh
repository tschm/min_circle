#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
whoami
uv venv
uv pip install marimo
uv pip install -r requirements.txt
