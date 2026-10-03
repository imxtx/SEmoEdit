#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${SEMOEDIT_PYTHON:-python}" "$root/scripts/run.py" --model cosyvoice2 "$@"
