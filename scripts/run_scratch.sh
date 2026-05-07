#!/usr/bin/env bash
# Train ResNet-50 from random initialization on Cornell Grasping Dataset.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
python -m src.train --output results/scratch "$@"
