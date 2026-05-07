#!/usr/bin/env bash
# Download Cornell Grasping Dataset to data/cornell/.
# Idempotent: skips download if data already present.
# Primary source: Kaggle (https://www.kaggle.com/datasets/oneoneliu/cornell-grasp).
# Requires the kaggle CLI authenticated via ~/.kaggle/kaggle.json.

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/cornell"

# Idempotency guard: any pcd*r.png anywhere under data/cornell/ means we're done
mkdir -p "${DATA_DIR}"
if find "${DATA_DIR}" -name 'pcd*r.png' -print -quit 2>/dev/null | grep -q .; then
    echo "[download_data.sh] Cornell dataset already present at ${DATA_DIR} — skipping."
    exit 0
fi

echo "[download_data.sh] Cornell dataset not found at ${DATA_DIR} — attempting download."

# Verify kaggle CLI is available and authenticated
if ! command -v kaggle >/dev/null 2>&1; then
    echo "[download_data.sh] ERROR: 'kaggle' CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

# Kaggle now offers two token formats. Accept either:
#   1. KAGGLE_API_TOKEN env var (new format, KGAT_... string)
#   2. ~/.kaggle/access_token file containing the new-format token
#   3. ~/.kaggle/kaggle.json (legacy username+key format)
if [[ -z "${KAGGLE_API_TOKEN:-}" \
      && ! -f "${HOME}/.kaggle/access_token" \
      && ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    cat >&2 <<'EOF'
[download_data.sh] ERROR: no Kaggle credentials found.

Provide one of the following:
  - export KAGGLE_API_TOKEN="KGAT_..."        (new format, recommended)
  - ~/.kaggle/access_token  with the KGAT_... token as its only content
  - ~/.kaggle/kaggle.json   (legacy username+key file)

Get a token at https://www.kaggle.com/settings/account → "Create New API Token".
EOF
    exit 1
fi

TMP_DIR="${REPO_ROOT}/data/.cornell_download"
mkdir -p "${TMP_DIR}"

echo "[download_data.sh] Downloading oneoneliu/cornell-grasp from Kaggle..."
kaggle datasets download -d oneoneliu/cornell-grasp -p "${TMP_DIR}" --unzip --force

# The Kaggle dataset typically extracts to a subdir like "cornell-grasp/" or directly to pcd*.
# Move all pcd* files into data/cornell/ so dataset.py finds them by glob.
echo "[download_data.sh] Organizing files into ${DATA_DIR}..."
find "${TMP_DIR}" -type f \( -name 'pcd*' -o -name 'z.txt' \) -exec mv -n {} "${DATA_DIR}/" \;

# Cleanup
rm -rf "${TMP_DIR}"

# Verify
COUNT=$(find "${DATA_DIR}" -name 'pcd*r.png' | wc -l | tr -d ' ')
if [[ "${COUNT}" -eq 0 ]]; then
    echo "[download_data.sh] ERROR: download finished but no pcd*r.png files in ${DATA_DIR}." >&2
    exit 1
fi

echo "[download_data.sh] Done. ${COUNT} RGB images in ${DATA_DIR}."
