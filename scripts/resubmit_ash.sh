#!/usr/bin/env bash
# scripts/resubmit_ash.sh - prepares exp002 (Ash variance probe) by pulling our
# existing Ash fork and pushing it back unchanged. Does NOT auto-submit; prints
# the manual `kaggle competitions submit-code` command at the end so you can
# review the new kernel version before burning today's daily submission slot.
#
# Usage:
#   bash scripts/resubmit_ash.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env so KAGGLE_USERNAME / KAGGLE_KEY are in the environment.
set -a
[ -f .env ] && . .env
set +a

# KGAT_ tokens require Bearer auth via KAGGLE_API_TOKEN; the kernels CLI honours it.
export KAGGLE_API_TOKEN="${KAGGLE_KEY:-}"

KERNEL_REF="${KERNEL_REF:-cataluna84/ash-s-arc-agi-3-agent}"
COMP_NAME="${COMP_NAME:-arc-prize-2026-arc-agi-3}"
PULL_DIR="${ROOT_DIR}/experiments/exp002_ash_variance_probe/_pulled"

KAGGLE="${ROOT_DIR}/.venv/bin/kaggle"

echo "=== exp002 - Ash variance probe ==="
echo "Kernel       : ${KERNEL_REF}"
echo "Competition  : ${COMP_NAME}"
echo "Pull dir     : ${PULL_DIR}"
echo

mkdir -p "${PULL_DIR}"

echo "[1/3] Pulling kernel sources + metadata..."
"${KAGGLE}" kernels pull "${KERNEL_REF}" -p "${PULL_DIR}" -m
ls -la "${PULL_DIR}"
echo

echo "[2/3] Pushing kernel back unchanged (this triggers a fresh run)..."
PUSH_OUT="$("${KAGGLE}" kernels push -p "${PULL_DIR}" 2>&1)"
echo "${PUSH_OUT}"
NEW_VER="$(echo "${PUSH_OUT}" | grep -oE 'version [0-9]+' | head -1 | awk '{print $2}')"
echo
echo "New kernel version: ${NEW_VER:-<could not parse>}"
echo

echo "[3/3] Wait for the kernel to finish before submitting. Poll with:"
echo "  ${KAGGLE} kernels status ${KERNEL_REF}"
echo
echo "When status == COMPLETE, manually run (this BURNS one daily submission slot):"
cat <<EOF
  ${KAGGLE} competitions submit-code \\
    -c ${COMP_NAME} \\
    --kernel ${KERNEL_REF} \\
    --kernel-version ${NEW_VER:-<VER>} \\
    -f submission.parquet \\
    -m "exp002 variance probe - Ash unchanged"
EOF
