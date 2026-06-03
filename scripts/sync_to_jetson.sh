#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-jetson-n1}"
JETSON_DIR="${JETSON_DIR:-~/dev/tiny-cuda-kernels}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Syncing ${REPO_ROOT} -> ${JETSON_HOST}:${JETSON_DIR}"

ssh "${JETSON_HOST}" "mkdir -p ${JETSON_DIR}"

rsync -az --delete \
  --exclude ".git" \
  --exclude "build" \
  --exclude "build-*" \
  "${REPO_ROOT}/" "${JETSON_HOST}:${JETSON_DIR}/"

echo "Sync complete."
