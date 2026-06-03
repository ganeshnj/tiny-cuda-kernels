#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-jetson-n1}"
JETSON_DIR="${JETSON_DIR:-~/dev/tiny-cuda-kernels}"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: ./scripts/run_on_jetson.sh <target> [args ...]" >&2
  echo "Example: ./scripts/run_on_jetson.sh vector_add 16777216 100 10 --csv" >&2
  exit 1
fi
shift

REMOTE_EXE="${JETSON_DIR}/build/${TARGET}"

cmd="$(printf '%q ' "${REMOTE_EXE}" "$@")"

echo "Running ${TARGET} on ${JETSON_HOST}"
ssh "${JETSON_HOST}" "${cmd}"
