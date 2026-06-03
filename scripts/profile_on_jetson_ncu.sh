#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-jetson-n1}"
JETSON_DIR="${JETSON_DIR:-~/dev/tiny-cuda-kernels}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCU_SET="${NCU_SET:-launchstats}"
USE_SUDO="${USE_SUDO:-auto}"

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: ./scripts/profile_on_jetson_ncu.sh <target> [app args ...]" >&2
  echo "Example: ./scripts/profile_on_jetson_ncu.sh vector_add 16777216 50 10" >&2
  exit 1
fi
shift

REPORT_TAG="${REPORT_TAG:-${TARGET}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_REPORT_DIR="${JETSON_DIR}/reports/ncu"
LOCAL_REPORT_DIR="${REPO_ROOT}/reports/ncu"
REPORT_BASENAME="${REPORT_TAG}_${TIMESTAMP}"
REMOTE_REPORT_BASE="${REMOTE_REPORT_DIR}/${REPORT_BASENAME}"
REMOTE_NCU="${CUDA_HOME}/bin/ncu"
REMOTE_APP="${JETSON_DIR}/build/${TARGET}"

mkdir -p "${LOCAL_REPORT_DIR}"

echo "Profiling ${TARGET} with Nsight Compute on ${JETSON_HOST}"
echo "NCU set: ${NCU_SET}"

ssh "${JETSON_HOST}" "mkdir -p ${REMOTE_REPORT_DIR}"

build_remote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    out+="$(printf '%q ' "${arg}")"
  done
  printf '%s' "${out}"
}

run_plain() {
  local remote
  remote="$(build_remote_cmd "${REMOTE_NCU}" --set "${NCU_SET}" --target-processes all --force-overwrite --export "${REMOTE_REPORT_BASE}" "${REMOTE_APP}" "$@")"
  ssh "${JETSON_HOST}" "${remote}"
}

run_sudo_non_interactive() {
  local remote
  remote="$(build_remote_cmd sudo -n "${REMOTE_NCU}" --set "${NCU_SET}" --target-processes all --force-overwrite --export "${REMOTE_REPORT_BASE}" "${REMOTE_APP}" "$@")"
  ssh "${JETSON_HOST}" "${remote}"
}

run_sudo_interactive() {
  local remote
  remote="$(build_remote_cmd sudo "${REMOTE_NCU}" --set "${NCU_SET}" --target-processes all --force-overwrite --export "${REMOTE_REPORT_BASE}" "${REMOTE_APP}" "$@")"
  ssh -t "${JETSON_HOST}" "${remote}"
}

has_report() {
  ssh "${JETSON_HOST}" "test -f ${REMOTE_REPORT_BASE}.ncu-rep"
}

if [[ "${USE_SUDO}" == "always" ]]; then
  run_sudo_non_interactive "$@"
elif [[ "${USE_SUDO}" == "interactive" ]]; then
  run_sudo_interactive "$@"
elif [[ "${USE_SUDO}" == "never" ]]; then
  run_plain "$@"
else
  if ! run_plain "$@" || ! has_report; then
    echo "Plain Nsight Compute run failed. Trying sudo if available..."
    if ! run_sudo_non_interactive "$@" || ! has_report; then
      echo "Could not run Nsight Compute with non-interactive sudo." >&2
      echo "Re-run with interactive sudo prompt:" >&2
      echo "  USE_SUDO=interactive ./scripts/profile_on_jetson_ncu.sh ${TARGET} $*" >&2
      exit 1
    fi
  fi
fi

if ! has_report; then
  echo "Nsight Compute did not produce a report file on Jetson." >&2
  echo "Try running with: USE_SUDO=interactive ./scripts/profile_on_jetson_ncu.sh ${TARGET} $*" >&2
  exit 1
fi

scp "${JETSON_HOST}:${REMOTE_REPORT_BASE}.ncu-rep" "${LOCAL_REPORT_DIR}/"

echo "NCU report copied to ${LOCAL_REPORT_DIR}/${REPORT_BASENAME}.ncu-rep"
echo "Tip: open it with Nsight Compute UI (ncu-ui)."
