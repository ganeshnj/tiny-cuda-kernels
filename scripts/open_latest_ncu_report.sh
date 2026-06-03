#!/usr/bin/env bash
set -euo pipefail

REPORT_DIR="${1:-reports/ncu}"
APP_NAME="${NCU_APP_NAME:-NVIDIA Nsight Compute}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${REPORT_DIR}" ]]; then
  echo "Report directory not found: ${REPORT_DIR}" >&2
  exit 1
fi

LATEST_REPORT="$(find "${REPORT_DIR}" -maxdepth 1 -type f -name '*.ncu-rep' -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_REPORT}" ]]; then
  echo "No .ncu-rep files found in ${REPORT_DIR}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Would open: ${LATEST_REPORT}"
  exit 0
fi

if ! open -Ra "${APP_NAME}" 2>/dev/null; then
  echo "Application not found: ${APP_NAME}" >&2
  echo "Set NCU_APP_NAME to your installed app name." >&2
  exit 1
fi

open -a "${APP_NAME}" "${LATEST_REPORT}"
echo "Opened ${LATEST_REPORT} with ${APP_NAME}."
