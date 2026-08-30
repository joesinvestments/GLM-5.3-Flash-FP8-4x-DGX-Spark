#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_URL="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git"
UPSTREAM_COMMIT="b5ab8091dec88e324c943deb96c2dfd957db9f36"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-${PWD}/glm53-exl3-2x-abliterated}"

apply_once() {
  local patch_file="$1"
  if git -C "${DESTINATION}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    return 0
  fi
  git -C "${DESTINATION}" apply --check "${patch_file}"
  git -C "${DESTINATION}" apply "${patch_file}"
}

if [[ -e "${DESTINATION}" && ! -d "${DESTINATION}/.git" ]]; then
  echo "Refusing to replace non-git path: ${DESTINATION}" >&2
  exit 1
fi

if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone --filter=blob:none "${UPSTREAM_URL}" "${DESTINATION}"
fi

git -C "${DESTINATION}" fetch --depth=1 origin "${UPSTREAM_COMMIT}"
git -C "${DESTINATION}" checkout --detach "${UPSTREAM_COMMIT}"
apply_once "${SCRIPT_DIR}/dflash-revision-pin.patch"
apply_once "${SCRIPT_DIR}/ablit-donor-revision-pin.patch"
install -m 0600 "${SCRIPT_DIR}/cluster.spark2-spark1.env" "${DESTINATION}/.env"

actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
  echo "Unexpected upstream commit: ${actual_commit}" >&2
  exit 1
fi

echo "Pinned EXL3 recipe installed at ${DESTINATION}"
echo "Review .env, then run: cd '${DESTINATION}' && ./start.sh download"
