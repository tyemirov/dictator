#!/usr/bin/env bash
set -euo pipefail

install_dir="${1:-/opt/cosyvoice}"
requirements_file="${2:-/app/requirements-cosyvoice3.txt}"
patch_script="${3:-/app/scripts/patch_cosyvoice3_runtime.py}"
repo_url="${COSYVOICE3_REPO_URL:-https://github.com/FunAudioLLM/CosyVoice.git}"
repo_ref="${COSYVOICE3_REPO_REF:-04bcadc6340e266b4ba09f4474b4668c444aa063}"

if [[ ! -f "${requirements_file}" ]]; then
  echo "CosyVoice requirements file not found: ${requirements_file}" >&2
  exit 1
fi

if [[ ! -f "${patch_script}" ]]; then
  echo "CosyVoice patch script not found: ${patch_script}" >&2
  exit 1
fi

python -m pip install --no-cache-dir -r "${requirements_file}"

git clone "${repo_url}" "${install_dir}"
git -C "${install_dir}" checkout "${repo_ref}"
git -C "${install_dir}" submodule update --init --recursive
python "${patch_script}" "${install_dir}"

# Keep the runtime copy lean and avoid carrying git metadata into the image.
find "${install_dir}" -name .git -prune -exec rm -rf {} +
rm -f "${install_dir}/.gitmodules"
