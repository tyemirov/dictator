#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
pipeline="${repo_root}/scripts/release/publish_release.sh"
[[ -x "${pipeline}" ]] || {
  echo "error: repository-owned prepared-release publish pipeline not found: ${pipeline}" >&2
  exit 1
}

exec "${pipeline}" "$@"
