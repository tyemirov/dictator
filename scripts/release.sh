#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/release.sh [options]

Cuts a repository release from the default branch without publishing images,
publishing Pages, or deploying production.

Options:
  --bump <patch|minor|major>  SemVer bump when RELEASE_VERSION is not set. Default: patch
  --version <value>           Exact release tag/version to use, e.g. v1.2.3
  --scheme <semver|calver>    Override detected versioning scheme
  --dry-run                   Run preflight and report the selected version only
  --skip-pages-verify         Skip GitHub Pages reachability during release verification
  --help                      Show this help text
USAGE
}

if [[ -v RELEASE_HELPER ]]; then
  HELPER="${RELEASE_HELPER}"
else
  HELPER=""
fi
if [[ -v RELEASE_BUMP ]] && [[ -n "${RELEASE_BUMP}" ]]; then
  BUMP="${RELEASE_BUMP}"
else
  BUMP="patch"
fi
if [[ -v RELEASE_VERSION ]]; then
  VERSION="${RELEASE_VERSION}"
else
  VERSION=""
fi
if [[ -v RELEASE_SCHEME ]]; then
  SCHEME="${RELEASE_SCHEME}"
else
  SCHEME=""
fi
DRY_RUN="false"
SKIP_PAGES_VERIFY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bump)
      [[ $# -ge 2 ]] || { echo "error: --bump requires a value" >&2; exit 1; }
      BUMP="$2"
      shift 2
      ;;
    --version)
      [[ $# -ge 2 ]] || { echo "error: --version requires a value" >&2; exit 1; }
      VERSION="$2"
      shift 2
      ;;
    --scheme)
      [[ $# -ge 2 ]] || { echo "error: --scheme requires a value" >&2; exit 1; }
      SCHEME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --skip-pages-verify)
      SKIP_PAGES_VERIFY="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${BUMP}" in
  patch|minor|major) ;;
  *) echo "error: --bump must be patch, minor, or major" >&2; exit 1 ;;
esac
case "${SCHEME}" in
  ""|semver|calver) ;;
  *) echo "error: --scheme must be semver or calver" >&2; exit 1 ;;
esac

command -v git >/dev/null 2>&1 || { echo "error: git is required" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "error: gh is required" >&2; exit 1; }
command -v gix >/dev/null 2>&1 || { echo "error: gix is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 is required" >&2; exit 1; }

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

resolve_release_helper() {
  local candidate
  if [[ -n "${HELPER}" ]]; then
    printf "%s\n" "${HELPER}"
    return
  fi
  if command -v release_helper.py >/dev/null 2>&1; then
    command -v release_helper.py
    return
  fi
  candidate="${repo_root}/../agentSkills/gitrelease/scripts/release_helper.py"
  if [[ -x "${candidate}" ]]; then
    printf "%s\n" "${candidate}"
    return
  fi
  if [[ -v CODEX_HOME ]] && [[ -n "${CODEX_HOME}" ]]; then
    candidate="${CODEX_HOME}/skills/gitrelease/scripts/release_helper.py"
    if [[ -x "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return
    fi
  fi
  if [[ -v HOME ]] && [[ -n "${HOME}" ]]; then
    candidate="${HOME}/.codex/skills/gitrelease/scripts/release_helper.py"
    if [[ -x "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return
    fi
  fi
}

HELPER="$(resolve_release_helper)"
[[ -n "${HELPER}" ]] || { echo "error: release helper not found; set RELEASE_HELPER=/path/to/release_helper.py or install the Git Release skill at ../agentSkills/gitrelease or ~/.codex/skills/gitrelease" >&2; exit 1; }
[[ -x "${HELPER}" ]] || { echo "error: release helper is not executable: ${HELPER}" >&2; exit 1; }

json_value() {
  local json_file="$1"
  local path="$2"
  python3 - "$json_file" "$path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = json.load(handle)
for part in sys.argv[2].split("."):
    if value is None:
        break
    value = value.get(part)
print("" if value is None else value)
PY
}

semver_bump() {
  local latest_tag="$1"
  local bump="$2"
  python3 - "$latest_tag" "$bump" <<'PY'
import re
import sys

latest = sys.argv[1].strip()
bump = sys.argv[2]
if not latest:
    print("v1.0.0")
    raise SystemExit
match = re.match(r"^(v?)(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", latest)
if not match:
    raise SystemExit(f"latest SemVer tag is invalid: {latest}")
prefix, major, minor, patch = match.groups()
major, minor, patch = int(major), int(minor), int(patch)
if bump == "major":
    major, minor, patch = major + 1, 0, 0
elif bump == "minor":
    minor, patch = minor + 1, 0
else:
    patch += 1
print(f"{prefix or 'v'}{major}.{minor}.{patch}")
PY
}

select_version() {
  local preflight_json="$1"
  if [[ -n "${VERSION}" ]]; then
    printf '%s\n' "${VERSION}"
    return
  fi

  local detected_scheme="${SCHEME}"
  if [[ -z "${detected_scheme}" ]]; then
    detected_scheme="$(json_value "${preflight_json}" "version_info.scheme_guess")"
  fi

  case "${detected_scheme}" in
    semver|mixed)
      semver_bump "$(json_value "${preflight_json}" "version_info.latest_semver_tag")" "${BUMP}"
      ;;
    calver)
      local calver_ok
      calver_ok="$(json_value "${preflight_json}" "version_info.calver_candidate.ok")"
      [[ "${calver_ok}" == "True" || "${calver_ok}" == "true" ]] || {
        echo "error: CalVer candidate is not valid for this release timestamp" >&2
        exit 1
      }
      json_value "${preflight_json}" "version_info.next_calver"
      ;;
    none|"")
      semver_bump "" "${BUMP}"
      ;;
    *)
      echo "error: unsupported versioning scheme: ${detected_scheme}" >&2
      exit 1
      ;;
  esac
}

select_changelog_boundary() {
  local preflight_json="$1"
  local selected_version="$2"
  local detected_scheme="${SCHEME}"
  if [[ -z "${detected_scheme}" ]]; then
    detected_scheme="$(json_value "${preflight_json}" "version_info.scheme_guess")"
  fi

  if [[ "${selected_version}" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    json_value "${preflight_json}" "version_info.latest_semver_tag"
    return
  fi
  if [[ "${detected_scheme}" == "calver" ]]; then
    json_value "${preflight_json}" "version_info.latest_calver_tag"
    return
  fi
  json_value "${preflight_json}" "version_info.latest_tag"
}

release_timestamp="$(date +%Y-%m-%dT%H:%M:%S)"
release_date="${release_timestamp%%T*}"
preflight_json="$(mktemp)"
notes_file="$(mktemp)"
cleanup() {
  rm -f "${preflight_json}" "${notes_file}"
}
trap cleanup EXIT

echo "==> [release] Running preflight"
"${HELPER}" preflight --release-timestamp "${release_timestamp}" | tee "${preflight_json}"
default_branch="$(json_value "${preflight_json}" "default_branch")"
[[ -n "${default_branch}" ]] || { echo "error: release helper did not report default_branch" >&2; exit 1; }

if [[ "${DRY_RUN}" == "true" ]]; then
  next_version="$(select_version "${preflight_json}")"
  boundary_tag="$(select_changelog_boundary "${preflight_json}" "${next_version}")"
  echo "release_dry_run=true"
  echo "default_branch=${default_branch}"
  echo "next_version=${next_version}"
  if [[ -n "${boundary_tag}" ]]; then
    echo "changelog_boundary=${boundary_tag}"
  else
    echo "changelog_boundary=<none>"
  fi
  exit 0
fi

echo "==> [release] Refreshing ${default_branch}"
timeout -k 120s -s SIGKILL 120s gix cd "${default_branch}"

echo "==> [release] Re-running preflight after refresh"
"${HELPER}" preflight --default-branch "${default_branch}" --release-timestamp "${release_timestamp}" | tee "${preflight_json}"

echo "==> [release] Running make ci"
timeout -k 1200s -s SIGKILL 1200s make ci

next_version="$(select_version "${preflight_json}")"
boundary_tag="$(select_changelog_boundary "${preflight_json}" "${next_version}")"
if [[ -n "${boundary_tag}" ]]; then
  boundary_label="${boundary_tag}"
else
  boundary_label="none"
fi
echo "==> [release] Selected ${next_version} (boundary: ${boundary_label})"

if [[ -n "${boundary_tag}" ]]; then
  timeout -k 120s -s SIGKILL 120s gix message changelog --since-tag "${boundary_tag}" --version "${next_version}" --release-date "${release_date}" | tee "${notes_file}"
else
  timeout -k 120s -s SIGKILL 120s gix message changelog --version "${next_version}" --release-date "${release_date}" | tee "${notes_file}"
fi

"${HELPER}" insert-changelog --notes-file "${notes_file}"

git add CHANGELOG.md
if git diff --cached --quiet -- CHANGELOG.md; then
  echo "error: CHANGELOG.md has no staged release changes" >&2
  exit 1
fi

git commit -m "Release ${next_version}"
release_commit="$(git rev-parse HEAD)"
git push origin "${default_branch}"

timeout -k 120s -s SIGKILL 120s gix release "${next_version}"
"${HELPER}" publish-release --version "${next_version}" --notes-file "${notes_file}"

verify_args=(verify-release --version "${next_version}" --release-commit "${release_commit}" --notes-file "${notes_file}" --default-branch "${default_branch}")
if [[ "${SKIP_PAGES_VERIFY}" == "true" ]]; then
  verify_args+=(--skip-pages)
fi
"${HELPER}" "${verify_args[@]}"

git fetch --tags origin
echo "Released ${next_version} at ${release_commit}"
