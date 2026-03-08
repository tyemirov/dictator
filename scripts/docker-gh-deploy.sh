#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/docker-gh-deploy.sh [options] [git-tag]

Builds the GPU Docker image locally and pushes it to GHCR.

If no git tag is provided, the script requires HEAD to be checked out at an exact
tag and uses that tag automatically.

Examples:
  ./scripts/docker-gh-deploy.sh v1.2.3
  ./scripts/docker-gh-deploy.sh --dry-run
  ./scripts/docker-gh-deploy.sh --image-name ghcr.io/acme/dictator-gpu v1.2.3

Options:
  --dry-run        Validate inputs and print the Docker tags without building
  --skip-tests     Skip `make ci`
  --skip-image-test Skip the Docker image blackbox smoke test
  --skip-login     Skip `docker login ghcr.io`
  --platform       Override Docker platform (default: linux/amd64)
  --image-name     Override image repo (default: derived from origin remote)
  --builder        Override Buildx builder name (default: dictator-ghcr)
  --cache-dir      Override local Buildx cache dir (default: .buildx-cache-gpu)
  -h, --help       Show this help

Environment:
  GHCR_USERNAME    GitHub username for docker login
  GHCR_TOKEN       GitHub token / PAT for docker login
  DICTATOR_IMAGE   Full GHCR image repo override, e.g. ghcr.io/acme/dictator-gpu
  BUILDX_BUILDER   Buildx builder name override
EOF
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

to_lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

derive_image_name() {
  local remote_url path

  remote_url="$(git remote get-url origin 2>/dev/null || true)"
  case "$remote_url" in
    git@github.com:*)
      path="${remote_url#git@github.com:}"
      ;;
    https://github.com/*)
      path="${remote_url#https://github.com/}"
      ;;
    ssh://git@github.com/*)
      path="${remote_url#ssh://git@github.com/}"
      ;;
    *)
      die "Unable to derive GHCR image name from origin remote '${remote_url}'. Use --image-name or DICTATOR_IMAGE."
      ;;
  esac

  path="${path%.git}"
  printf 'ghcr.io/%s-gpu\n' "$(to_lower "$path")"
}

build_image_tags() {
  local git_tag="$1"
  local image_name="$2"
  local semver_regex='^v?([0-9]+)\.([0-9]+)\.([0-9]+)(-([0-9A-Za-z.-]+))?$'
  local major minor patch prerelease version

  if [[ ! "$git_tag" =~ $semver_regex ]]; then
    die "Git tag '${git_tag}' must be semver like 1.2.3 or v1.2.3. Optional prerelease suffixes such as -rc.1 are supported. Build metadata (+...) is not supported."
  fi

  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"
  prerelease="${BASH_REMATCH[4]}"
  version="${major}.${minor}.${patch}${prerelease}"

  IMAGE_TAGS=(
    "${image_name}:${version}"
  )

  if [[ "$git_tag" != "$version" ]]; then
    IMAGE_TAGS+=("${image_name}:${git_tag}")
  fi

  if [[ -z "$prerelease" ]]; then
    IMAGE_TAGS+=(
      "${image_name}:${major}.${minor}"
      "${image_name}:${major}"
      "${image_name}:latest"
    )
  fi
}

ensure_clean_tree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    die "Working tree must be clean before publishing."
  fi
}

verify_tag_checkout() {
  local git_tag="$1"
  local tag_commit head_commit

  git rev-parse -q --verify "refs/tags/${git_tag}^{commit}" >/dev/null 2>&1 \
    || die "Git tag '${git_tag}' does not exist locally."

  tag_commit="$(git rev-list -n 1 "$git_tag")"
  head_commit="$(git rev-parse HEAD)"

  [[ "$tag_commit" == "$head_commit" ]] \
    || die "HEAD (${head_commit}) does not match tag '${git_tag}' (${tag_commit}). Check out the tagged commit before publishing."

  git fetch --no-tags origin master:refs/remotes/origin/master
  git merge-base --is-ancestor "$tag_commit" origin/master \
    || die "Tag '${git_tag}' resolves to ${tag_commit}, which is not contained in origin/master."

  log "Validated tag '${git_tag}' at commit ${tag_commit} on origin/master."
}

docker_login_ghcr() {
  local username token

  if [[ -n "${GHCR_TOKEN:-}" ]]; then
    [[ -n "${GHCR_USERNAME:-}" ]] || die "GHCR_USERNAME is required when GHCR_TOKEN is set."
    printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
    return
  fi

  if command -v gh >/dev/null 2>&1; then
    username="$(gh api user --jq .login 2>/dev/null || true)"
    token="$(gh auth token 2>/dev/null || true)"
    if [[ -n "$username" && -n "$token" ]]; then
      printf '%s' "$token" | docker login ghcr.io -u "$username" --password-stdin
      return
    fi
  fi

  log "No GHCR credentials supplied; assuming docker is already logged in to ghcr.io."
}

run_build() {
  local builder_name="$1"
  local cache_dir="$2"
  local platform="$3"
  local new_cache_dir="${cache_dir}-new"
  local cmd=(
    docker buildx build
    --builder "$builder_name"
    --file Dockerfile.gpu
    --platform "$platform"
    --push
    --cache-to "type=local,dest=${new_cache_dir},mode=max"
  )
  local image_tag

  rm -rf "$new_cache_dir"

  if [[ -f "${cache_dir}/index.json" ]]; then
    cmd+=(--cache-from "type=local,src=${cache_dir}")
  else
    log "No existing local Buildx cache at '${cache_dir}'; building cold."
  fi

  for image_tag in "${IMAGE_TAGS[@]}"; do
    cmd+=(--tag "$image_tag")
  done

  cmd+=(.)

  "${cmd[@]}"

  rm -rf "$cache_dir"
  mv "$new_cache_dir" "$cache_dir"
}

ensure_builder() {
  local builder_name="$1"

  if ! docker buildx inspect "$builder_name" >/dev/null 2>&1; then
    log "Creating Buildx builder '${builder_name}' with docker-container driver."
    docker buildx create --name "$builder_name" --driver docker-container >/dev/null
  fi

  docker buildx inspect --bootstrap "$builder_name" >/dev/null
}

DRY_RUN=0
SKIP_TESTS=0
SKIP_IMAGE_TEST=0
SKIP_LOGIN=0
PLATFORM="${PLATFORM:-linux/amd64}"
CACHE_DIR="${CACHE_DIR:-.buildx-cache-gpu}"
IMAGE_NAME="${DICTATOR_IMAGE:-}"
BUILDER_NAME="${BUILDX_BUILDER:-dictator-ghcr}"
GIT_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-tests)
      SKIP_TESTS=1
      ;;
    --skip-image-test)
      SKIP_IMAGE_TEST=1
      ;;
    --skip-login)
      SKIP_LOGIN=1
      ;;
    --platform)
      shift
      [[ $# -gt 0 ]] || die "--platform requires a value."
      PLATFORM="$1"
      ;;
    --image-name)
      shift
      [[ $# -gt 0 ]] || die "--image-name requires a value."
      IMAGE_NAME="$1"
      ;;
    --builder)
      shift
      [[ $# -gt 0 ]] || die "--builder requires a value."
      BUILDER_NAME="$1"
      ;;
    --cache-dir)
      shift
      [[ $# -gt 0 ]] || die "--cache-dir requires a value."
      CACHE_DIR="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      [[ -z "$GIT_TAG" ]] || die "Only one git tag may be supplied."
      GIT_TAG="$1"
      ;;
  esac
  shift
done

if [[ -z "$GIT_TAG" ]]; then
  GIT_TAG="$(git describe --tags --exact-match 2>/dev/null || true)"
  [[ -n "$GIT_TAG" ]] || die "No git tag provided and HEAD is not checked out at an exact tag."
fi

require_cmd git

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  require_cmd make
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  require_cmd docker
  docker buildx version >/dev/null 2>&1 || die "Docker Buildx is required."
fi

ensure_clean_tree
verify_tag_checkout "$GIT_TAG"

if [[ -z "$IMAGE_NAME" ]]; then
  IMAGE_NAME="$(derive_image_name)"
fi

build_image_tags "$GIT_TAG" "$IMAGE_NAME"

log "Image tags to publish:"
printf '  %s\n' "${IMAGE_TAGS[@]}"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  log "Running test suite before publish."
  make ci
fi

if [[ "$SKIP_IMAGE_TEST" -eq 0 ]]; then
  log "Running Docker image blackbox smoke test before publish."
  ./scripts/test-docker-image.sh --tag dictator:publish-blackbox
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run complete; skipping docker login and build."
  exit 0
fi

if [[ "$SKIP_LOGIN" -eq 0 ]]; then
  docker_login_ghcr
fi

ensure_builder "$BUILDER_NAME"

log "Building and pushing GPU image with Buildx builder '${BUILDER_NAME}' and local cache '${CACHE_DIR}'."
run_build "$BUILDER_NAME" "$CACHE_DIR" "$PLATFORM"
log "Publish complete."
