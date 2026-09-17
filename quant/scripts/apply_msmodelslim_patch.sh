#!/usr/bin/env bash
# apply_msmodelslim_patch.sh — apply the DeepSeek-V4.1 W4A8 msmodelslim patch set.
#
# Usage:
#   scripts/apply_msmodelslim_patch.sh --dir /path/to/msmodelslim [--check] [--with-hiaux] [--no-layout]
#
# The target must be a clean msmodelslim checkout whose HEAD is the documented
# upstream base commit (or a descendant that still contains the patched files
# unchanged).  Patches are applied with `git apply -p1`; every patch is
# `--check`ed first and the script aborts on the first mismatch.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_DIR=${PATCH_DIR:-$HERE/../patches}
BASE_COMMIT=92e219fa9565a5bad84d90474a27bb11524d691c
MAIN_PATCH=$PATCH_DIR/msmodelslim_v41_w4a8.patch
HIAUX_PATCH=$PATCH_DIR/msmodelslim_v41_w4a8_hiaux_optional.patch

REPO=${MSMODELSLIM_REPO:-}
WITH_HIAUX=0
CHECK_ONLY=0
MAKE_LAYOUT=1

usage() {
  sed -n '2,8p' "$0"
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dir) REPO=$2; shift 2 ;;
    --with-hiaux) WITH_HIAUX=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --no-layout) MAKE_LAYOUT=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[ -n "$REPO" ] || { echo "error: --dir <msmodelslim checkout> or MSMODELSLIM_REPO is required" >&2; exit 2; }
[ -f "$MAIN_PATCH" ] || { echo "error: patch not found: $MAIN_PATCH" >&2; exit 2; }
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "error: not a git checkout: $REPO" >&2; exit 2; }

head_commit=$(git -C "$REPO" rev-parse HEAD)
if [ "$head_commit" != "$BASE_COMMIT" ]; then
  echo "warning: HEAD=$head_commit, documented base is $BASE_COMMIT" >&2
  echo "warning: continuing; 'git apply --check' will fail if the trees differ" >&2
fi

apply_one() {
  local patch=$1 label=$2
  echo "== $label =="
  echo "   $(md5sum "$patch" | awk '{print $1}')  $(basename "$patch")"
  git -C "$REPO" apply --check -p1 "$patch"
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "   check OK (not applied)"
  else
    git -C "$REPO" apply -p1 "$patch"
    echo "   applied"
  fi
}

apply_one "$MAIN_PATCH" "main patch (adapter + recipe + config entry)"
if [ "$WITH_HIAUX" = "1" ]; then
  [ -f "$HIAUX_PATCH" ] || { echo "error: optional patch not found: $HIAUX_PATCH" >&2; exit 2; }
  apply_one "$HIAUX_PATCH" "optional hi-aux recipe (layers 37-39 routed experts -> W8A8)"
fi

if [ "$CHECK_ONLY" = "0" ] && [ "$MAKE_LAYOUT" = "1" ]; then
  # msmodelslim is installed editable and looks for config/lab_practice/lab_calib
  # inside the package; the repo keeps them at the root.  Same effect as the
  # machine-local scripts/prepare_mslim_layout.sh, but relative to $REPO.
  echo "== editable-install layout =="
  mkdir -p "$REPO/msmodelslim/config"
  cp -f "$REPO/config/config.ini" "$REPO/msmodelslim/config/config.ini"
  ln -sfn ../config "$REPO/msmodelslim/config_repo"
  [ -e "$REPO/msmodelslim/lab_practice" ] || ln -s ../lab_practice "$REPO/msmodelslim/lab_practice"
  [ -e "$REPO/msmodelslim/lab_calib" ] || ln -s ../lab_calib "$REPO/msmodelslim/lab_calib"
  ls -l "$REPO/msmodelslim/config_repo" "$REPO/msmodelslim/lab_practice" "$REPO/msmodelslim/lab_calib"
  echo "   copied config/config.ini -> msmodelslim/config/config.ini"
fi

if [ "$CHECK_ONLY" = "0" ]; then
  echo "[ok] patch set applied to $REPO"
  git -C "$REPO" status --short | head -30
else
  echo "[ok] all patches pass 'git apply --check' on $REPO"
fi
