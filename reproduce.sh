#!/usr/bin/env bash
# Source and datasets are never rewritten by this helper.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
GO="${GO:-go}"
command="${1:-build}"
if (($#)); then shift; fi
cd "$ROOT"
case "$command" in
  build) mkdir -p build; "$GO" build -o build/enc-dashformer . ;;
  test) "$GO" test -p 1 ./... ;;
  check) "$GO" run . --check-inputs "$@" ;;
  run) mkdir -p build; "$GO" build -o build/enc-dashformer .; exec ./build/enc-dashformer "$@" ;;
  *) echo "usage: $0 {build|test|check|run} [application options]" >&2; exit 2 ;;
esac
