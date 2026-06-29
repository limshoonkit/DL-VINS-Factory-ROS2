#!/usr/bin/env bash
# Build the letvins:noetic Docker image.
# Can be run from any directory; build context is always the VINS-Mono directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VINS_MONO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build \
  -f "$SCRIPT_DIR/Dockerfile.noetic" \
  -t letvins:noetic \
  "$VINS_MONO_DIR"
