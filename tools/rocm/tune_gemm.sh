#!/usr/bin/env bash
# Tune the GEMMs of the set DiTs with PyTorch TunableOp (rocBLAS + hipBLASLt solutions benchmarked per shape on this
# GPU); results go to $HAIRDECK_ROOT/.cache/tunableop/<arch>.csv, read back when PYTORCH_TUNABLEOP_ENABLED=1.
set -euo pipefail
cd "$(dirname "$0")/../.."
dir=${HAIRDECK_ROOT:-/workspace/hairdeck-root}/.cache/tunableop
mkdir -p "$dir"
export PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_VERBOSE=${VERBOSE:-1}
export PYTORCH_TUNABLEOP_FILENAME="$dir/gfx1151.csv"
export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=${MS:-15} PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=5
export PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=1 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=1
# no graph capture while tuning: hipBLASLt cannot benchmark inside a capture
exec timeout "${TIMEOUT:-1500}" python3 tools/rocm/check_fast.py --modes orig nograph
