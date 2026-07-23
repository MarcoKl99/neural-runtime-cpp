#!/usr/bin/env bash
set -euo pipefail

# Usage: ./benchmarks/benchmark.sh <label>
# Example: ./benchmarks/benchmark.sh v1.1.1
#
# Builds nrt_bench_cnn in a dedicated Release build directory and runs it with
# a fixed statistical protocol, saving the result as a versioned, redacted JSON
# artifact. Note, that the result files are nevertheless on .gitignore.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <label>" >&2
    echo "  <label> becomes the output filename: benchmarks/results/<label>.json" >&2
    exit 1
fi

LABEL="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-release"
RESULTS_DIR="${ROOT_DIR}/benchmarks/results"
OUT_FILE="${RESULTS_DIR}/${LABEL}.json"
RAW_FILE="${OUT_FILE}.raw"

mkdir -p "${RESULTS_DIR}"

if [ -f "${OUT_FILE}" ]; then
    echo "Refusing to overwrite existing result: ${OUT_FILE}" >&2
    exit 1
fi

echo "==> Configuring dedicated Release build in ${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "==> Building nrt_bench_cnn"
cmake --build "${BUILD_DIR}" --target nrt_bench_cnn -j

echo "==> Running benchmarks (10 repetitions) -> ${RAW_FILE}"
"${BUILD_DIR}/nrt_bench_cnn" \
    --benchmark_repetitions=10 \
    --benchmark_display_aggregates_only=true \
    --benchmark_out="${RAW_FILE}" \
    --benchmark_out_format=json

echo "==> Redacting machine-identifying fields -> ${OUT_FILE}"
python3 - "${RAW_FILE}" "${OUT_FILE}" <<'EOF'
import json
import sys

raw_path, out_path = sys.argv[1], sys.argv[2]
with open(raw_path) as f:
    data = json.load(f)

context = data["context"]
del context["executable"]
del context["host_name"]

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
EOF

rm "${RAW_FILE}"

echo "==> Done. Results saved to ${OUT_FILE}"
