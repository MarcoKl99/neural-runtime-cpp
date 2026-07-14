#!/usr/bin/env bash
set -euo pipefail

rm -rf ./build
cmake -B build -DNRT_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target nrt_tests
ctest --test-dir build --output-on-failure

# Excluded header files as the implementation is tested in the .cpp files and this gives a more meaningful coverage report
gcovr -r . --object-directory build --exclude '.*_deps.*' --exclude '.*tests.*' --exclude '.*examples.*' --exclude '.*\.hpp'
