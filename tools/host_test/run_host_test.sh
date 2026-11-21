#!/bin/bash
# Optional PC test (Linux or WSL, needs g++): checks the firmware feature code against NumPy and the
# firmware inference path against the notebook, without any board.
#
#   ./run_host_test.sh /path/to/Chirale_TensorFLowLite [/path/to/firmware/hushnet_poc]
#
# -DARDUINO is set because the library selects its kernel variants with that macro, exactly as on the boards.
# The first argument is the folder of the Arduino library (the one that contains "src").
# On Windows the Arduino IDE keeps libraries in Documents/Arduino/libraries; from WSL that is
# /mnt/c/Users/<you>/Documents/Arduino/libraries/Chirale_TensorFLowLite
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
LIB="$1"
FW="${2:-$HERE/../../firmware/hushnet_poc}"
OUT="${OUT:-/tmp/hushnet_host}"
JOBS="${JOBS:-$(nproc)}"
if [ ! -d "$LIB/src/tensorflow" ]; then
  echo "usage: $0 /path/to/Chirale_TensorFLowLite [/path/to/firmware/hushnet_poc]"
  exit 2
fi
mkdir -p "$OUT/obj" "$OUT/stub"
echo "// empty on the PC: the Arduino version of this header pulls in Arduino.h" > "$OUT/stub/Chirale_TensorFlowLite.h"

echo "== 1/2 feature code against NumPy =="
g++ -O2 -std=c++17 -I "$FW" "$HERE/feat_test.cpp" -o "$OUT/feat_test" -lm
"$OUT/feat_test"

echo "== 2/2 inference path against the notebook (building the library for the PC, this takes a few minutes) =="
CFLAGS="-O2 -w -DARDUINO=10800 -DARDUINO_EXCLUDE_CODE -I $LIB/src"
compile_one() {
  src="$1"
  obj="$OUT/obj/$(echo "${src#$LIB/src/}" | tr '/' '_').o"
  [ -f "$obj" ] && [ "$obj" -nt "$src" ] && return 0
  case "$src" in
    *.c) gcc $CFLAGS -std=c11 -c "$src" -o "$obj" ;;
    *)   g++ $CFLAGS -std=c++17 -c "$src" -o "$obj" ;;
  esac
}
export -f compile_one
export LIB OUT CFLAGS
find "$LIB/src/tensorflow" "$LIB/src/third_party" \( -name '*.cpp' -o -name '*.cc' -o -name '*.c' \) \
  | grep -v -e 'system_setup' -e '/kissfft/' \
  | xargs -P "$JOBS" -I{} bash -c 'compile_one "$@"' _ {}
rm -f "$OUT/libtflm.a"
ar rcs "$OUT/libtflm.a" "$OUT"/obj/*.o
g++ -O2 -std=c++17 -w -DARDUINO=10800 -DARDUINO_EXCLUDE_CODE -I "$OUT/stub" -I "$FW" -I "$LIB/src" "$HERE/tflm_test.cpp" "$OUT/libtflm.a" -o "$OUT/tflm_test" -lm
"$OUT/tflm_test"
