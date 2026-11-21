// PC check of the firmware inference path: the same hn_features.h and hn_inference.h as on the boards,
// linked against the same TensorFlow Lite Micro library sources (Chirale_TensorFLowLite), built for the PC.
// It classifies the replay clips and compares the int8 scores with the ones stored by the notebook.
// Build and run: see run_host_test.sh
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline uint32_t micros() {
  using namespace std::chrono;
  return (uint32_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

#include "hn_inference.h"

// Platform hooks that the library normally gets from its Arduino glue code.
extern "C" void DebugLog(const char* s) { fputs(s, stderr); }
namespace peripherals {
uint32_t MicrosecondsCounter() { return micros(); }
}  // namespace peripherals

int main() {
  const char* err = hn::mlBegin();
  if (err) {
    printf("mlBegin failed: %s\n", err);
    return 2;
  }
  printf("model %u bytes, arena used %u of %u bytes\n", (unsigned)g_model_len, (unsigned)hn::mlArenaUsed(),
         (unsigned)hn::kArenaSize);
  int classMismatch = 0, worstScore = 0, correct = 0;
  for (int c = 0; c < HN_N_REPLAY; c++) {
    hn::InferResult r;
    if (!hn::classify(HN_REPLAY_AUDIO[c], r)) {
      printf("clip %d: interpreter failed\n", c);
      return 3;
    }
    int sd = 0;
    for (int i = 0; i < HN_N_CLASSES; i++) {
      int d = abs((int)r.scores[i] - (int)HN_REPLAY_PC_SCORES[c][i]);
      if (d > sd) sd = d;
    }
    if (sd > worstScore) worstScore = sd;
    if (r.cls != HN_REPLAY_PC_PRED[c]) classMismatch++;
    if (r.cls == HN_REPLAY_TRUE[c]) correct++;
    printf("clip %d true %-10s here %-10s %.3f | notebook %-10s | largest score difference %d | scores", c,
           HN_CLASS_NAMES[HN_REPLAY_TRUE[c]], HN_CLASS_NAMES[r.cls], r.conf, HN_CLASS_NAMES[HN_REPLAY_PC_PRED[c]], sd);
    for (int i = 0; i < HN_N_CLASSES; i++) printf(" %d", r.scores[i]);
    printf("\n");
  }
  printf("correct %d of %d, classes that differ from the notebook %d, largest score difference %d\n", correct, HN_N_REPLAY,
         classMismatch, worstScore);
  return classMismatch ? 1 : 0;
}
