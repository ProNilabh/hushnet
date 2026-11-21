// PC check of the device feature code (firmware/hushnet_poc/features.h) against the features
// that the training notebook computed with NumPy and stored in replay_clips.h.
// Build and run:  see tools/host_test/README.txt
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "hn_features.h"
#include "hn_replay_clips.h"

int main() {
  static int8_t feat[HN_N_FRAMES * HN_N_MELS];
  long totalDiff = 0;
  int worst = 0;
  double ms = 0;
  for (int c = 0; c < HN_N_REPLAY; c++) {
    auto t0 = std::chrono::steady_clock::now();
    hn::computeFeatures(HN_REPLAY_AUDIO[c], feat);
    auto t1 = std::chrono::steady_clock::now();
    ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    int nDiff = 0, maxDiff = 0;
    for (int i = 0; i < HN_N_FRAMES * HN_N_MELS; i++) {
      int d = abs((int)feat[i] - (int)HN_REPLAY_PC_FEAT[c][i]);
      if (d) nDiff++;
      if (d > maxDiff) maxDiff = d;
    }
    totalDiff += nDiff;
    if (maxDiff > worst) worst = maxDiff;
    printf("clip %d %-28s differing cells %4d of %d, largest difference %d\n", c, HN_REPLAY_SRC[c], nDiff,
           HN_N_FRAMES * HN_N_MELS, maxDiff);
  }
  printf("total differing cells %ld, largest difference %d, mean time per clip on this PC %.2f ms\n", totalDiff, worst,
         ms / HN_N_REPLAY);
  return worst > 1 ? 1 : 0;
}
