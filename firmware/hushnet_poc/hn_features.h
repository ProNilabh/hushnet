// Log-Mel features on the microcontroller, same maths as the training notebook:
//   1 s of 16 kHz audio -> 49 frames (512 point Hann window, hop 320) -> power spectrum
//   -> 40 triangular Mel bands -> natural log -> subtract the mean of the window
//   -> divide by the training set standard deviation -> quantise to the int8 input of the model.
// Plain C++ with single precision floats and no library, so the same file runs on both boards
// (and on a PC, which is how it is checked against NumPy: tools/host_test).
#pragma once
#include <math.h>
#include <stdint.h>
#include "hn_feature_params.h"

namespace hn {

static float    s_cos[HN_N_FFT / 2];
static float    s_sin[HN_N_FFT / 2];
static uint16_t s_rev[HN_N_FFT];
static float    s_re[HN_N_FFT];
static float    s_im[HN_N_FFT];
static float    s_logmel[HN_N_FRAMES * HN_N_MELS];
static bool     s_featReady = false;

inline void featuresInit() {
  for (int k = 0; k < HN_N_FFT / 2; k++) {
    double a = -2.0 * 3.14159265358979323846 * (double)k / (double)HN_N_FFT;
    s_cos[k] = (float)cos(a);
    s_sin[k] = (float)sin(a);
  }
  int bits = 0;
  while ((1 << bits) < HN_N_FFT) bits++;
  for (int i = 0; i < HN_N_FFT; i++) {
    unsigned r = 0;
    for (int b = 0; b < bits; b++) {
      if (i & (1 << b)) r |= 1u << (bits - 1 - b);
    }
    s_rev[i] = (uint16_t)r;
  }
  s_featReady = true;
}

// In place radix-2 FFT of s_re/s_im. The input must already be in bit reversed order.
inline void fftRun() {
  // first stage: the twiddle factor is 1, so no multiplications
  for (int i = 0; i < HN_N_FFT; i += 2) {
    float ar = s_re[i], ai = s_im[i], br = s_re[i + 1], bi = s_im[i + 1];
    s_re[i] = ar + br;
    s_im[i] = ai + bi;
    s_re[i + 1] = ar - br;
    s_im[i + 1] = ai - bi;
  }
  for (int len = 4; len <= HN_N_FFT; len <<= 1) {
    const int half = len >> 1;
    const int step = HN_N_FFT / len;
    for (int i = 0; i < HN_N_FFT; i += len) {
      for (int j = 0, k = 0; j < half; j++, k += step) {
        const int a = i + j, b = a + half;
        const float wr = s_cos[k], wi = s_sin[k];
        const float xr = s_re[b] * wr - s_im[b] * wi;
        const float xi = s_re[b] * wi + s_im[b] * wr;
        s_re[b] = s_re[a] - xr;
        s_im[b] = s_im[a] - xi;
        s_re[a] += xr;
        s_im[a] += xi;
      }
    }
  }
}

// audio: HN_WIN_SAMPLES int16 samples at HN_SR. out: HN_N_FRAMES * HN_N_MELS int8 values (frame major).
inline void computeFeatures(const int16_t* audio, int8_t* out) {
  if (!s_featReady) featuresInit();
  double sum = 0.0;
  for (int f = 0; f < HN_N_FRAMES; f++) {
    const int16_t* p = audio + f * HN_HOP;
    for (int n = 0; n < HN_N_FFT; n++) {
      const int r = s_rev[n];
      s_re[r] = ((float)p[n] * (1.0f / 32768.0f)) * HN_HANN[n];
      s_im[r] = 0.0f;
    }
    fftRun();
    for (int k = 0; k <= HN_N_FFT / 2; k++) {
      s_re[k] = s_re[k] * s_re[k] + s_im[k] * s_im[k];   // power spectrum, kept in s_re
    }
    const float* w = HN_MEL_W;
    float* row = s_logmel + f * HN_N_MELS;
    for (int m = 0; m < HN_N_MELS; m++) {
      const int start = HN_MEL_START[m], n = HN_MEL_LEN[m];
      float acc = 0.0f;
      for (int i = 0; i < n; i++) acc += s_re[start + i] * w[i];
      w += n;
      const float v = logf(acc + HN_LOG_EPS);
      row[m] = v;
      sum += (double)v;
    }
  }
  const float mean = (float)(sum / (double)(HN_N_FRAMES * HN_N_MELS));
  for (int i = 0; i < HN_N_FRAMES * HN_N_MELS; i++) {
    const float x = (s_logmel[i] - mean) / HN_FEAT_STD;
    float q = floorf(x / HN_IN_SCALE + 0.5f) + (float)HN_IN_ZP;
    if (q < -128.0f) q = -128.0f;
    if (q > 127.0f) q = 127.0f;
    out[i] = (int8_t)q;
  }
}

}  // namespace hn
