# %% [markdown]
# # HushNet proof of concept: train a tiny forest-sound classifier and export it for microcontrollers
#
# What this notebook does, top to bottom:
#
# 1. Downloads **ESC-50** (2000 five-second clips, 50 classes, 44.1 kHz mono WAV).
# 2. Keeps four classes: `chainsaw`, `fire` (crackling fire), `birds` (chirping birds) and
#    `background` (rain + wind).
# 3. Cuts 1-second windows at 16 kHz and computes **log-Mel features** (49 frames x 40 bands)
#    with plain NumPy. The firmware uses the same maths and the same tables, exported from here.
# 4. Trains a small CNN (about 4 000 parameters), converts it to **int8 TensorFlow Lite**
#    and tests the int8 model with the same reference kernels that the microcontroller uses.
# 5. Writes three header files for the Arduino sketch:
#    `hn_model_data.h`, `hn_feature_params.h`, `hn_replay_clips.h`.
#
# Run every cell in order (Runtime > Run all). It takes about 5 to 10 minutes on a free Colab CPU.
#
# **Data split.** ESC-50 comes with 5 folds, and clips cut from the same original recording
# always share a fold. Folds 1 to 4 are used for training (and for cross-validation when
# choosing the model), fold 5 is the test set and is never used for any decision.
#
# **Dataset licence.** ESC-50 is CC BY-NC 3.0 (non-commercial). Fine for a thesis or a course
# project, not for a product. Cite: K. J. Piczak, "ESC: Dataset for Environmental Sound
# Classification", ACM Multimedia 2015, doi:10.1145/2733373.2806390.

# %%
import os, sys, csv, json, time, subprocess
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sps

DATA_DIR = os.environ.get("ESC50_DIR", "ESC-50")              # dataset folder (created by the next cell)
OUT_DIR = os.environ.get("HUSHNET_OUT", "hushnet_export")     # exported files go here
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 7
np.random.seed(SEED)

# %% [markdown]
# ## 1. Download ESC-50 (about 600 MB of audio)

# %%
if not os.path.isdir(os.path.join(DATA_DIR, "audio")):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/karolpiczak/ESC-50.git", DATA_DIR], check=True)
meta = list(csv.DictReader(open(os.path.join(DATA_DIR, "meta", "esc50.csv"))))
print(len(meta), "clips listed in meta/esc50.csv")

# %% [markdown]
# ## 2. Settings shared by training and firmware
#
# Change values only here. The feature settings are exported to `hn_feature_params.h`,
# so the firmware follows automatically.

# %%
SR = 16000            # sample rate used by the node (Hz)
WIN_SAMPLES = 16000   # one classification window = 1.0 s
N_FFT = 512           # 32 ms analysis frame (must be a power of two)
HOP = 320             # 20 ms hop between frames
N_MELS = 40
FMIN, FMAX = 50.0, 8000.0
LOG_EPS = 1e-6
N_FRAMES = 1 + (WIN_SAMPLES - N_FFT) // HOP   # 49

# HushNet class -> ESC-50 categories
CLASS_MAP = {
    "background": ["rain", "wind"],
    "chainsaw":   ["chainsaw"],
    "fire":       ["crackling_fire"],
    "birds":      ["chirping_birds"],
}
CLASS_NAMES = list(CLASS_MAP.keys())
N_CLASSES = len(CLASS_NAMES)
TRAIN_FOLDS, TEST_FOLDS = {"1", "2", "3", "4"}, {"5"}

# Model and training settings. They were chosen by 4-fold cross-validation on folds 1 to 4
# (window accuracy, mean +- standard deviation over the folds, 40 epochs each):
#   width (16, 32, 64), no mixup      4 036 parameters   0.890 +- 0.022
#   width (24, 48, 96), no mixup      7 972 parameters   0.899 +- 0.019
#   width (16, 32, 64), mixup 0.3     4 036 parameters   0.898 +- 0.009   <- used here
#   width (24, 48, 96), mixup 0.3,    7 972 parameters   0.909 +- 0.011   (about 2.2 x the compute)
#     dropout 0.4, label smoothing 0.05
# The differences are within about one standard deviation, so the smallest model is kept.
# Set RUN_CV = True to repeat the cross-validation for the current settings (adds about 4 minutes).
MODEL_WIDTH = (16, 32, 64)
DROPOUT = 0.3
MIXUP_ALPHA = 0.3
LABEL_SMOOTHING = 0.0
EPOCHS = int(os.environ.get("HUSHNET_EPOCHS", "40"))
RUN_CV = os.environ.get("HUSHNET_CV", "0") == "1"
print("frames per window:", N_FRAMES, "| classes:", CLASS_NAMES)

# %% [markdown]
# ## 3. Feature extraction (the firmware file `hn_features.h` mirrors this exactly)

# %%
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

def make_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Triangular filters on the HTK mel scale. Shape (n_mels, n_fft//2 + 1), float32."""
    n_bins = n_fft // 2 + 1
    edges_hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    edges_bin = edges_hz * n_fft / sr
    k = np.arange(n_bins, dtype=np.float64)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, ce, hi = edges_bin[m], edges_bin[m + 1], edges_bin[m + 2]
        fb[m] = np.maximum(0.0, np.minimum((k - lo) / (ce - lo), (hi - k) / (hi - ce)))
    return fb.astype(np.float32)

HANN = np.hanning(N_FFT).astype(np.float32)
MEL_FB = make_mel_filterbank(SR, N_FFT, N_MELS, FMIN, FMAX)
assert (MEL_FB.sum(axis=1) > 0).all(), "a mel filter is empty, change N_MELS or FMIN"

def logmel(window_i16):
    """int16 window of WIN_SAMPLES -> float32 log-Mel (N_FRAMES, N_MELS), mean removed."""
    x = window_i16.astype(np.float32) / 32768.0
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    frames = x[idx] * HANN[None, :]
    spec = np.fft.rfft(frames.astype(np.float32), axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = power @ MEL_FB.T
    f = np.log(mel + np.float32(LOG_EPS)).astype(np.float32)
    f -= f.mean(dtype=np.float32)           # gain invariance: remove the window mean
    return f

def quantize(feat_norm, scale, zero_point):
    """Float features (already divided by FEAT_STD) -> int8, same rounding as the firmware."""
    q = np.floor(feat_norm / np.float32(scale) + np.float32(0.5)) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)

# %% [markdown]
# ## 4. Load clips, cut windows, compute features

# %%
def load_clip_16k(path):
    sr, d = wavfile.read(path)
    if d.ndim > 1:
        d = d[:, 0]
    y = sps.resample_poly(d.astype(np.float32), SR, sr)   # 44.1 kHz -> 16 kHz
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)

def is_silent(w):
    # ESC-50 pads short recordings with digital silence. Skip those windows.
    return (np.mean(w == 0) > 0.5) or (np.sqrt(np.mean((w.astype(np.float32) / 32768.0) ** 2)) < 1e-4)

cat_to_class = {c: i for i, name in enumerate(CLASS_NAMES) for c in CLASS_MAP[name]}
X = {"train": [], "test": []}
Y = {"train": [], "test": []}
FOLD = {"train": [], "test": []}
CLIP_ID = {"train": [], "test": []}
HALF = {"train": [], "test": []}     # True for windows that start on a 0.5 s grid
test_audio = []                      # (class, file, start, int16 window) kept for the replay export
t0 = time.time()
for row in meta:
    if row["category"] not in cat_to_class:
        continue
    split = "train" if row["fold"] in TRAIN_FOLDS else "test"
    clip = load_clip_16k(os.path.join(DATA_DIR, "audio", row["filename"]))
    hop = WIN_SAMPLES // 4 if split == "train" else WIN_SAMPLES // 2   # denser windows for training
    for start in range(0, len(clip) - WIN_SAMPLES + 1, hop):
        w = clip[start:start + WIN_SAMPLES]
        if is_silent(w):
            continue
        X[split].append(logmel(w))
        Y[split].append(cat_to_class[row["category"]])
        FOLD[split].append(int(row["fold"]))
        CLIP_ID[split].append(row["filename"])
        HALF[split].append(start % (WIN_SAMPLES // 2) == 0)
        if split == "test":
            test_audio.append((cat_to_class[row["category"]], row["filename"], start, w.copy()))
for s in X:
    X[s] = np.stack(X[s]).astype(np.float32)
    Y[s] = np.array(Y[s], dtype=np.int64)
    FOLD[s] = np.array(FOLD[s]); HALF[s] = np.array(HALF[s])
    print(f"{s:5s}: {X[s].shape[0]:5d} windows from {len(set(CLIP_ID[s])):3d} clips | per class:",
          np.bincount(Y[s], minlength=N_CLASSES).tolist())
print("feature time: %.1f s" % (time.time() - t0))

# One global scale so the network sees values of about unit size.
FEAT_STD = float(X["train"].std())
print("FEAT_STD =", FEAT_STD)
for s in X:
    X[s] = (X[s] / np.float32(FEAT_STD))[..., None]      # (N, 49, 40, 1)

# %%
# Optional picture: one log-Mel window per class (handy for slides). Skipped if matplotlib is missing.
try:
    import matplotlib
    if "google.colab" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(3.2 * N_CLASSES, 3))
    for c, ax in enumerate(axes):
        i = int(np.nonzero(Y["test"] == c)[0][4])
        ax.imshow(X["test"][i, :, :, 0].T, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(CLASS_NAMES[c]); ax.set_xlabel("frame (20 ms)"); ax.set_ylabel("Mel band")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "feature_examples.png"), dpi=130)
    plt.show()
except Exception as e:
    print("no figure:", e)

# %% [markdown]
# ## 5. Model: a small depthwise-separable CNN

# %%
import tensorflow as tf
print("TensorFlow", tf.__version__)
L = tf.keras.layers

def conv_bn_relu(x, filters, kernel, strides=1):
    x = L.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x)

def dw_bn_relu(x):
    x = L.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    return L.ReLU()(x)

def build_model(width=MODEL_WIDTH, dropout=DROPOUT, batch_size=None):
    inp = L.Input((N_FRAMES, N_MELS, 1), batch_size=batch_size, name="logmel")
    x = conv_bn_relu(inp, width[0], 3, strides=2)          # 25 x 20 x w0
    x = dw_bn_relu(x); x = conv_bn_relu(x, width[1], 1)    # 25 x 20 x w1
    x = L.MaxPooling2D(2)(x)                               # 12 x 10 x w1
    x = dw_bn_relu(x); x = conv_bn_relu(x, width[2], 1)    # 12 x 10 x w2
    x = L.AveragePooling2D((x.shape[1], x.shape[2]))(x)    # 1 x 1 x w2
    x = L.Dropout(dropout)(x)
    x = L.Conv2D(N_CLASSES, 1)(x)                          # 1 x 1 x classes
    x = L.Reshape((N_CLASSES,))(x)
    return tf.keras.Model(inp, L.Softmax()(x))

build_model().summary()

# %% [markdown]
# ## 6. Train
#
# Augmentation on the features: random time and frequency masks, a little noise, and mixup.

# %%
def augment(xb, yb, rng):
    xb = xb.copy()
    for i in range(len(xb)):
        t = rng.integers(0, 8); t_start = rng.integers(0, N_FRAMES - t + 1)
        f = rng.integers(0, 6); f_start = rng.integers(0, N_MELS - f + 1)
        xb[i, t_start:t_start + t, :, 0] = 0.0
        xb[i, :, f_start:f_start + f, 0] = 0.0
    xb = xb + rng.normal(0, 0.05, xb.shape).astype(np.float32)
    y1h = np.eye(N_CLASSES, dtype=np.float32)[yb]
    if MIXUP_ALPHA > 0:
        lam = rng.beta(MIXUP_ALPHA, MIXUP_ALPHA, size=len(xb)).astype(np.float32)
        p = rng.permutation(len(xb))
        xb = lam[:, None, None, None] * xb + (1 - lam[:, None, None, None]) * xb[p]
        y1h = lam[:, None] * y1h + (1 - lam[:, None]) * y1h[p]
    return xb, y1h

def train_model(x_train, y_train, seed=SEED, epochs=EPOCHS, verbose=True):
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    m = build_model()
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING))
    counts = np.bincount(y_train, minlength=N_CLASSES)
    sample_w = (len(y_train) / (N_CLASSES * counts))[y_train].astype(np.float32)   # balance the classes
    for epoch in range(epochs):
        if epoch == int(epochs * 0.6):
            m.optimizer.learning_rate = 5e-4
        if epoch == int(epochs * 0.85):
            m.optimizer.learning_rate = 1e-4
        xa, ya = augment(x_train, y_train, rng)
        h = m.fit(xa, ya, batch_size=32, epochs=1, shuffle=True, sample_weight=sample_w, verbose=0)
        if verbose and (epoch % 5 == 4 or epoch == epochs - 1):
            acc = float(np.mean(np.argmax(m.predict(x_train, verbose=0), axis=1) == y_train))
            print(f"epoch {epoch + 1:3d}  loss {h.history['loss'][0]:.3f}  accuracy on clean training windows {acc:.3f}")
    return m

if RUN_CV:
    accs = []
    for vf in (1, 2, 3, 4):
        tr = FOLD["train"] != vf
        va = (FOLD["train"] == vf) & HALF["train"]
        mcv = train_model(X["train"][tr], Y["train"][tr], verbose=False)
        accs.append(float(np.mean(np.argmax(mcv.predict(X["train"][va], verbose=0), axis=1) == Y["train"][va])))
        print("cross-validation, held-out fold", vf, "window accuracy %.3f" % accs[-1])
    print("cross-validation mean %.3f +- %.3f" % (np.mean(accs), np.std(accs)))

model = train_model(X["train"], Y["train"])

# %% [markdown]
# ## 7. Convert to int8 TensorFlow Lite
#
# Full integer quantisation with int8 input and output. The batch size is fixed to 1 so the
# converted graph only contains operations that TensorFlow Lite Micro has.

# %%
def representative_data():
    idx = np.random.default_rng(SEED).permutation(len(X["train"]))[:400]
    for i in idx:
        yield [X["train"][i:i + 1]]

export_model = build_model(batch_size=1)          # same network with the batch size fixed to 1
export_model.set_weights(model.get_weights())
conv = tf.lite.TFLiteConverter.from_keras_model(export_model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_model = conv.convert()
open(os.path.join(OUT_DIR, "hushnet_int8.tflite"), "wb").write(tflite_model)
print("int8 model size: %d bytes (%.1f KB)" % (len(tflite_model), len(tflite_model) / 1024))

# Interpreter with the reference kernels: the same integer arithmetic as TensorFlow Lite Micro,
# so the scores computed here can be compared one to one with the microcontroller.
# (The default interpreter uses optimised kernels whose int8 results can differ by 1.)
def make_interpreter(model_bytes):
    candidates = []
    try:                                    # newer installs ship the interpreter as LiteRT
        from ai_edge_litert import interpreter as litert
        candidates.append((litert.Interpreter, litert.OpResolverType))
    except Exception:
        pass
    try:
        candidates.append((tf.lite.Interpreter, tf.lite.experimental.OpResolverType))
    except Exception:
        pass
    for use_reference in (True, False):
        for cls, resolver in candidates:
            try:
                kw = {"experimental_op_resolver_type": resolver.BUILTIN_REF} if use_reference else {}
                it = cls(model_content=model_bytes, **kw)
                it.allocate_tensors()
                return it, use_reference
            except Exception as e:
                last_error = e
    raise RuntimeError("no TensorFlow Lite interpreter available: %r" % (last_error,))

interp, REF_KERNELS = make_interpreter(tflite_model)
print("interpreter uses reference kernels:", REF_KERNELS,
      "" if REF_KERNELS else "(scores may differ by 1 from the microcontroller)")
in_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
IN_SCALE, IN_ZP = float(in_d["quantization"][0]), int(in_d["quantization"][1])
OUT_SCALE, OUT_ZP = float(out_d["quantization"][0]), int(out_d["quantization"][1])
print("input :", in_d["shape"], in_d["dtype"].__name__, "scale", IN_SCALE, "zero point", IN_ZP)
print("output:", out_d["shape"], out_d["dtype"].__name__, "scale", OUT_SCALE, "zero point", OUT_ZP)
try:
    print("operations:", sorted({o["op_name"] for o in interp._get_ops_details()}))
except Exception:
    pass

def run_int8(q):
    """q: int8 array (49, 40) -> int8 scores (N_CLASSES,)"""
    interp.set_tensor(in_d["index"], q.reshape(1, N_FRAMES, N_MELS, 1))
    interp.invoke()
    return interp.get_tensor(out_d["index"])[0].copy()

# %% [markdown]
# ## 8. Test the float model and the int8 model on the held-out fold 5

# %%
def report(name, y_true, y_pred, clip_ids):
    acc = float(np.mean(y_true == y_pred))
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\n{name}: window accuracy {acc:.3f} on {len(y_true)} windows")
    print("confusion matrix (rows = true, columns = predicted):")
    print("            " + " ".join(f"{n[:10]:>10s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n[:10]:>10s}  " + " ".join(f"{v:10d}" for v in cm[i]))
    # clip level: majority vote over the windows of each 5 s clip
    votes = {}
    for cid, t, p in zip(clip_ids, y_true, y_pred):
        votes.setdefault(cid, [t, np.zeros(N_CLASSES, int)])[1][p] += 1
    clip_acc = float(np.mean([v[0] == int(np.argmax(v[1])) for v in votes.values()]))
    print(f"clip accuracy (majority vote per 5 s clip): {clip_acc:.3f} on {len(votes)} clips")
    return acc, clip_acc, cm

pred_float = np.argmax(model.predict(X["test"], verbose=0), axis=1)
acc_f, clip_f, _ = report("float model", Y["test"], pred_float, CLIP_ID["test"])
Xq_test = quantize(X["test"][..., 0], IN_SCALE, IN_ZP)
pred_int8 = np.array([int(np.argmax(run_int8(q))) for q in Xq_test])
acc_q, clip_q, cm_q = report("int8 model", Y["test"], pred_int8, CLIP_ID["test"])

# %% [markdown]
# ## 9. Export the three header files for the firmware

# %%
def cf(v):
    """float -> C float literal (always with a decimal point or an exponent)"""
    s = "%.9g" % float(v)
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"

def c_array(values, fmt, per_line=12):
    vals = [(fmt(v) if callable(fmt) else fmt % v) for v in values]
    return ",\n".join("  " + ", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line))

# ---- hn_model_data.h
with open(os.path.join(OUT_DIR, "hn_model_data.h"), "w") as f:
    f.write("// Generated by hushnet_train. int8 TensorFlow Lite model.\n#pragma once\n")
    f.write("const unsigned int g_model_len = %d;\n" % len(tflite_model))
    f.write("alignas(16) const unsigned char g_model[] = {\n")
    f.write(c_array(tflite_model, "0x%02x", 16))
    f.write("\n};\n")

# ---- hn_feature_params.h  (sparse mel filterbank: for each band, first bin, length, weights)
starts, lens, weights = [], [], []
for m in range(N_MELS):
    nz = np.nonzero(MEL_FB[m])[0]
    starts.append(int(nz[0])); lens.append(int(nz[-1] - nz[0] + 1))
    weights.extend(MEL_FB[m, nz[0]:nz[-1] + 1].tolist())
with open(os.path.join(OUT_DIR, "hn_feature_params.h"), "w") as f:
    f.write("// Generated by hushnet_train. Keep in sync with hn_model_data.h (same training run).\n#pragma once\n")
    f.write("#define HN_SR %d\n#define HN_WIN_SAMPLES %d\n#define HN_N_FFT %d\n#define HN_HOP %d\n"
            % (SR, WIN_SAMPLES, N_FFT, HOP))
    f.write("#define HN_N_FRAMES %d\n#define HN_N_MELS %d\n#define HN_N_CLASSES %d\n"
            % (N_FRAMES, N_MELS, N_CLASSES))
    f.write("#define HN_LOG_EPS %s\n#define HN_FEAT_STD %s\n" % (cf(LOG_EPS), cf(FEAT_STD)))
    f.write("#define HN_IN_SCALE %s\n#define HN_IN_ZP %d\n#define HN_OUT_SCALE %s\n#define HN_OUT_ZP %d\n"
            % (cf(IN_SCALE), IN_ZP, cf(OUT_SCALE), OUT_ZP))
    f.write("#define HN_MEL_NWEIGHTS %d\n" % len(weights))
    f.write("static const char* const HN_CLASS_NAMES[HN_N_CLASSES] = { %s };\n"
            % ", ".join('"%s"' % n for n in CLASS_NAMES))
    f.write("static const float HN_HANN[HN_N_FFT] = {\n" + c_array(HANN, cf, 8) + "\n};\n")
    f.write("static const unsigned short HN_MEL_START[HN_N_MELS] = {\n" + c_array(starts, "%d", 20) + "\n};\n")
    f.write("static const unsigned short HN_MEL_LEN[HN_N_MELS] = {\n" + c_array(lens, "%d", 20) + "\n};\n")
    f.write("static const float HN_MEL_W[HN_MEL_NWEIGHTS] = {\n" + c_array(weights, cf, 8) + "\n};\n")

# ---- hn_replay_clips.h : two test-fold windows per class, with what the PC computed for them.
# The clips are picked by a fixed rule (first two files of each class, window at 2.0 s),
# not by how well the model does on them.
PER_CLASS = 2
replay = []
for c in range(N_CLASSES):
    seen = set()
    for (cls, fname, start, w) in test_audio:
        if cls == c and fname not in seen and start == 2 * WIN_SAMPLES:
            seen.add(fname); replay.append((cls, fname, start, w))
            if len(seen) == PER_CLASS:
                break
n_rep = len(replay)
rep_feat = [quantize(logmel(w) / np.float32(FEAT_STD), IN_SCALE, IN_ZP) for (_, _, _, w) in replay]
rep_out = [run_int8(q) for q in rep_feat]
with open(os.path.join(OUT_DIR, "hn_replay_clips.h"), "w") as f:
    f.write("// Generated by hushnet_train. %d one-second test windows from ESC-50 fold 5 (CC BY-NC 3.0).\n" % n_rep)
    f.write("// Each has: raw 16 kHz int16 audio, true class, and the PC result (int8 features, scores).\n#pragma once\n")
    f.write("#define HN_N_REPLAY %d\n" % n_rep)
    f.write("static const unsigned char HN_REPLAY_TRUE[HN_N_REPLAY] = { %s };\n"
            % ", ".join(str(r[0]) for r in replay))
    f.write("static const unsigned char HN_REPLAY_PC_PRED[HN_N_REPLAY] = { %s };\n"
            % ", ".join(str(int(np.argmax(o))) for o in rep_out))
    f.write("static const char* const HN_REPLAY_SRC[HN_N_REPLAY] = { %s };\n"
            % ", ".join('"%s@%.1fs"' % (r[1], r[2] / SR) for r in replay))
    f.write("static const signed char HN_REPLAY_PC_SCORES[HN_N_REPLAY][HN_N_CLASSES] = {\n")
    f.write(",\n".join("  { " + ", ".join(str(int(v)) for v in o) + " }" for o in rep_out) + "\n};\n")
    f.write("static const signed char HN_REPLAY_PC_FEAT[HN_N_REPLAY][HN_N_FRAMES * HN_N_MELS] = {\n")
    f.write(",\n".join("{\n" + c_array(q.reshape(-1), "%d", 24) + "\n}" for q in rep_feat) + "\n};\n")
    f.write("static const short HN_REPLAY_AUDIO[HN_N_REPLAY][HN_WIN_SAMPLES] = {\n")
    f.write(",\n".join("{\n" + c_array(r[3], "%d", 16) + "\n}" for r in replay) + "\n};\n")

print("\nreplay clips (true class -> PC prediction, PC confidence):")
for r, o in zip(replay, rep_out):
    conf = (int(np.max(o)) - OUT_ZP) * OUT_SCALE
    print(f"  {r[1]:22s} @ {r[2] / SR:.1f} s  {CLASS_NAMES[r[0]]:>10s} -> {CLASS_NAMES[int(np.argmax(o))]:<10s} {conf:.2f}")

summary = {
    "tensorflow": tf.__version__, "classes": CLASS_NAMES,
    "windows": {s: int(len(Y[s])) for s in Y},
    "model_width": list(MODEL_WIDTH), "mixup_alpha": MIXUP_ALPHA, "epochs": EPOCHS,
    "model_bytes": len(tflite_model), "params": int(model.count_params()), "reference_kernels": bool(REF_KERNELS),
    "test_window_acc_float": round(acc_f, 4), "test_clip_acc_float": round(clip_f, 4),
    "test_window_acc_int8": round(acc_q, 4), "test_clip_acc_int8": round(clip_q, 4),
    "confusion_int8": cm_q.tolist(), "feat_std": FEAT_STD,
    "in_scale": IN_SCALE, "in_zp": IN_ZP, "out_scale": OUT_SCALE, "out_zp": OUT_ZP,
}
json.dump(summary, open(os.path.join(OUT_DIR, "training_summary.json"), "w"), indent=1)
print("\nfiles written to", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))

# %% [markdown]
# ## 10. Download the files (Colab)
#
# Copy `hn_model_data.h`, `hn_feature_params.h` and `hn_replay_clips.h` into the sketch folder
# `firmware/hushnet_poc/`, replacing the ones that came with the project. Always copy all
# three together: they belong to the same training run.

# %%
try:
    from google.colab import files
    import shutil
    shutil.make_archive("hushnet_export", "zip", OUT_DIR)
    files.download("hushnet_export.zip")
except Exception:
    print("Not on Colab: the files are in", os.path.abspath(OUT_DIR))
