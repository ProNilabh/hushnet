// HushNet proof of concept
// A tiny int8 sound classifier (TensorFlow Lite Micro) whose result is sent over raw LoRa (not LoRaWAN).
// One sketch for two boards:
//   Heltec WiFi LoRa 32 V4 (ESP32-S3 + SX1262)   Tools > Board > esp32 > Heltec WiFi LoRa 32(V4)
//   Arduino Nesso N1       (ESP32-C6 + SX1262)   Tools > Board > esp32 > Arduino Nesso N1
// Neither board has a microphone here: "replay mode" classifies test clips stored in flash.
//
// Serial console: 115200 baud, one command per line (send HELP for the list).
// Every output line starts with a tag followed by key=value pairs, so the PC scripts can parse it.

#include "hn_config.h"
#include "hn_board.h"
#include "hn_ui.h"
#include "hn_radio.h"
#if ENABLE_ML
#include "hn_inference.h"
#endif
#include <Preferences.h>
#include <stdarg.h>

// ---------------------------------------------------------------------------
// Packet format (application payload inside the LoRa frame)
//   byte 0     magic 0x48 ('H')
//   byte 1     node id
//   byte 2,3   sequence number, little endian
//   byte 4     type: 0..(classes-1) = sound class, 0xFF = test packet, 0xFE = ping, 0xFD = pong
//   byte 5     class packet: confidence 0..255 | test packet: burst number | pong: RSSI seen by the replier (dBm)
//   byte 6     pong: SNR seen by the replier, in quarter dB | test packet: filler from here to the chosen length
// ---------------------------------------------------------------------------
static const uint8_t PKT_MAGIC = 0x48;
static const uint8_t PKT_TEST  = 0xFF;
static const uint8_t PKT_PING  = 0xFE;
static const uint8_t PKT_PONG  = 0xFD;
static const size_t  PKT_CLASS_LEN = 6;
static const size_t  PKT_PONG_LEN  = 7;

enum class AppMode : uint8_t { Idle = 0, Rx = 1, Node = 2, Beacon = 3 };

struct Band {
  uint32_t lo, hi;        // Hz
  uint16_t dutyTenths;    // duty cycle limit in tenths of a percent
  const char* name;
};

// EU 863-870 MHz short range device sub-bands (ETSI EN 300 220-2, CEPT ERC/REC 70-03 annex 1).
// 865-868 and 868.0-868.6 MHz have the same limit and are merged here.
static const Band BANDS[] = {
  { 863000000UL, 865000000UL, 1,   "863-865 MHz, 0.1 %" },
  { 865000000UL, 868600000UL, 10,  "865-868.6 MHz, 1 %" },
  { 868700000UL, 869200000UL, 1,   "868.7-869.2 MHz, 0.1 %" },
  { 869400000UL, 869650000UL, 100, "869.4-869.65 MHz, 10 %" },
  { 869700000UL, 870000000UL, 10,  "869.7-870 MHz, 1 %" },
};

struct SeqTrack {
  bool     valid;
  uint8_t  burst;
  uint16_t last;
};

struct Saved {
  uint8_t  version;
  uint8_t  mode;
  RadioCfg cfg;
};
static const uint8_t SAVED_VERSION = 3;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
static RadioCfg g_cfg;
static AppMode  g_mode = AppMode::Idle;
static bool     g_radioOk = false;
static bool     g_mlOk = false;
static int16_t  g_radioErr = 0;
static const char* g_mlErr = "built with ENABLE_ML 0";

static uint16_t g_seq = 0;          // sequence number of class and beacon packets
static uint8_t  g_burst = 0;        // number of the current TX burst
static uint8_t  g_clip = 0;         // next replay clip in node mode
static uint32_t g_nextTick = 0;

static uint32_t g_rxOk = 0, g_rxCrc = 0, g_rxOther = 0, g_lost = 0, g_txCount = 0;
static SeqTrack g_track[16][2];     // [node id & 15][0 = class packets, 1 = test packets]

static uint32_t g_dcBins[60];       // transmit time per minute, in ms, for the last hour
static uint32_t g_dcMinute = 0;

static RxPacket g_rx;
static char     g_ui[UI_LINES][UI_COLS + 1];
static char     g_line[200];
static size_t   g_lineLen = 0;

#if defined(BOARD_NESSO_N1) && NESSO_ENABLE_CHARGING
static NessoBattery g_battery;
#endif

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static const char* modeName(AppMode m) {
  switch (m) {
    case AppMode::Rx:     return "RX";
    case AppMode::Node:   return "NODE";
    case AppMode::Beacon: return "BEACON";
    default:              return "IDLE";
  }
}

static const char* className(uint8_t c) {
#if ENABLE_ML
  if (c < HN_N_CLASSES) return HN_CLASS_NAMES[c];
#endif
  static char buf[8];
  snprintf(buf, sizeof(buf), "c%u", c);
  return buf;
}

static void uiLine(int i, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_ui[i], UI_COLS + 1, fmt, ap);
  va_end(ap);
}

static void uiRefresh() {
#if UI_AVAILABLE
  uiLine(0, "HushNet %s n%u", modeName(g_mode), (unsigned)NODE_ID);
  uiLine(1, "%.3f SF%u %ddBm", g_cfg.freqHz / 1.0e6, g_cfg.sf, (int)g_antPwr);
  uiShow(g_ui);
#endif
}

static void printHex(const uint8_t* d, size_t n) {
  for (size_t i = 0; i < n; i++) CONSOLE.printf("%02X", d[i]);
}

// True when the user typed something while a long command is running (used to stop it).
// Line endings left over from the command itself are skipped.
static bool userAbort() {
  while (CONSOLE.available()) {
    const int c = CONSOLE.peek();
    if (c == '\r' || c == '\n' || c == ' ') {
      CONSOLE.read();
      continue;
    }
    return true;
  }
  return false;
}

static const Band* findBand(uint32_t freqHz, uint32_t bwHz) {
  const uint32_t lo = freqHz - bwHz / 2, hi = freqHz + bwHz / 2;
  for (size_t i = 0; i < sizeof(BANDS) / sizeof(BANDS[0]); i++) {
    if (lo >= BANDS[i].lo && hi <= BANDS[i].hi) return &BANDS[i];
  }
  return nullptr;
}

// ---- duty cycle guard: transmit time summed over the last 60 minutes -------
static void dcAdvance() {
  const uint32_t m = millis() / 60000UL;
  if (m - g_dcMinute >= 60) {
    memset(g_dcBins, 0, sizeof(g_dcBins));
    g_dcMinute = m;
  }
  while (g_dcMinute != m) {
    g_dcMinute++;
    g_dcBins[g_dcMinute % 60] = 0;
  }
}

static uint32_t dcUsedMs() {
  dcAdvance();
  uint32_t s = 0;
  for (int i = 0; i < 60; i++) s += g_dcBins[i];
  return s;
}

static uint32_t dcBudgetMs() {
  const Band* b = findBand(g_cfg.freqHz, g_cfg.bwHz);
  return b ? 3600UL * b->dutyTenths : 0;      // 1 % of an hour is 36 000 ms
}

static void dcAdd(uint32_t us) {
  dcAdvance();
  g_dcBins[g_dcMinute % 60] += (us + 999) / 1000;
}

// True when a packet of this length may be sent now. Prints the reason when it may not.
static bool txAllowed(size_t len) {
  if (!g_radioOk) {
    CONSOLE.printf("ERR radio not running (code %d)\n", g_radioErr);
    return false;
  }
  const uint32_t budget = dcBudgetMs();
  if (budget == 0) {
    CONSOLE.println("TX_BLOCKED reason=\"channel is not inside an EU868 sub-band\"");
    return false;
  }
  const uint32_t toaMs = ((uint32_t)radio.getTimeOnAir(len) + 999) / 1000;
  const uint32_t used = dcUsedMs();
  if (used + toaMs > budget) {
    CONSOLE.printf("TX_BLOCKED reason=\"duty cycle budget used\" used_ms=%lu budget_ms=%lu\n", (unsigned long)used,
                   (unsigned long)budget);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Printing the state
// ---------------------------------------------------------------------------
static void printCfg() {
  const Band* b = findBand(g_cfg.freqHz, g_cfg.bwHz);
  CONSOLE.printf("CFG freq=%lu bw=%lu sf=%u cr=%u sync=0x%02X pre=%u pwr=%d chip_pwr=%d crc=%u iq=%u lna=%u len=%u",
                 (unsigned long)g_cfg.freqHz, (unsigned long)g_cfg.bwHz, g_cfg.sf, g_cfg.cr, g_cfg.sync, g_cfg.preamble,
                 (int)g_antPwr, (int)g_chipPwr, g_cfg.crc, g_cfg.iq, g_cfg.lna, g_cfg.len);
  if (g_radioOk) {
    CONSOLE.printf(" toa_us=%lu toa_cls_us=%lu", (unsigned long)radio.getTimeOnAir(g_cfg.len),
                   (unsigned long)radio.getTimeOnAir(PKT_CLASS_LEN));
  }
  CONSOLE.printf(" duty_pct=%.1f band=\"%s\"\n", b ? b->dutyTenths / 10.0 : 0.0, b ? b->name : "outside EU868, transmit disabled");
}

static void printStat() {
  CONSOLE.printf("STAT mode=%s rx_ok=%lu rx_crc=%lu rx_other=%lu lost=%lu tx=%lu airtime_ms_last_hour=%lu budget_ms=%lu t_ms=%lu\n",
                 modeName(g_mode), (unsigned long)g_rxOk, (unsigned long)g_rxCrc, (unsigned long)g_rxOther,
                 (unsigned long)g_lost, (unsigned long)g_txCount, (unsigned long)dcUsedMs(), (unsigned long)dcBudgetMs(),
                 (unsigned long)millis());
}

static void printInfo() {
  CONSOLE.printf("INFO board=\"%s\" tag=%s fw=%s node=%u chip=\"%s\" rev=%u cores=%u cpu_mhz=%lu flash_mb=%lu psram_kb=%lu "
                 "sketch_kb=%lu heap_free=%lu heap_min=%lu\n",
                 BOARD_NAME, BOARD_TAG, FW_VERSION, (unsigned)NODE_ID, ESP.getChipModel(), (unsigned)ESP.getChipRevision(),
                 (unsigned)ESP.getChipCores(), (unsigned long)ESP.getCpuFreqMHz(),
                 (unsigned long)(ESP.getFlashChipSize() / (1024UL * 1024UL)), (unsigned long)(ESP.getPsramSize() / 1024UL),
                 (unsigned long)(ESP.getSketchSize() / 1024UL), (unsigned long)ESP.getFreeHeap(),
                 (unsigned long)ESP.getMinFreeHeap());
  if (g_radioOk) {
    CONSOLE.printf("INFO radio=ok radiolib=%d.%d.%d fem=%s fem_detected=%s tcxo_v=%.1f pwr_min=%d pwr_max=%d\n",
                   RADIOLIB_VERSION_MAJOR, RADIOLIB_VERSION_MINOR, RADIOLIB_VERSION_PATCH, femName(g_fem), femName(g_femDetected),
                   g_tcxoUsed, (int)minAntPwr(), (int)maxAntPwr());
    if (femMismatch()) {
      CONSOLE.println("WARN the front end setting (HELTEC_FEM in hn_config.h) and the boot time test disagree. Check the marking "
                      "on the board (V4.2 or V4.3). Power is computed for the KCT8103L (+13 dB) to stay on the safe side.");
    }
  } else {
    CONSOLE.printf("INFO radio=fail code=%d radiolib=%d.%d.%d\n", g_radioErr, RADIOLIB_VERSION_MAJOR, RADIOLIB_VERSION_MINOR,
                   RADIOLIB_VERSION_PATCH);
  }
#if ENABLE_ML
  if (g_mlOk) {
    CONSOLE.printf("INFO ml=ok model_bytes=%u arena_used=%u arena_size=%u classes=%d replay_clips=%d input=\"1x%dx%dx1 int8\"\n",
                   (unsigned)g_model_len, (unsigned)hn::mlArenaUsed(), (unsigned)hn::kArenaSize, HN_N_CLASSES, HN_N_REPLAY,
                   HN_N_FRAMES, HN_N_MELS);
  } else {
    CONSOLE.printf("INFO ml=fail reason=\"%s\"\n", g_mlErr);
  }
#else
  CONSOLE.println("INFO ml=off");
#endif
  printCfg();
  printStat();
}

static void printHelp() {
  CONSOLE.println("HELP commands (one per line):");
  CONSOLE.println("HELP  INFO                      board, radio, model, settings, counters");
  CONSOLE.println("HELP  CFG key=value ...         freq= (Hz or MHz) bw= (Hz or kHz) sf=7..12 cr=5..8 pwr= (dBm at antenna)");
  CONSOLE.println("HELP                            sync=0x12 pre=8 crc=0|1 iq=0|1 lna=0|1 len= (test packet bytes)");
  CONSOLE.println("HELP  MODE IDLE|RX|NODE|BEACON  RX = print every packet, NODE = classify + send, BEACON = test packets");
  CONSOLE.println("HELP  INFER [n]                 classify the replay clips (or clip n) and compare with the PC results");
  CONSOLE.println("HELP  SEND n                    classify replay clip n and send the result once");
  CONSOLE.println("HELP  TX n [gap_ms]             send n test packets (needs another board in MODE RX)");
  CONSOLE.println("HELP  PING n                    send n pings and wait for the answers (other board in MODE RX)");
  CONSOLE.println("HELP  RAW hexbytes              send these bytes once, unchanged (to replay a captured frame)");
  CONSOLE.println("HELP  STAT [RESET]              counters");
  CONSOLE.println("HELP  SAVE | FACTORY            keep settings and mode over power cycles | back to defaults");
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
static void defaultCfg() {
  g_cfg.freqHz = DEF_FREQ_HZ;
  g_cfg.bwHz = DEF_BW_HZ;
  g_cfg.sf = DEF_SF;
  g_cfg.cr = DEF_CR;
  g_cfg.sync = DEF_SYNC;
  g_cfg.preamble = DEF_PREAMBLE;
  g_cfg.pwrDbm = DEF_PWR_DBM;
  g_cfg.crc = 1;
  g_cfg.iq = 0;
  g_cfg.lna = 0;
  g_cfg.len = DEF_TEST_LEN;
}

static bool loadSaved() {
  Preferences p;
  if (!p.begin("hushnet", true)) return false;
  Saved s;
  const size_t n = p.getBytes("s", &s, sizeof(s));
  p.end();
  if (n != sizeof(s) || s.version != SAVED_VERSION) return false;
  g_cfg = s.cfg;
  g_mode = (AppMode)s.mode;
  return true;
}

static void saveNow() {
  Preferences p;
  p.begin("hushnet", false);
  Saved s;
  s.version = SAVED_VERSION;
  s.mode = (uint8_t)g_mode;
  s.cfg = g_cfg;
  p.putBytes("s", &s, sizeof(s));
  p.end();
}

static void enterMode(AppMode m) {
  g_mode = m;
  if (g_radioOk) {
    if (m == AppMode::Rx) radioStartRx(g_cfg.lna);
    else radioIdle();
  }
  g_nextTick = millis() + 1000;
  for (int i = 2; i < UI_LINES; i++) g_ui[i][0] = 0;
  uiRefresh();
}

// Parses "key=value" tokens of a CFG command into a copy of the settings. Returns false on a bad token.
static bool parseCfg(char* args, RadioCfg& c) {
  for (char* tok = strtok(args, " \t"); tok; tok = strtok(nullptr, " \t")) {
    char* eq = strchr(tok, '=');
    if (!eq) {
      CONSOLE.printf("ERR expected key=value, got \"%s\"\n", tok);
      return false;
    }
    *eq = 0;
    const char* k = tok;
    const char* v = eq + 1;
    const double d = strtod(v, nullptr);
    const long n = strtol(v, nullptr, 0);
    if (!strcasecmp(k, "freq"))      c.freqHz = (uint32_t)(d < 100000.0 ? d * 1.0e6 + 0.5 : d + 0.5);
    else if (!strcasecmp(k, "bw"))   c.bwHz = (uint32_t)(d < 1000.0 ? d * 1000.0 + 0.5 : d + 0.5);
    else if (!strcasecmp(k, "sf"))   c.sf = (uint8_t)n;
    else if (!strcasecmp(k, "cr"))   c.cr = (uint8_t)n;
    else if (!strcasecmp(k, "pwr"))  c.pwrDbm = (int8_t)n;
    else if (!strcasecmp(k, "sync")) c.sync = (uint8_t)n;
    else if (!strcasecmp(k, "pre"))  c.preamble = (uint16_t)n;
    else if (!strcasecmp(k, "crc"))  c.crc = n ? 1 : 0;
    else if (!strcasecmp(k, "iq"))   c.iq = n ? 1 : 0;
    else if (!strcasecmp(k, "lna"))  c.lna = n ? 1 : 0;
    else if (!strcasecmp(k, "len"))  c.len = (uint8_t)n;
    else {
      CONSOLE.printf("ERR unknown setting \"%s\"\n", k);
      return false;
    }
  }
  if (c.sf < 7 || c.sf > 12) { CONSOLE.println("ERR sf must be 7..12"); return false; }
  if (c.cr < 5 || c.cr > 8) { CONSOLE.println("ERR cr must be 5..8 (coding rate 4/5..4/8)"); return false; }
  if (c.bwHz != 125000UL && c.bwHz != 250000UL && c.bwHz != 500000UL && c.bwHz != 62500UL) {
    CONSOLE.println("ERR bw must be 62.5, 125, 250 or 500 kHz");
    return false;
  }
  if (c.freqHz < 850000000UL || c.freqHz > 930000000UL) { CONSOLE.println("ERR freq outside 850..930 MHz"); return false; }
  if (c.len < 6 || c.len > 200) { CONSOLE.println("ERR len must be 6..200"); return false; }
  if (c.preamble < 6 || c.preamble > 1024) { CONSOLE.println("ERR pre must be 6..1024"); return false; }
  return true;
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------
static void fillHeader(uint8_t* b, uint16_t seq, uint8_t type, uint8_t b5) {
  b[0] = PKT_MAGIC;
  b[1] = NODE_ID;
  b[2] = (uint8_t)(seq & 0xFF);
  b[3] = (uint8_t)(seq >> 8);
  b[4] = type;
  b[5] = b5;
}

// Sends one packet (blocking) and accounts its time on air. Returns the report; state != 0 means failure.
static TxReport sendPacket(const uint8_t* b, size_t len) {
  ledSet(true);
  TxReport t = radioSend(b, len);
  ledSet(false);
  if (t.state == RADIOLIB_ERR_NONE) {
    dcAdd(t.toaCalcUs);
    g_txCount++;
  }
  return t;
}

static void backToMode() {
  if (!g_radioOk) return;
  if (g_mode == AppMode::Rx) radioStartRx(g_cfg.lna);
  else radioIdle();
}

static void cmdTx(int n, int gapMs) {
  if (n < 1) n = 1;
  if (gapMs < 20) gapMs = 20;
  uint8_t b[255];
  int sent = 0;
  g_burst++;
  CONSOLE.printf("TX_BEGIN burst=%u n=%d len=%u gap_ms=%d\n", g_burst, n, g_cfg.len, gapMs);
  for (int i = 0; i < n; i++) {
    if (userAbort()) {
      CONSOLE.println("TX_ABORT reason=\"serial input\"");
      break;
    }
    fillHeader(b, (uint16_t)i, PKT_TEST, g_burst);
    for (size_t k = 6; k < g_cfg.len; k++) b[k] = (uint8_t)(k * 7 + i);
    if (!txAllowed(g_cfg.len)) break;
    const TxReport t = sendPacket(b, g_cfg.len);
    if (t.state != RADIOLIB_ERR_NONE) {
      CONSOLE.printf("TX_ERR code=%d\n", t.state);
      break;
    }
    sent++;
    CONSOLE.printf("TX burst=%u seq=%d len=%u toa_meas_us=%lu toa_calc_us=%lu total_us=%lu t_ms=%lu\n", g_burst, i, g_cfg.len,
                   (unsigned long)t.toaMeasUs, (unsigned long)t.toaCalcUs, (unsigned long)t.totalUs, (unsigned long)millis());
    delay(gapMs);
  }
  CONSOLE.printf("TX_END burst=%u sent=%d n=%d\n", g_burst, sent, n);
  uiLine(5, "tx burst %u: %d", g_burst, sent);
  uiRefresh();
  backToMode();
}

static void cmdPing(int n) {
  if (n < 1) n = 1;
  uint8_t b[PKT_PONG_LEN];
  int sent = 0, replies = 0;
  const uint32_t waitMs = (uint32_t)radio.getTimeOnAir(PKT_PONG_LEN) / 1000UL + 600UL;
  for (int i = 0; i < n; i++) {
    if (userAbort()) break;
    fillHeader(b, (uint16_t)i, PKT_PING, 0);
    if (!txAllowed(PKT_CLASS_LEN)) break;
    const TxReport t = sendPacket(b, PKT_CLASS_LEN);
    if (t.state != RADIOLIB_ERR_NONE) {
      CONSOLE.printf("TX_ERR code=%d\n", t.state);
      break;
    }
    sent++;
    radioStartRx(g_cfg.lna);
    const uint32_t t0 = millis();
    bool got = false;
    while (millis() - t0 < waitMs) {
      if (g_dio1Flag) {
        const uint32_t tRx = millis();
        radioRead(g_rx, g_cfg.lna);
        if (g_rx.state == RADIOLIB_ERR_NONE && g_rx.len >= PKT_PONG_LEN && g_rx.data[0] == PKT_MAGIC && g_rx.data[4] == PKT_PONG &&
            (uint16_t)(g_rx.data[2] | (g_rx.data[3] << 8)) == (uint16_t)i) {
          got = true;
          replies++;
          CONSOLE.printf("PING seq=%d reply=1 from=%u rtt_ms=%lu rssi_here=%.1f snr_here=%.2f ferr_hz=%ld rssi_there=%d snr_there=%.2f\n",
                         i, g_rx.data[1], (unsigned long)(tRx - t0), g_rx.rssi, g_rx.snr, g_rx.ferr, (int)(int8_t)g_rx.data[5],
                         (int8_t)g_rx.data[6] / 4.0);
          uiLine(2, "ping %d ok %lums", i, (unsigned long)(tRx - t0));
          uiLine(4, "RSSI %.0f SNR %.1f", g_rx.rssi, g_rx.snr);
          break;
        }
      }
      delay(1);
    }
    if (!got) {
      CONSOLE.printf("PING seq=%d reply=0\n", i);
      uiLine(2, "ping %d no reply", i);
    }
    uiLine(5, "replies %d/%d", replies, sent);
    uiRefresh();
    delay(200);
  }
  CONSOLE.printf("PING_END sent=%d replies=%d\n", sent, replies);
  backToMode();
}

// Sends the bytes given as a hex string once, exactly as they are (no HushNet header).
// Used to replay a frame captured from another LoRa device, for example the Waveshare dongle.
static void cmdRaw(const char* hex) {
  uint8_t b[200];
  size_t n = 0;
  while (hex[0] && hex[1] && n < sizeof(b)) {
    char two[3] = { hex[0], hex[1], 0 };
    char* end = nullptr;
    const long v = strtol(two, &end, 16);
    if (end != two + 2) {
      CONSOLE.println("ERR RAW needs an even number of hex digits, for example RAW 48656C6C6F");
      return;
    }
    b[n++] = (uint8_t)v;
    hex += 2;
  }
  if (n == 0 || hex[0]) {
    CONSOLE.println("ERR RAW needs 1 to 200 bytes as hex digits without spaces");
    return;
  }
  if (!txAllowed(n)) return;
  const TxReport t = sendPacket(b, n);
  CONSOLE.printf("TX type=raw len=%u code=%d toa_meas_us=%lu toa_calc_us=%lu t_ms=%lu\n", (unsigned)n, t.state,
                 (unsigned long)t.toaMeasUs, (unsigned long)t.toaCalcUs, (unsigned long)millis());
  backToMode();
}

// ---------------------------------------------------------------------------
// Machine learning commands
// ---------------------------------------------------------------------------
#if ENABLE_ML
// Classifies replay clip c, prints one INFER line and returns the result. Also compares with the PC results.
static bool inferClip(int c, hn::InferResult& r, bool& sameAsPc) {
  const uint32_t t0 = micros();
  hn::computeFeatures(HN_REPLAY_AUDIO[c], hn::s_featQ);
  r.featUs = micros() - t0;
  int nDiff = 0, maxDiff = 0;
  for (int i = 0; i < HN_N_FRAMES * HN_N_MELS; i++) {
    const int d = abs((int)hn::s_featQ[i] - (int)HN_REPLAY_PC_FEAT[c][i]);
    if (d) nDiff++;
    if (d > maxDiff) maxDiff = d;
  }
  if (!hn::runModel(r)) {
    CONSOLE.println("ERR interpreter failed");
    return false;
  }
  int scoreDiff = 0;
  for (int i = 0; i < HN_N_CLASSES; i++) {
    const int d = abs((int)r.scores[i] - (int)HN_REPLAY_PC_SCORES[c][i]);
    if (d > scoreDiff) scoreDiff = d;
  }
  sameAsPc = (r.cls == HN_REPLAY_PC_PRED[c]);
  CONSOLE.printf("INFER clip=%d src=%s true=%s pred=%s conf=%.3f correct=%d pc_pred=%s same_as_pc=%d feat_ms=%.1f infer_ms=%.1f "
                 "feat_ndiff=%d feat_maxdiff=%d score_maxdiff=%d heap_free=%lu\n",
                 c, HN_REPLAY_SRC[c], HN_CLASS_NAMES[HN_REPLAY_TRUE[c]], HN_CLASS_NAMES[r.cls], r.conf,
                 r.cls == HN_REPLAY_TRUE[c] ? 1 : 0, HN_CLASS_NAMES[HN_REPLAY_PC_PRED[c]], sameAsPc ? 1 : 0, r.featUs / 1000.0,
                 r.inferUs / 1000.0, nDiff, maxDiff, scoreDiff, (unsigned long)ESP.getFreeHeap());
  return true;
}

static void cmdInfer(int only) {
  if (!g_mlOk) {
    CONSOLE.printf("ERR model not running: %s\n", g_mlErr);
    return;
  }
  int n = 0, correct = 0, same = 0;
  double featMs = 0, inferMs = 0;
  for (int c = 0; c < HN_N_REPLAY; c++) {
    if (only >= 0 && c != only) continue;
    hn::InferResult r;
    bool sameAsPc = false;
    if (!inferClip(c, r, sameAsPc)) return;
    n++;
    if (r.cls == HN_REPLAY_TRUE[c]) correct++;
    if (sameAsPc) same++;
    featMs += r.featUs / 1000.0;
    inferMs += r.inferUs / 1000.0;
    delay(2);
  }
  if (n == 0) {
    CONSOLE.printf("ERR clip number must be 0..%d\n", HN_N_REPLAY - 1);
    return;
  }
  CONSOLE.printf("INFER_SUMMARY board=%s clips=%d correct=%d same_as_pc=%d feat_ms_avg=%.1f infer_ms_avg=%.1f total_ms_avg=%.1f "
                 "arena_used=%u model_bytes=%u heap_free=%lu heap_min=%lu\n",
                 BOARD_TAG, n, correct, same, featMs / n, inferMs / n, (featMs + inferMs) / n, (unsigned)hn::mlArenaUsed(),
                 (unsigned)g_model_len, (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getMinFreeHeap());
  uiLine(2, "replay %d clips", n);
  uiLine(3, "correct %d same %d", correct, same);
  uiLine(4, "ft %.0fms inf %.0fms", featMs / n, inferMs / n);
  uiRefresh();
}

// Classifies replay clip c and sends the result as a 6 byte packet.
static void classifyAndSend(int c) {
  if (!g_mlOk) {
    CONSOLE.printf("ERR model not running: %s\n", g_mlErr);
    return;
  }
  if (c < 0 || c >= HN_N_REPLAY) {
    CONSOLE.printf("ERR clip number must be 0..%d\n", HN_N_REPLAY - 1);
    return;
  }
  hn::InferResult r;
  bool sameAsPc = false;
  if (!inferClip(c, r, sameAsPc)) return;
  uiLine(2, "clip %d %s", c, HN_CLASS_NAMES[HN_REPLAY_TRUE[c]]);
  uiLine(3, "%s %u%%", HN_CLASS_NAMES[r.cls], (unsigned)(r.conf * 100.0f + 0.5f));
  uiLine(4, "ft %.0fms inf %.0fms", r.featUs / 1000.0, r.inferUs / 1000.0);
  if (txAllowed(PKT_CLASS_LEN)) {
    uint8_t b[PKT_CLASS_LEN];
    fillHeader(b, g_seq, r.cls, r.confByte);
    const TxReport t = sendPacket(b, PKT_CLASS_LEN);
    if (t.state == RADIOLIB_ERR_NONE) {
      CONSOLE.printf("TX type=cls seq=%u class=%u name=%s conf=%.3f len=%u toa_meas_us=%lu toa_calc_us=%lu t_ms=%lu hex=", g_seq,
                     r.cls, HN_CLASS_NAMES[r.cls], r.conf, (unsigned)PKT_CLASS_LEN, (unsigned long)t.toaMeasUs,
                     (unsigned long)t.toaCalcUs, (unsigned long)millis());
      printHex(b, PKT_CLASS_LEN);
      CONSOLE.println();
      uiLine(5, "tx #%u toa %lums", g_seq, (unsigned long)(t.toaMeasUs / 1000UL));
      g_seq++;
    } else {
      CONSOLE.printf("TX_ERR code=%d\n", t.state);
      uiLine(5, "tx error %d", t.state);
    }
  } else {
    uiLine(5, "tx blocked");
  }
  uiRefresh();
  backToMode();
}
#endif  // ENABLE_ML

// ---------------------------------------------------------------------------
// Receiving
// ---------------------------------------------------------------------------
static void trackSeq(uint8_t node, int kind, uint8_t burst, uint16_t seq) {
  SeqTrack& t = g_track[node & 15][kind];
  if (t.valid && t.burst == burst) {
    const uint16_t gap = (uint16_t)(seq - t.last);
    if (gap >= 2 && gap < 1000) g_lost += (uint32_t)(gap - 1);   // a gap in the numbering means lost packets
  }
  t.valid = true;
  t.burst = burst;
  t.last = seq;
}

static void handleRx() {
  radioRead(g_rx, g_cfg.lna);
  const unsigned long now = millis();
  if (g_rx.state == RADIOLIB_ERR_CRC_MISMATCH) {
    g_rxCrc++;
    CONSOLE.printf("RX type=crc_error len=%u rssi=%.1f snr=%.2f ferr_hz=%ld t_ms=%lu hex=", (unsigned)g_rx.len, g_rx.rssi,
                   g_rx.snr, g_rx.ferr, now);
    printHex(g_rx.data, g_rx.len);
    CONSOLE.println();
    return;
  }
  if (g_rx.state != RADIOLIB_ERR_NONE) {
    CONSOLE.printf("RX_ERR code=%d\n", g_rx.state);
    return;
  }
  ledSet(true);
  const uint8_t* d = g_rx.data;
  if (g_rx.len < PKT_CLASS_LEN || d[0] != PKT_MAGIC) {
    // Not one of ours (for example a frame from the Waveshare dongle): show the raw bytes.
    g_rxOther++;
    CONSOLE.printf("RX type=other len=%u rssi=%.1f snr=%.2f ferr_hz=%ld t_ms=%lu hex=", (unsigned)g_rx.len, g_rx.rssi, g_rx.snr,
                   g_rx.ferr, now);
    printHex(d, g_rx.len);
    CONSOLE.println();
    uiLine(2, "other len %u", (unsigned)g_rx.len);
    uiLine(3, "");
  } else {
    const uint8_t node = d[1], type = d[4];
    const uint16_t seq = (uint16_t)(d[2] | (d[3] << 8));
    g_rxOk++;
    if (type == PKT_TEST) {
      trackSeq(node, 1, d[5], seq);
      CONSOLE.printf("RX type=test node=%u burst=%u seq=%u len=%u rssi=%.1f snr=%.2f ferr_hz=%ld lost=%lu t_ms=%lu\n", node, d[5],
                     seq, (unsigned)g_rx.len, g_rx.rssi, g_rx.snr, g_rx.ferr, (unsigned long)g_lost, now);
      uiLine(2, "n%u test #%u", node, seq);
      uiLine(3, "burst %u len %u", d[5], (unsigned)g_rx.len);
    } else if (type == PKT_PING) {
      CONSOLE.printf("RX type=ping node=%u seq=%u rssi=%.1f snr=%.2f ferr_hz=%ld t_ms=%lu\n", node, seq, g_rx.rssi, g_rx.snr,
                     g_rx.ferr, now);
      uiLine(2, "n%u ping #%u", node, seq);
      uiLine(3, "");
      delay(30);                                 // give the other side time to start listening
      if (txAllowed(PKT_PONG_LEN)) {
        uint8_t b[PKT_PONG_LEN];
        float rs = g_rx.rssi < -128 ? -128 : (g_rx.rssi > 127 ? 127 : g_rx.rssi);
        float sn = g_rx.snr * 4.0f;
        sn = sn < -128 ? -128 : (sn > 127 ? 127 : sn);
        fillHeader(b, seq, PKT_PONG, (uint8_t)(int8_t)lroundf(rs));
        b[6] = (uint8_t)(int8_t)lroundf(sn);
        const TxReport t = sendPacket(b, PKT_PONG_LEN);
        CONSOLE.printf("TX type=pong seq=%u code=%d toa_meas_us=%lu\n", seq, t.state, (unsigned long)t.toaMeasUs);
      }
      radioStartRx(g_cfg.lna);
    } else if (type == PKT_PONG) {
      CONSOLE.printf("RX type=pong node=%u seq=%u rssi=%.1f snr=%.2f ferr_hz=%ld t_ms=%lu\n", node, seq, g_rx.rssi, g_rx.snr,
                     g_rx.ferr, now);
    } else {
      trackSeq(node, 0, 0, seq);
      CONSOLE.printf("RX type=cls node=%u seq=%u class=%u name=%s conf=%.3f len=%u rssi=%.1f snr=%.2f ferr_hz=%ld lost=%lu t_ms=%lu hex=",
                     node, seq, type, className(type), d[5] / 255.0, (unsigned)g_rx.len, g_rx.rssi, g_rx.snr, g_rx.ferr,
                     (unsigned long)g_lost, now);
      printHex(d, g_rx.len);
      CONSOLE.println();
      uiLine(2, "n%u #%u", node, seq);
      uiLine(3, "%s %u%%", className(type), (unsigned)(d[5] * 100UL / 255UL));
    }
  }
  uiLine(4, "RSSI %.0f SNR %.1f", g_rx.rssi, g_rx.snr);
  uiLine(5, "ok%lu lost%lu crc%lu", (unsigned long)g_rxOk, (unsigned long)g_lost, (unsigned long)g_rxCrc);
  uiRefresh();
  ledSet(false);
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------
static void handleLine(char* line) {
  while (*line == ' ' || *line == '\t') line++;
  if (!*line) return;
  char* args = line;
  while (*args && *args != ' ' && *args != '\t') args++;
  if (*args) *args++ = 0;
  while (*args == ' ' || *args == '\t') args++;
  for (char* p = line; *p; p++) *p = (char)toupper((unsigned char)*p);

  if (!strcmp(line, "HELP") || !strcmp(line, "?")) {
    printHelp();
  } else if (!strcmp(line, "INFO")) {
    printInfo();
  } else if (!strcmp(line, "CFG")) {
    RadioCfg c = g_cfg;
    if (*args && !parseCfg(args, c)) return;
    if (g_radioOk && *args) {
      const int16_t st = radioApply(c);
      if (st != RADIOLIB_ERR_NONE) {
        CONSOLE.printf("ERR radio rejected the settings (code %d), keeping the old ones\n", st);
        radioApply(g_cfg);
      } else {
        g_cfg = c;
      }
      g_cfg.pwrDbm = g_antPwr;                    // keep what the board can really do
      backToMode();
    } else if (*args) {
      g_cfg = c;
    }
    printCfg();
    uiRefresh();
  } else if (!strcmp(line, "MODE")) {
    for (char* p = args; *p; p++) *p = (char)toupper((unsigned char)*p);
    if (!strcmp(args, "RX")) enterMode(AppMode::Rx);
    else if (!strcmp(args, "IDLE")) enterMode(AppMode::Idle);
    else if (!strcmp(args, "NODE")) enterMode(AppMode::Node);
    else if (!strcmp(args, "BEACON")) enterMode(AppMode::Beacon);
    else if (*args) {
      CONSOLE.println("ERR use MODE IDLE, RX, NODE or BEACON");
      return;
    }
    CONSOLE.printf("MODE mode=%s\n", modeName(g_mode));
  } else if (!strcmp(line, "STAT")) {
    for (char* p = args; *p; p++) *p = (char)toupper((unsigned char)*p);
    if (!strcmp(args, "RESET")) {
      g_rxOk = g_rxCrc = g_rxOther = g_lost = 0;
      memset(g_track, 0, sizeof(g_track));
    }
    printStat();
  } else if (!strcmp(line, "TX")) {
    int n = 10, gap = 100;
    sscanf(args, "%d %d", &n, &gap);
    cmdTx(n, gap);
  } else if (!strcmp(line, "PING")) {
    int n = 5;
    sscanf(args, "%d", &n);
    if (!g_radioOk) CONSOLE.printf("ERR radio not running (code %d)\n", g_radioErr);
    else cmdPing(n);
  } else if (!strcmp(line, "RAW")) {
    cmdRaw(args);
  } else if (!strcmp(line, "INFER")) {
#if ENABLE_ML
    int only = -1;
    sscanf(args, "%d", &only);
    cmdInfer(only);
#else
    CONSOLE.println("ERR built with ENABLE_ML 0");
#endif
  } else if (!strcmp(line, "SEND")) {
#if ENABLE_ML
    int c = 0;
    sscanf(args, "%d", &c);
    classifyAndSend(c);
#else
    CONSOLE.println("ERR built with ENABLE_ML 0");
#endif
  } else if (!strcmp(line, "SAVE")) {
    saveNow();
    CONSOLE.printf("SAVED mode=%s\n", modeName(g_mode));
  } else if (!strcmp(line, "FACTORY")) {
    Preferences p;
    p.begin("hushnet", false);
    p.clear();
    p.end();
    defaultCfg();
    if (g_radioOk) radioApply(g_cfg);
    g_cfg.pwrDbm = g_antPwr;
    enterMode((AppMode)BOOT_MODE);
    CONSOLE.println("FACTORY done");
    printCfg();
  } else {
    CONSOLE.printf("ERR unknown command \"%s\" (send HELP)\n", line);
  }
}

static void pollSerial() {
  while (CONSOLE.available()) {
    const char ch = (char)CONSOLE.read();
    if (ch == '\r' || ch == '\n') {
      if (g_lineLen) {
        g_line[g_lineLen] = 0;
        g_lineLen = 0;
        if (ch == '\r' && CONSOLE.peek() == '\n') CONSOLE.read();   // CR LF line ending
        handleLine(g_line);
      }
    } else if (g_lineLen < sizeof(g_line) - 1) {
      g_line[g_lineLen++] = ch;
    }
  }
}

// ---------------------------------------------------------------------------
// setup and loop
// ---------------------------------------------------------------------------
void setup() {
  CONSOLE.begin(115200);
  radioPinsIdle();
  ledBegin();
  uiBegin();
  uiLine(0, "HushNet PoC %s", FW_VERSION);
  uiLine(1, "starting...");
  uiShow(g_ui);
#if defined(BOARD_NESSO_N1) && NESSO_ENABLE_CHARGING
  g_battery.begin();
  g_battery.enableCharge();
#endif
  const uint32_t t0 = millis();
  while (!CONSOLE && millis() - t0 < 3000) delay(10);      // give the PC a moment to open the port

  defaultCfg();
  g_mode = (AppMode)BOOT_MODE;
  const bool saved = loadSaved();
  CONSOLE.printf("\nBOOT board=\"%s\" tag=%s fw=%s node=%u saved_settings=%d\n", BOARD_NAME, BOARD_TAG, FW_VERSION,
                 (unsigned)NODE_ID, saved ? 1 : 0);

  g_radioErr = radioBegin(g_cfg);
  g_radioOk = (g_radioErr == RADIOLIB_ERR_NONE);
  g_cfg.pwrDbm = g_antPwr;
  if (g_radioOk) {
    CONSOLE.printf("RADIO_OK fem=%s fem_detected=%s tcxo_v=%.1f\n", femName(g_fem), femName(g_femDetected), g_tcxoUsed);
    if (femMismatch()) {
      CONSOLE.println("WARN the front end setting (HELTEC_FEM in hn_config.h) and the boot time test disagree, see INFO");
    }
  } else {
    CONSOLE.printf("RADIO_FAIL code=%d (see README: troubleshooting)\n", g_radioErr);
  }

#if ENABLE_ML
  g_mlErr = hn::mlBegin();
  g_mlOk = (g_mlErr == nullptr);
  if (g_mlOk) {
    CONSOLE.printf("ML_OK model_bytes=%u arena_used=%u\n", (unsigned)g_model_len, (unsigned)hn::mlArenaUsed());
  } else {
    CONSOLE.printf("ML_FAIL reason=\"%s\"\n", g_mlErr);
  }
#endif

  enterMode(g_mode);
  printCfg();
  CONSOLE.printf("READY mode=%s (send HELP for commands)\n", modeName(g_mode));
}

void loop() {
  pollSerial();

  if (g_radioOk && g_mode == AppMode::Rx && g_dio1Flag) handleRx();

  const uint32_t now = millis();
  if ((g_mode == AppMode::Node || g_mode == AppMode::Beacon) && (int32_t)(now - g_nextTick) >= 0) {
    g_nextTick = now + (uint32_t)NODE_PERIOD_S * 1000UL;
    if (g_mode == AppMode::Node) {
#if ENABLE_ML
      classifyAndSend(g_clip);
      g_clip = (uint8_t)((g_clip + 1) % HN_N_REPLAY);
#else
      CONSOLE.println("ERR built with ENABLE_ML 0, NODE mode needs the model");
      enterMode(AppMode::Idle);
#endif
    } else if (txAllowed(g_cfg.len)) {
      uint8_t b[255];
      fillHeader(b, g_seq, PKT_TEST, 0);
      for (size_t k = 6; k < g_cfg.len; k++) b[k] = (uint8_t)(k * 7 + g_seq);
      const TxReport t = sendPacket(b, g_cfg.len);
      CONSOLE.printf("TX burst=0 seq=%u len=%u code=%d toa_meas_us=%lu toa_calc_us=%lu t_ms=%lu\n", g_seq, g_cfg.len, t.state,
                     (unsigned long)t.toaMeasUs, (unsigned long)t.toaCalcUs, (unsigned long)millis());
      uiLine(2, "beacon #%u", g_seq);
      uiLine(5, "toa %lums", (unsigned long)(t.toaMeasUs / 1000UL));
      uiRefresh();
      g_seq++;
      backToMode();
    }
  }
  delay(1);
}
