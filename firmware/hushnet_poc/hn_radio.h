// Raw LoRa link (no LoRaWAN) on the SX1262 with RadioLib.
// Works with RadioLib 7.x and 8.x: version 8.0.0 changed begin(), the frequency and bandwidth units
// and the type of getFrequencyError(), so those calls are wrapped here.
#pragma once
#include <RadioLib.h>
#include "hn_board.h"

struct RadioCfg {
  uint32_t freqHz;
  uint32_t bwHz;
  uint8_t  sf;
  uint8_t  cr;        // 5..8 means coding rate 4/5..4/8
  uint8_t  sync;      // 0x12 private, 0x34 LoRaWAN public
  uint16_t preamble;
  int8_t   pwrDbm;    // wanted power at the antenna connector
  uint8_t  crc;       // 1 = payload CRC on
  uint8_t  iq;        // 1 = inverted IQ
  uint8_t  lna;       // Heltec V4.3 only: 1 = receive through the front end LNA, 0 = bypass it
  uint8_t  len;       // payload length of test packets
};

// ---------------------------------------------------------------------------
// Radio object and pins
// ---------------------------------------------------------------------------
#if defined(BOARD_HELTEC_V4)
  // Core variant heltec_wifi_lora_32_V4: SS 8, SCK 9, MOSI 10, MISO 11, RST_LoRa 12, BUSY_LoRa 13.
  // The variant calls the interrupt line "DIO0" for historical reasons; on the SX1262 it is DIO1 (GPIO14).
  static SX1262 radio = new Module(SS, DIO0, RST_LoRa, BUSY_LoRa);
  static const float TCXO_TRY[] = { HELTEC_TCXO_V, 1.6f, 0.0f };   // voltages tried in this order, 0 = crystal
  #define RADIO_USE_LDO HELTEC_USE_LDO

  // Front end module (FEM) between the SX1262 and the antenna. Pin use follows the Meshtastic firmware:
  static const uint8_t PIN_FEM_POWER = 7;    // FEM supply enable, HIGH = on
  static const uint8_t PIN_FEM_CSD   = 2;    // FEM chip enable, HIGH = on (also used to detect the FEM type)
  static const uint8_t PIN_KCT_CTX   = 5;    // KCT8103L (V4.3): LOW = receive through LNA, HIGH = receive bypass and transmit
  static const uint8_t PIN_GC_CPS    = 46;   // GC1109 (V4.2): HIGH = power amplifier on, LOW = amplifier bypassed
#else
  // Core variant arduino_nesso_n1: LORA_CS 23, LORA_IRQ 15 (DIO1), LORA_BUSY 19, SCK 20, MOSI 21, MISO 22.
  // The reset line is on the I/O expander (LORA_ENABLE), so RadioLib gets no reset pin.
  static SX1262 radio = new Module(LORA_CS, LORA_IRQ, RADIOLIB_NC, LORA_BUSY);
  static const float TCXO_TRY[] = { NESSO_TCXO_V, 1.6f, 0.0f };    // voltages tried in this order, 0 = crystal
  #define RADIO_USE_LDO NESSO_USE_LDO
#endif

enum FemType : uint8_t { FEM_NONE = 0, FEM_KCT8103L = 1, FEM_GC1109 = 2 };
static FemType g_fem = FEM_NONE;          // front end the sketch drives
static FemType g_femDetected = FEM_NONE;  // what the boot time test saw (Heltec only)
static float   g_tcxoUsed = 0;
static int8_t  g_chipPwr = 0;     // value written to the SX1262
static int8_t  g_antPwr = 0;      // estimated power at the antenna connector

static volatile bool     g_dio1Flag = false;
static volatile uint32_t g_dio1Micros = 0;

static void IRAM_ATTR onDio1() {
  g_dio1Micros = micros();
  g_dio1Flag = true;
}

inline const char* femName(FemType f) {
  switch (f) {
    case FEM_KCT8103L: return "KCT8103L";
    case FEM_GC1109:   return "GC1109";
    default:           return "none";
  }
}

// True when the configured front end and the boot time test disagree (Heltec only).
inline bool femMismatch() { return g_femDetected != FEM_NONE && g_fem != g_femDetected; }

// Gain (dB) between the SX1262 output and the antenna connector while transmitting.
//   KCT8103L (Heltec V4.3): the power amplifier is always in the transmit path, about +13 dB
//                           (Meshtastic gain table for SX1262 settings of 0 to 13 dBm).
//   GC1109 (Heltec V4.2):   this sketch keeps the amplifier bypassed, about 1 dB loss.
//   Nesso N1:               no amplifier.
// Safety rule: if either the setting or the boot time test says KCT8103L, the +13 dB are assumed,
// so a wrong setting can only make the signal weaker, never stronger than asked for.
inline int8_t femTxGainDb() {
  if (g_fem == FEM_KCT8103L || g_femDetected == FEM_KCT8103L) return 13;
  if (g_fem == FEM_GC1109) return -1;
  return 0;
}

// Lowest and highest antenna power this board can be set to (the SX1262 itself covers -9 to +22 dBm).
inline int8_t minAntPwr() { return (int8_t)(-9 + femTxGainDb()); }
inline int8_t maxAntPwr() {
  int v = 22 + femTxGainDb();
  return (int8_t)(v > MAX_PWR_DBM ? MAX_PWR_DBM : v);
}

#if defined(BOARD_HELTEC_V4)
inline void femBegin() {
  pinMode(PIN_FEM_POWER, OUTPUT);
  digitalWrite(PIN_FEM_POWER, HIGH);
  delay(5);
  // Same test as the Meshtastic firmware: with the FEM powered, GPIO2 reads HIGH on KCT8103L boards (V4.3)
  // and LOW on GC1109 boards (V4.2).
  pinMode(PIN_FEM_CSD, INPUT);
  delay(2);
  g_femDetected = (digitalRead(PIN_FEM_CSD) == HIGH) ? FEM_KCT8103L : FEM_GC1109;
#if HELTEC_FEM == 1
  g_fem = FEM_KCT8103L;
#elif HELTEC_FEM == 2
  g_fem = FEM_GC1109;
#else
  g_fem = g_femDetected;
#endif
  pinMode(PIN_FEM_CSD, OUTPUT);
  digitalWrite(PIN_FEM_CSD, HIGH);
  if (g_fem == FEM_KCT8103L) {
    pinMode(PIN_KCT_CTX, OUTPUT);
    digitalWrite(PIN_KCT_CTX, HIGH);          // receive bypass until a receive call decides
  } else {
    pinMode(PIN_GC_CPS, OUTPUT);
    digitalWrite(PIN_GC_CPS, LOW);            // amplifier bypassed
  }
}
#endif

// Called right before the radio is put into transmit or receive.
inline void femForTx() {
#if defined(BOARD_HELTEC_V4)
  if (g_fem == FEM_KCT8103L) digitalWrite(PIN_KCT_CTX, HIGH);
#endif
}

inline void femForRx(bool lna) {
#if defined(BOARD_HELTEC_V4)
  if (g_fem == FEM_KCT8103L) digitalWrite(PIN_KCT_CTX, lna ? LOW : HIGH);
#else
  (void)lna;
#endif
}

// ---------------------------------------------------------------------------
// RadioLib 7.x / 8.x differences
// ---------------------------------------------------------------------------
inline int16_t rlSetFrequency(uint32_t hz) {
#if RADIOLIB_VERSION_MAJOR >= 8
  return radio.setFrequency(hz);
#else
  return radio.setFrequency((float)((double)hz / 1.0e6));
#endif
}

inline int16_t rlSetBandwidth(uint32_t hz) {
#if RADIOLIB_VERSION_MAJOR >= 8
  return radio.setBandwidth(hz);
#else
  return radio.setBandwidth((float)hz / 1000.0f);
#endif
}

inline long rlFrequencyError() { return (long)radio.getFrequencyError(); }

inline int16_t rlBegin(const RadioCfg& c, int8_t chipPwr, float tcxoV, bool useLdo) {
#if RADIOLIB_VERSION_MAJOR >= 8
  if (tcxoV < 0.1f)        radio.tcxoVoltage = VoltageNone;   // crystal instead of TCXO
  else if (tcxoV < 1.65f)  radio.tcxoVoltage = Voltage1V6;
  else if (tcxoV < 1.75f)  radio.tcxoVoltage = Voltage1V7;
  else if (tcxoV < 2.0f)   radio.tcxoVoltage = Voltage1V8;
  else if (tcxoV < 2.3f)   radio.tcxoVoltage = Voltage2V2;
  else if (tcxoV < 2.55f)  radio.tcxoVoltage = Voltage2V4;
  else if (tcxoV < 2.85f)  radio.tcxoVoltage = Voltage2V7;
  else if (tcxoV < 3.15f)  radio.tcxoVoltage = Voltage3V0;
  else                     radio.tcxoVoltage = Voltage3V3;
  radio.useRegulatorLDO = useLdo;
  ConfigLoRa_t lc;
  lc.frequency = c.freqHz;
  lc.bandwidth = c.bwHz;
  lc.spreadingFactor = c.sf;
  lc.codingRate = c.cr;
  lc.syncWord = c.sync;
  lc.power = chipPwr;
  lc.preambleLength = c.preamble;
  return radio.begin(lc);
#else
  return radio.begin((float)((double)c.freqHz / 1.0e6), (float)c.bwHz / 1000.0f, c.sf, c.cr, c.sync,
                     chipPwr, c.preamble, tcxoV, useLdo);
#endif
}

// ---------------------------------------------------------------------------
// Power, configuration, start
// ---------------------------------------------------------------------------
inline void computePower(int8_t wantedAnt) {
  int v = wantedAnt;
  if (v > maxAntPwr()) v = maxAntPwr();
  if (v < minAntPwr()) v = minAntPwr();
  g_antPwr = (int8_t)v;
  g_chipPwr = (int8_t)(v - femTxGainDb());
}

// Applies every setting of c to a radio that is already running. Returns a RadioLib status code.
inline int16_t radioApply(const RadioCfg& c) {
  int16_t st = radio.standby();
  if (st != RADIOLIB_ERR_NONE) return st;
  computePower(c.pwrDbm);
  if ((st = rlSetFrequency(c.freqHz)) != RADIOLIB_ERR_NONE) return st;
  if ((st = rlSetBandwidth(c.bwHz)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setSpreadingFactor(c.sf)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setCodingRate(c.cr)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setSyncWord(c.sync)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setPreambleLength(c.preamble)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setCRC(c.crc ? 2 : 0)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.invertIQ(c.iq != 0)) != RADIOLIB_ERR_NONE) return st;
  if ((st = radio.setOutputPower(g_chipPwr)) != RADIOLIB_ERR_NONE) return st;
  return RADIOLIB_ERR_NONE;
}

// Call first thing in setup(): keeps the SX1262 deselected while other devices on the same SPI bus
// (the LCD of the Nesso N1) are being set up.
inline void radioPinsIdle() {
#if defined(BOARD_HELTEC_V4)
  pinMode(SS, OUTPUT);
  digitalWrite(SS, HIGH);
#else
  pinMode(LORA_CS, OUTPUT);
  digitalWrite(LORA_CS, HIGH);
#endif
}

// Nesso N1: the SX1262 reset line is on the I/O expander, so RadioLib cannot pulse it. Do it here.
inline void radioHardReset() {
#if !defined(BOARD_HELTEC_V4)
  digitalWrite(LORA_ENABLE, LOW);
  delay(100);
  digitalWrite(LORA_ENABLE, HIGH);
  delay(100);
#endif
}

inline int16_t radioBegin(const RadioCfg& c) {
#if defined(BOARD_HELTEC_V4)
  pinMode(Vext, OUTPUT);
  digitalWrite(Vext, LOW);                    // Vext rail on (active low), as the Meshtastic firmware does
  femBegin();
#else
  // Arduino Nesso N1 user manual: LORA_ENABLE high = module enabled (low = reset),
  // LORA_ANTENNA_SWITCH high = RF switch powered, LORA_LNA_ENABLE high = receive path connected.
  pinMode(LORA_ENABLE, OUTPUT);
  pinMode(LORA_ANTENNA_SWITCH, OUTPUT);
  digitalWrite(LORA_ANTENNA_SWITCH, HIGH);
  pinMode(LORA_LNA_ENABLE, OUTPUT);
  digitalWrite(LORA_LNA_ENABLE, HIGH);
#endif
  computePower(c.pwrDbm);
  int16_t st = RADIOLIB_ERR_UNKNOWN;
  for (size_t i = 0; i < sizeof(TCXO_TRY) / sizeof(TCXO_TRY[0]); i++) {
    if (i > 0 && TCXO_TRY[i] == TCXO_TRY[0]) continue;      // already tried
    radioHardReset();
    g_tcxoUsed = TCXO_TRY[i];
    st = rlBegin(c, g_chipPwr, g_tcxoUsed, RADIO_USE_LDO != 0);
    // RadioLib reports these two codes when the oscillator does not start (wrong TCXO voltage, or a crystal).
    if (st != RADIOLIB_ERR_SPI_CMD_FAILED && st != RADIOLIB_ERR_SPI_CMD_INVALID) break;
  }
  if (st != RADIOLIB_ERR_NONE) return st;
  radio.setDio2AsRfSwitch(true);              // DIO2 drives the transmit/receive switch on both boards
  radio.setCurrentLimit(140.0);               // SX1262 over current protection; the RadioLib default of 60 mA limits the output
  radio.setDio1Action(onDio1);
  return radioApply(c);
}

// ---------------------------------------------------------------------------
// Transmit (blocking) and receive (interrupt flag)
// ---------------------------------------------------------------------------
struct TxReport {
  int16_t  state;
  uint32_t toaMeasUs;   // from "transmission started" to the TX done interrupt
  uint32_t toaCalcUs;   // RadioLib getTimeOnAir()
  uint32_t totalUs;     // whole call, including SPI commands and oscillator start
};

inline TxReport radioSend(const uint8_t* data, size_t len) {
  TxReport r;
  r.toaCalcUs = (uint32_t)radio.getTimeOnAir(len);
  femForTx();
  g_dio1Flag = false;
  uint32_t t0 = micros();
  r.state = radio.startTransmit(data, len);   // returns when the power amplifier has ramped up
  uint32_t t1 = micros();
  if (r.state == RADIOLIB_ERR_NONE) {
    uint32_t limitMs = r.toaCalcUs / 1000UL + 1000UL;
    uint32_t w0 = millis();
    while (!g_dio1Flag && (millis() - w0) < limitMs) delay(1);
    if (!g_dio1Flag) r.state = RADIOLIB_ERR_TX_TIMEOUT;
  }
  uint32_t tDone = g_dio1Micros;
  r.toaMeasUs = g_dio1Flag ? (tDone - t1) : 0;
  r.totalUs = g_dio1Flag ? (tDone - t0) : (micros() - t0);
  radio.finishTransmit();
  g_dio1Flag = false;
  return r;
}

inline int16_t radioStartRx(bool lna) {
  femForRx(lna);
  g_dio1Flag = false;
  return radio.startReceive();
}

inline int16_t radioIdle() {
  g_dio1Flag = false;
  return radio.standby();
}

struct RxPacket {
  int16_t state;        // RADIOLIB_ERR_NONE or RADIOLIB_ERR_CRC_MISMATCH
  size_t  len;
  float   rssi;
  float   snr;
  long    ferr;
  uint8_t data[255];
};

// Call when g_dio1Flag is set while receiving. Leaves the radio in receive mode.
inline void radioRead(RxPacket& p, bool lna) {
  g_dio1Flag = false;
  p.len = radio.getPacketLength();
  if (p.len > sizeof(p.data)) p.len = sizeof(p.data);
  p.state = radio.readData(p.data, p.len);
  p.rssi = radio.getRSSI();
  p.snr = radio.getSNR();
  p.ferr = rlFrequencyError();
  radioStartRx(lna);
}
