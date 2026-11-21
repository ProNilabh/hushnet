// HushNet proof of concept: settings you may want to change.
// One sketch for both boards. Pick the board in Tools > Board and upload.
#pragma once

#define FW_VERSION "0.1"

// ---- features -------------------------------------------------------------
#define ENABLE_ML             1   // 0 = build without TensorFlow Lite (LoRa tests only, compiles much faster)
#define USE_NESSO_DISPLAY     1   // 0 = do not use the Nesso N1 screen
#define NESSO_ENABLE_CHARGING 1   // 1 = let the Nesso N1 battery charge from USB (as in the Arduino user manual)
#define USE_HELTEC_OLED       0   // 1 = show status on the Heltec OLED (needs the U8g2 library)

// Heltec V4 radio front end (the chip between the SX1262 and the antenna):
//   1 = KCT8103L, boards marked V4.3 (default: your board is marked HTIT-WB32LAF V4.3)
//   2 = GC1109, boards marked V4.2
//   0 = detect at boot, the way the Meshtastic firmware does
// The sketch also runs the detection in every case and prints a warning if it disagrees with this setting.
#define HELTEC_FEM            1

// SX1262 clock and regulator. The TCXO (oscillator) is powered from SX1262 pin DIO3.
// If the radio does not start with the first value, the sketch also tries 1.6 V and then "crystal".
//   Heltec V4: 1.8 V and DC-DC regulator, the values the Meshtastic firmware uses for this board.
//   Nesso N1:  3.0 V and LDO regulator, the values in M5Stack's own LoRa example for this board.
//              (The Arduino user manual example uses the RadioLib defaults, 1.6 V and DC-DC.)
#define HELTEC_TCXO_V         1.8f
#define HELTEC_USE_LDO        0
#define NESSO_TCXO_V          3.0f
#define NESSO_USE_LDO         1

// ---- identity -------------------------------------------------------------
#define NODE_ID_HELTEC 1
#define NODE_ID_NESSO  2

// ---- what each board does after power-up: 0 = wait for commands, 1 = receive, 2 = node (classify + send),
//      3 = beacon (numbered test packets). SAVE on the serial console overrides this until FACTORY.
#define BOOT_MODE_HELTEC 0
#define BOOT_MODE_NESSO  1

// ---- default radio settings (change at run time with the CFG command) ------
#define DEF_FREQ_HZ   868100000UL  // 868.1 MHz, EU868 band M (1 % duty cycle)
#define DEF_BW_HZ     125000UL
#define DEF_SF        7
#define DEF_CR        5            // coding rate 4/5
#define DEF_SYNC      0x12         // private LoRa sync word
#define DEF_PREAMBLE  8
#define DEF_PWR_DBM   4            // power at the antenna connector, kept low for desk tests
#define DEF_TEST_LEN  20           // payload bytes of TX test packets
#define MAX_PWR_DBM   14           // this firmware never asks for more than 14 dBm at the antenna

#define NODE_PERIOD_S 10           // node mode: one classification + one packet every N seconds
