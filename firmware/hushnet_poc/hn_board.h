// Board detection, serial console and LED for the two supported boards.
// Pin numbers come from the ESP32 Arduino core variant files (see README, section Sources).
#pragma once
#include <Arduino.h>
#include "hn_config.h"

#if defined(ARDUINO_HELTEC_WIFI_LORA_32_V4)
  #define BOARD_HELTEC_V4 1
  #define BOARD_NAME  "Heltec WiFi LoRa 32 V4"
  #define BOARD_TAG   "S3"
  #define NODE_ID     NODE_ID_HELTEC
  #define BOOT_MODE   BOOT_MODE_HELTEC
#elif defined(ARDUINO_ARDUINO_NESSO_N1)
  // The Nesso N1 keeps several control lines (LoRa enable, LNA, LED, LCD) on an I2C I/O expander.
  // This library provides pinMode/digitalWrite for those lines, the display and the battery charger.
  #include <Arduino_Nesso_N1.h>
  #define BOARD_NESSO_N1 1
  #define BOARD_NAME  "Arduino Nesso N1"
  #define BOARD_TAG   "C6"
  #define NODE_ID     NODE_ID_NESSO
  #define BOOT_MODE   BOOT_MODE_NESSO
#else
  #error "Unsupported board. Choose Tools > Board > esp32 > 'Heltec WiFi LoRa 32(V4)' or 'Arduino Nesso N1'."
#endif

// ---- serial console over the USB-C port ------------------------------------
// Both boards use the native USB port of the ESP32 (no USB to UART chip).
// If "USB CDC On Boot" is Enabled, Serial is already that port.
// If it is Disabled (the default for the Heltec V4 in the Tools menu), Serial would be UART0,
// which is not connected to USB, so the sketch opens the USB port itself.
#if ARDUINO_USB_CDC_ON_BOOT
  #define CONSOLE Serial
#elif defined(ARDUINO_USB_MODE) && (ARDUINO_USB_MODE == 1)
  HWCDC HWCDCSerial;
  #define CONSOLE HWCDCSerial
#else
  #error "Set Tools > USB CDC On Boot: Enabled, or Tools > USB Mode: Hardware CDC and JTAG."
#endif

// ---- LED -------------------------------------------------------------------
inline void ledSet(bool on) {
#if defined(BOARD_NESSO_N1)
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);   // inverted logic on the Nesso N1
#else
  digitalWrite(LED_BUILTIN, on ? HIGH : LOW);   // GPIO35 on the Heltec V4
#endif
}

inline void ledBegin() {
  pinMode(LED_BUILTIN, OUTPUT);
  ledSet(false);
}
