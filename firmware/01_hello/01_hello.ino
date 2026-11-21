// Step 1 of the test checklist: proves that the toolchain, the upload and the USB serial port work.
// Works on the Heltec WiFi LoRa 32 V4 and on the Arduino Nesso N1. No extra library is needed for the Heltec;
// the Nesso N1 needs the Arduino_Nesso_N1 library (its LED sits on an I/O expander).
//
// Expected: the LED blinks once per second and the Serial Monitor (115200 baud) prints one line per second.

#include <Arduino.h>
#if defined(ARDUINO_ARDUINO_NESSO_N1)
#include <Arduino_Nesso_N1.h>
#endif

// Both boards use the native USB port of the ESP32. If "USB CDC On Boot" is Disabled in the Tools menu
// (the Heltec V4 default), Serial would be UART0, which is not connected to USB, so open the USB port directly.
#if ARDUINO_USB_CDC_ON_BOOT
  #define CONSOLE Serial
#elif defined(ARDUINO_USB_MODE) && (ARDUINO_USB_MODE == 1)
  HWCDC HWCDCSerial;
  #define CONSOLE HWCDCSerial
#else
  #error "Set Tools > USB CDC On Boot: Enabled, or Tools > USB Mode: Hardware CDC and JTAG."
#endif

static void led(bool on) {
#if defined(ARDUINO_ARDUINO_NESSO_N1)
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);   // inverted logic on the Nesso N1
#else
  digitalWrite(LED_BUILTIN, on ? HIGH : LOW);
#endif
}

void setup() {
  CONSOLE.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  led(false);
}

void loop() {
  static uint32_t n = 0;
  led(true);
  delay(100);
  led(false);
  delay(900);
  CONSOLE.printf("hello %lu chip=%s rev=%u cores=%u cpu=%luMHz flash=%luMB psram=%luKB free_heap=%lu cdc_on_boot=%d\n",
                 (unsigned long)n++, ESP.getChipModel(), (unsigned)ESP.getChipRevision(), (unsigned)ESP.getChipCores(),
                 (unsigned long)ESP.getCpuFreqMHz(), (unsigned long)(ESP.getFlashChipSize() / (1024UL * 1024UL)),
                 (unsigned long)(ESP.getPsramSize() / 1024UL), (unsigned long)ESP.getFreeHeap(), (int)ARDUINO_USB_CDC_ON_BOOT);
}
