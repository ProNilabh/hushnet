// Small text display layer: up to 6 short lines.
// Nesso N1: 1.14 inch LCD through the Arduino_Nesso_N1 library (M5GFX).
// Heltec V4: optional 0.96 inch OLED through U8g2 (off by default, see config.h).
#pragma once
#include "hn_board.h"

#define UI_LINES 6
#define UI_COLS  20

#if defined(BOARD_NESSO_N1) && USE_NESSO_DISPLAY

static NessoDisplay g_lcd;
#define UI_AVAILABLE 1

inline void uiBegin() {
  g_lcd.begin();            // also resets the LCD and switches the backlight on
  g_lcd.setRotation(1);     // landscape, 240 x 135
  g_lcd.fillScreen(TFT_BLACK);
  g_lcd.setTextSize(2);     // 12 x 16 pixel characters: 20 columns
  g_lcd.setTextWrap(false);
}

inline void uiShow(const char lines[UI_LINES][UI_COLS + 1]) {
  g_lcd.startWrite();
  g_lcd.fillScreen(TFT_BLACK);
  for (int i = 0; i < UI_LINES; i++) {
    g_lcd.setTextColor(i == 0 ? TFT_YELLOW : (i == 3 ? TFT_GREEN : TFT_WHITE));
    g_lcd.setCursor(0, 2 + i * 22);
    g_lcd.print(lines[i]);
  }
  g_lcd.endWrite();
}

#elif defined(BOARD_HELTEC_V4) && USE_HELTEC_OLED

#include <U8g2lib.h>
#include <Wire.h>
// SSD1306 compatible OLED on its own I2C pins: SDA 17, SCL 18, reset 21 (core variant: SDA_OLED, SCL_OLED, RST_OLED)
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C g_oled(U8G2_R0, RST_OLED, SCL_OLED, SDA_OLED);
#define UI_AVAILABLE 1

inline void uiBegin() {
  g_oled.begin();
  g_oled.setFont(u8g2_font_6x10_tf);   // 6 x 10 pixel characters: 21 columns, 6 lines
  g_oled.clearBuffer();
  g_oled.sendBuffer();
}

inline void uiShow(const char lines[UI_LINES][UI_COLS + 1]) {
  g_oled.clearBuffer();
  for (int i = 0; i < UI_LINES; i++) g_oled.drawStr(0, 9 + i * 10, lines[i]);
  g_oled.sendBuffer();
}

#else

#define UI_AVAILABLE 0
inline void uiBegin() {
#if defined(BOARD_NESSO_N1)
  pinMode(LCD_CS, OUTPUT);       // the LCD shares the SPI bus with the radio: keep it deselected
  digitalWrite(LCD_CS, HIGH);
#endif
}
inline void uiShow(const char lines[UI_LINES][UI_COLS + 1]) { (void)lines; }

#endif
