/*
 * lcd_init.h
 *
 * Created: 2019. 09. 23. 20:00:39
 *  Author: Roza Tamas
 */ 
#include <inttypes.h>

#ifndef LCD_INIT_H_
#define LCD_INIT_H_


void LCD_init();
void LCD_cmd(uint8_t cmd);
void LCD_data(uint8_t data);
void LCD_clock();
void LCD_Puts(char*s);
void menu(void);
void timer(void);
void fenyujsag(void);
void egyedi_karakter(uint8_t valaszto);



#endif /* LCD_INIT_H_ */