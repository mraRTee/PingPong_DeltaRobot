/*
 * lcd_init.c
 *
 * Created: 2019. 09. 23. 20:01:24
 *  Author: Roza Tamas
 */ 
#define F_CPU 8000000UL
#include <util/delay.h>
#include <avr/io.h>
#include <inttypes.h>
#include "lcd_init.h"

#define LCD_CMD_DDR (DDRF)
#define LCD_DATA_DDR (DDRE)

#define LCD_CMD_PORT (PORTF)
#define LCD_DATA_PORT (PORTE)
#define LCD_DATA_PIN (PINE)

#define LCD_RS (PF1)
#define LCD_RW (PF2)
#define LCD_EN (PF3)

#define LCD_E 2//enable
#define LCD_CUR 1//cursor
#define LCD_BL 0//blink



void LCD_init(){

	DDRE|=0xF0; // data kimenetre állítása
	DDRF|=(1<<LCD_RS)|(1<<LCD_RW)|(1<<LCD_EN);	//DDRF bekapcs -  a speciális bitek beállítása 1-re BEKAPCS
	
	//PORTF-ek beállítása - írni szeretnénk, szóval RS és RW 0ra EN 0-1 DE AZT MAJD 3szor kell!!!!
	PORTF&=~(LCD_RW);		//R/W<-0 ->write
	PORTF&=~(1<<LCD_RS);
	//RS <-0 ->parancs
	PORTE=0x20;// Function set -Sets interface data length - EZ NEM KELL FELTÉTELNÜL MERT KÉSÕBB ÚGYIS BEÍRJUK
	LCD_clock();//delay
	LCD_clock();//__| |__
	LCD_clock();//4 bites üzemmód, 8x5pixel, 2soros//üzemmód választás//nem tudjuk, hogy bekapcsolás után éppen milyenben van
	//azért kell itt bevinni értéket, hogy ki lehessen maszkolni
	//itt nem biztos, hogy kell 3szor, de így tuti 1-0 az átmenet
	
	LCD_cmd(0x28);
	LCD_cmd(0x28);
	LCD_cmd(0x28);
	LCD_cmd(0b1111); //display on/off
//	LCD_cmd(0b100);
}

void LCD_busy(void){
	//BF olvasása
	uint8_t busy;
	DDRE&=~(1<<PE7);
	//ott olvassuk majd a BF-et (D7-PE7)
	PORTF&=~(1<<LCD_RS);//Státusz infó
	PORTF|=(1<<LCD_RW);//olvasás
	do
	{busy=0;
		PORTF|=(1<<LCD_EN);//EN<-1
		_delay_us(1);//felfutó
		busy=(LCD_DATA_PIN&(1<<PE7));//átadjuk a BF értékét
		PORTF&=~(1<<LCD_EN);//EN<-0
		_delay_us(1);
		PORTF|=(1<<LCD_EN);//EN<-1
		_delay_us(1);
		PORTF&=~(1<<LCD_EN);//EN<-0
		_delay_us(1);
	}while(busy);

	PORTF&=~(1<<LCD_RW);//R/W<-0 write
	DDRE|=(1<<PE7);//PE7<-1
}

void LCD_cmd(volatile uint8_t cmd)
{
	//itt írni akarunk (nem tudom a maszkolósdi mire van
	LCD_busy();//Megvárjuk még felszabadul
	PORTF&=~(1<<LCD_RS);//Parancs
	PORTF&=~(1<<LCD_RW);//Küldés
	PORTF&=~(1<<LCD_EN);//EN<-0
	PORTE&=~(0xF0);
	PORTE|=(cmd&0xF0);//felsõ 4 bit küldése
	LCD_clock();//__| |__
	PORTE&=~(0xF0);
	PORTE|=((cmd<<4)&0xF0);//alsó 4 bit küldése
	LCD_clock();//__| |__
}

void LCD_data(uint8_t data)
{
	LCD_busy();
	//Megvárjuk még felszabadul
	PORTF|=(1<<LCD_RS);//Adatregiszter
	PORTF&=~(1<<LCD_RW);//írás
	PORTF&=~(1<<LCD_EN);//EN<-0
	PORTE&=~(0xF0);
	PORTE|=(data&0xF0);
	//4 felsõ bit kitétele
	LCD_clock();//__| |__
	PORTE&=~(0xF0);
	PORTE|=((data<<4)&0xF0);
	//alsó 4 bit kitétele
	LCD_clock();//__| |__
}
void LCD_clock(){LCD_CMD_PORT|=(1<<LCD_EN);
	//__|
	_delay_us(2);//
	LCD_CMD_PORT&=~(1<<LCD_EN);// |__
_delay_us(2);}
void LCD_Puts(char*s)
{
	while(*s)
	{
		LCD_data(*s)
		;s++;
	}
}



void fenyujsag()
{
	
	volatile int seged=0;
	char tomb1[16]={"Obudai Egyetem  "};
	char tomb2[17]={" Roza Tamas"};

	LCD_Puts(tomb2);

	while(1)
	{
		
		LCD_cmd(0b10000000); // 3. sor címe
		
		LCD_Puts(tomb1);
		
		
		seged=tomb1[15];
		for(int i=15;i>=0;i--)
		{
			tomb1[i]=tomb1[i-1];
		}
		tomb1[0]=seged;
		
		_delay_ms(400);
	}
}

void egyedi_karakter(uint8_t valaszto)
{
	
	char egyeni1[8] = {
		0b01110,
		0b10101,
		0b10010,
		0b10100,
		0b10111,
		0b10001,
		0b01110,
		0b00000
	};
	char egyeni2[8] = {
		0b01110,
		0b10101,
		0b01001,
		0b00101,
		0b11101,
		0b10001,
		0b01110,
		0b00000
	};
	char egyeni3[8] = {
		0b00000,
		0b00000,
		0b01110,
		0b01110,
		0b01110,
		0b00000,
		0b00000,
		0b00000
	};
	char egyeni4[8] = {
		0b00000,
		0b01110,
		0b10001,
		0b11011,
		0b10001,
		0b10001,
		0b11011,
		0b10101
	};
	
	LCD_cmd(0x40);
	volatile uint8_t feltolt=0;
	while(feltolt<=7)
	{
		switch (valaszto)
		{
		case 1:LCD_data(egyeni1[feltolt]),feltolt++; break;
		case 2:LCD_data(egyeni2[feltolt]),feltolt++; break;
		case 3:LCD_data(egyeni3[feltolt]),feltolt++; break;
		case 4:LCD_data(egyeni4[feltolt]),feltolt++; break;
		}
	}
	LCD_cmd(0x01);
	LCD_cmd(0x80);
	LCD_data(0);

}


void menu(void)
{

	volatile uint16_t szamlalo=0;
	while (1)
	{
		
		LCD_cmd(0b1111); //display on/off
		LCD_cmd(0b100); // entry mode
		_delay_ms(200);
		DDRG=0xff;
		PORTG=0;
		
		if(PING==1)
		{
			szamlalo++;
		}
		if(PING==2)
		{
			szamlalo--;
		}
		switch(szamlalo)
		{
			case 1: LCD_cmd(0b1001111); break;
			case 2: LCD_cmd(0b10011111); break;
			case 3: LCD_cmd(0b1011111); break;
			case 4: LCD_cmd(0b10001111); break;
			default: LCD_cmd(0b10001111); break;
		}
		if(szamlalo==5)
		{
			szamlalo=1;
		}
		if(szamlalo<=0)
		{
			szamlalo=4;
		}
	
		if((PING==0b100)&&(szamlalo==4))
		
		{		
		
			LCD_cmd(0x01);  //disp clear
			LCD_cmd(0b101100); // function set
			LCD_cmd(0x02);
		while(1)
		{
				char egyeni[8] = {
				0b01110,
				0b10101,
				0b10010,
				0b10100,
				0b10111,
				0b10001,
				0b01110,
				0b00000
			};
	
			LCD_cmd(0x40);
			volatile uint8_t feltolt=0;
			while(feltolt<=7)
			{
				LCD_data(egyeni[feltolt]);
				feltolt++;
				
			}
			LCD_cmd(0x80);
			LCD_data(0);
			
		}
		}
		if((PING==0b100)&&(szamlalo==3))
		
		{
			
			LCD_cmd(0x01);  //disp clear
			LCD_cmd(0b101100); // function set
			LCD_cmd(0x02);
			while(1)
			{
				char egyeni[8] = {
					0b01110,
					0b10101,
					0b10010,
					0b10100,
					0b10111,
					0b10001,
					0b01110,
					0b00000
				};
				
				LCD_cmd(0x40);
				volatile uint8_t feltolt=0;
				while(feltolt<=7)
				{
					LCD_data(egyeni[feltolt]);
					feltolt++;
					
				}
				LCD_cmd(0x80);
				LCD_data(0);
				PING=0;
				szamlalo=0;
			}
		}
		if((PING==0b100)&&(szamlalo==2))
		
		{
			
			LCD_cmd(0x01);  //disp clear
			LCD_cmd(0b101100); // function set
			LCD_cmd(0x02);
			while(1)
			{
				char egyeni[8] = {
					0b01110,
					0b10101,
					0b10010,
					0b10100,
					0b10111,
					0b10001,
					0b01110,
					0b00000
				};
				
				LCD_cmd(0x40);
				volatile uint8_t feltolt=0;
				while(feltolt<=7)
				{
					LCD_data(egyeni[feltolt]);
					feltolt++;
					
				}
				LCD_cmd(0x80);
				LCD_data(0);
				PING=0;
				szamlalo=0;
			}
		}
		if((PING==0b100)&&(szamlalo==5))
		
		{
			
			LCD_cmd(0x01);  //disp clear
			LCD_cmd(0b101100); // function set
			LCD_cmd(0x02);
			while(1)
			{
				char egyeni[8] = {
					0b01110,
					0b10101,
					0b10010,
					0b10100,
					0b10111,
					0b10001,
					0b01110,
					0b00000
				};
				
				LCD_cmd(0x40);
				volatile uint8_t feltolt=0;
				while(feltolt<=7)
				{
					LCD_data(egyeni[feltolt]);
					feltolt++;
					
				}
				LCD_cmd(0x80);
				LCD_data(0);
				PING=0;
				szamlalo=0;
			}
		}
	}
}
