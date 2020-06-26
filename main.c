/*
 * deltarobot_servo.c
 *
 * Created: 2020. 03. 13. 19:05:18
 * Author : RozaTamas
 */ 
#define  F_CPU 8000000UL
#include <avr/io.h>
#include <util/delay.h>
#include <avr/interrupt.h>
#include <stdbool.h> 
#include "UART.h"
uint64_t count1=0,count2=0;
uint32_t ford[3];
volatile uint64_t szog1=0,szog2=0,szog3=0;
uint64_t atvett_adat1=0,atvett_adat2=0,atvett_adat3=0;
uint64_t seged1=0;
uint64_t seged2=0;
unsigned int seged3=0;
int main(void)
{
    /* Replace with your application code */
	DDRB=0xff;
	//DDRB=0xf0;
	DDRD=0xf0;
	DDRG=0;
	TCCR1A=0;
	TCCR1B=0b1011;
	OCR1A=2490;	//2490 - 20 ms-es kör
	TIMSK=0b00010001;
	
	
	USART_Init(MYUBRR);
				
	while(1){
		if(count1<7){
		USART_Receive();
		seged1=RXC;			//RXC ha bebillen akkor van fogadás
		if(seged1!=0)
		{	
			count1++;
		}
		if((count1<=3)&&(count1>2))
		{
			atvett_adat1+=UDR0;
			
		}
		if((count1>=4)&&(count1<=6))
		{	
			atvett_adat2+=UDR0;
			PORTD=atvett_adat1&0xf0;
			PORTB=(atvett_adat1<<4)&0xf0;
		}
		if(count1>=6)
		{	
			atvett_adat3+=UDR0;	
		count1++;
		}
		}
		if(count1==7) {
			PORTD=0;
			//sei();
		}
	}
	return 0;
}
ISR(TIMER1_COMPA_vect)
{
	
	seged2++;
	szog1=2575-(atvett_adat1*(17.5));
	if(seged2<50){
	for(int i=szog1;i>0;i--)
	{
		PORTB=1;
		
	}
	PORTB=0;
	
	}
	//szog1=szog1+(2575-szog1);
	if(seged2>50){
		
	for(int j=2575;j>0;j--)
	{
		PORTB=1;
	}
	PORTB=0;
	
	}
}
