
#ifndef _UART_H
#define _UART_H

#include <avr/io.h>

#define FOSC 8000000// Clock Speed			//ez eddig 8MHz volt, lehet át kell írni ha valami szar
#define BAUD 9600
#define MYUBRR FOSC/16/BAUD-1

void USART_Init( unsigned int ubrr );
void USART_Transmit( unsigned char data );
void USART_string(char *string);

unsigned char USART_Receive( void );


#endif