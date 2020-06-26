/*
 * UART.c
 *
 * Created: 2019. 10. 03. 23:40:47
 *  Author: Benedek
 */ 

#include "UART.h"

void USART_Init( unsigned int ubrr )
{
	/* Set baud rate */
	UBRR0H = (unsigned char)(ubrr>>8);
	UBRR0L = (unsigned char)ubrr;
	/* Enable receiver and transmitter */
	UCSR0B = (1<<RXEN)|(1<<TXEN);
	/* Set frame format: 8data, 2stop bit */
	UCSR0C = (1<<USBS)|(3<<UCSZ0);
	UCSR0C = (3<<UCSZ00);
	UCSR0B |=(1<<RXCIE0); //enable usart receiver interrupt
}

void USART_Transmit( unsigned char data )
{
	/* Wait for empty transmit buffer */
	while ( !( UCSR0A & (1<<UDRE)) )
	;
	/* Put data into buffer, sends the data */
	UDR0 = data;
}

unsigned char USART_Receive( void )
{
	/* Wait for data to be received */
	while ( !(UCSR0A & (1<<RXC)) )
	;
	/* Get and return received data from buffer */
	return UDR0;
}

void USART_String(char *string)
{	while ( !(UCSR0A & (1<<UDRE)) )
	;
	while(*string)
	USART_Transmit(*string++);
}