/*************
 * Header file for quote.c
 * 1999 E. Rouat
 ************/

#ifndef QUOTE_H_INCLUDED
#define QUOTE_H_INCLUDED


void cp_wstrip(char *str);
void cp_quoteword(char *str);
void cp_printword(char *string, FILE *fp);
void cp_striplist(wordlist *wlist);
char * cp_unquote(char *string);


#endif
