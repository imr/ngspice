/*************
 * Header file for lexical.c
 * 1999 E. Rouat
 ************/

#ifndef LEXICAL_H_INCLUDED
#define LEXICAL_H_INCLUDED


wordlist * cp_lexer(char *string);
int inchar(FILE *fp);
int input(FILE *fp);


#endif
