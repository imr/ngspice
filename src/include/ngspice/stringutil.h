/*************
 * Header file for string.c
 * 1999 E. Rouat
 ************/

#include "ngspice/config.h"
#include "ngspice/bool.h"

#ifndef STRING_H_INCLUDED
#define STRING_H_INCLUDED

int prefix(register char *p, register char *s);
char * copy(const char *str);
char * copy_substring(const char *str, const char *end);
int substring(register char *sub, register char *str);
void appendc(char *s, char c);
int scannum(char *str);
int cieq(register char *p, register char *s);
int ciprefix(register char *p, register char *s);
void strtolower(char *str);
void strtoupper(char *str);
char * stripWhiteSpacesInsideParens(char *str);
char * gettok(char **s);
char * gettok_instance(char **);
char * gettok_char(char **s, char p, bool inc_p, bool nested);


#ifdef CIDER
/* cider integration */ 

int cinprefix(register char *p, register char *s, register int n);
int cimatch(register char *p, register char *s); 
#endif

#ifndef HAVE_BCOPY
void bcopy(const void *from, void *to, size_t num);
#endif

#ifndef HAVE_BZERO
void bzero(void *ptr, size_t num);
#endif

bool isquote(char ch);
bool is_arith_char(char c);
bool str_has_arith_char(char *s);
int get_comma_separated_values( char *values[], char *str );

#endif
