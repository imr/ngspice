/*************
 * Header file for string.c
 * 1999 E. Rouat
 ************/

#ifndef ngspice_STRINGUTIL_H
#define ngspice_STRINGUTIL_H

#include "ngspice/config.h"
#include "ngspice/bool.h"

#include <stdarg.h>


int prefix(const char *p, const char *s);
char * copy(const char *str);
char * copy_substring(const char *str, const char *end);
int substring(const char *sub, const char *str);
void appendc(char *s, char c);
int scannum(char *str);
int cieq(const char *p, const char *s);
int ciprefix(const char *p, const char *s);
void strtolower(char *str);
void strtoupper(char *str);
char * stripWhiteSpacesInsideParens(char *str);
char * gettok(char **s);
char * gettok_instance(char **);
char * gettok_char(char **s, char p, bool inc_p, bool nested);
int model_name_match(const char *token, const char *model_name);

extern char *tvprintf(const char *fmt, va_list args);

#ifdef __GNUC__
extern char *tprintf(const char *fmt, ...) __attribute__ ((format (__printf__, 1, 2)));
#else
extern char *tprintf(const char *fmt, ...);
#endif


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
