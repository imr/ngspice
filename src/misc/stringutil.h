/*************
 * Header file for string.c
 * 1999 E. Rouat
 ************/

#ifndef STRING_H_INCLUDED
#define STRING_H_INCLUDED

int prefix(register char *p, register char *s);
char * copy(char *str);
int substring(register char *sub, register char *str);
void appendc(char *s, char c);
int scannum(char *str);
int cieq(register char *p, register char *s);
int ciprefix(register char *p, register char *s);
void strtolower(char *str);
char * gettok(char **s);

#if !defined(HAVE_INDEX) && !defined(HAVE_STRCHR)

char * index(register char *s, register char c);
char * rindex(register char *s,register char c );

#endif /* !defined(HAVE_INDEX) && !defined(HAVE_STRCHR) */

#ifndef HAVE_BCOPY

void bcopy(const void  *from, void *to, size_t num);
void bzero(void *ptr, size_t num);

#endif /* HAVE_BCOPY */

#endif
