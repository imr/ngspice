/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * String functions
 */

#include <config.h>
#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "ngspice.h"
#include "stringutil.h"

int
prefix(register char *p, register char *s)
{
    while (*p && (*p == *s))
        p++, s++;
    if (!*p)
        return (TRUE);
    else
        return (FALSE);
}

/* Create a copy of a string. */

char *
copy(char *str)
{
    char *p;
    
    if ((p = tmalloc(strlen(str) + 1)))
	    (void) strcpy(p, str);
    return(p);
}

/* Determine whether sub is a substring of str. */
/* Like strstr( ) XXX */

int
substring(register char *sub, register char *str)
{
    char *s, *t;

    while (*str) {
        if (*str == *sub) {
	    t = str;
            for (s = sub; *s; s++) {
                if (!*t || (*s != *t++))
                    break;
            }
            if (*s == '\0')
                return (TRUE);
        }
        str++;
    }
    return (FALSE);
}

/* Append one character to a string. Don't check for overflow. */
/* Almost like strcat( ) XXX */

void
appendc(char *s, char c)
{
    while (*s)
        s++;
    *s++ = c;
    *s = '\0';
    return;
}

/* Try to identify an integer that begins a string. Stop when a non-
 * numeric character is reached.
 */
/* Like atoi( ) XXX */

int
scannum(char *str)
{
    int i = 0;

    while(isdigit(*str))
        i = i * 10 + *(str++) - '0';
    return(i);
}

/* Case insensitive str eq. */
/* Like strcasecmp( ) XXX */

int
cieq(register char *p, register char *s)
{
    while (*p) {
        if ((isupper(*p) ? tolower(*p) : *p) !=
            (isupper(*s) ? tolower(*s) : *s))
            return(FALSE);
        p++;
        s++;
    }
    return (*s ? FALSE : TRUE);
}

/* Case insensitive prefix. */

int
ciprefix(register char *p, register char *s)
{
    while (*p) {
        if ((isupper(*p) ? tolower(*p) : *p) !=
            (isupper(*s) ? tolower(*s) : *s))
            return(FALSE);
        p++;
        s++;
    }
    return (TRUE);
}

void
strtolower(char *str)
{
    if (str)
	while (*str) {
	    *str = tolower(*str);
	    str++;
	}
}

char *
gettok(char **s)
{
    char buf[BSIZE_SP];
    int i = 0;
    char c;
    int paren;

    paren = 0;
    while (isspace(**s))
        (*s)++;
    if (!**s)
        return (NULL);
    while ((c = **s) && !isspace(c)) {
	if (c == '('/*)*/)
	    paren += 1;
	else if (c == /*(*/')')
	    paren -= 1;
	else if (c == ',' && paren < 1)
	    break;
        buf[i++] = *(*s)++;
    }
    buf[i] = '\0';
    while (isspace(**s) || **s == ',')
        (*s)++;
    return (copy(buf));
}



#ifndef HAVE_BCOPY

#ifndef bcopy
void
bcopy(register char *from, register char *to, register int num)
{
    while (num-- > 0)
        *to++ = *from++;
    return;
}
#endif

#ifndef bzero
/* can't declare void here, because we've already used it in this file */
/* and haven't declared it void before the use */
int
bzero(register char *ptr, register int num)
{
    while (num-- > 0)
        *ptr++ = '\0';
    return (0);
}

#endif
#endif
