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

#ifdef CIDER
/*
 * Imported from cider file support/strmatch.c
 * Original copyright notice:
 * Author: 1991 David A. Gates, U. C. Berkeley CAD Group
 *
 */
  
  
/*
 * Case-insensitive test of whether p is a prefix of s and at least the
 * first n characters are the same
 */
 
int
cinprefix(p, s, n)
register char *p, *s;
register int n;
{
  if (!p || !s) return( 0 );
 
  while (*p) {
    if ((isupper(*p) ? tolower(*p) : *p) != (isupper(*s) ? tolower(*s) : *s))
      return( 0 );
    p++;
    s++;
    n--;
  }
  if (n > 0)
    return( 0 );
   else
    return( 1 );
 }
  
/*
 * Case-insensitive match of prefix string p against string s
 * returns the number of matching characters
 * 
 */
 
 int
cimatch(p, s)
register char *p, *s;
{
  register int n = 0;
 
  if (!p || !s) return( 0 );
 
  while (*p) {
    if ((isupper(*p) ? tolower(*p) : *p) != (isupper(*s) ? tolower(*s) : *s))
      return( n );
    p++;
    s++;
    n++;
  }
  return( n );
 }  
  
#endif /* CIDER */


/*-------------------------------------------------------------------------*
 * gettok skips over whitespace and returns the next token found.  This is 
 * the original version.  It does not "do the right thing" when you have 
 * parens or commas anywhere in the nodelist.  Note that I left this unmodified
 * since I didn't want to break any fcns which called it from elsewhere than
 * subckt.c.  -- SDB 12.3.2003.
 *-------------------------------------------------------------------------*/
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

/*-------------------------------------------------------------------------*
 * gettok_noparens was added by SDB on 4.21.2003.
 * It acts like gettok, except that it treats parens and commas like
 * whitespace while looking for the POLY token.  That is, it stops 
 * parsing and returns when it finds one of those chars.  It is called from 
 * 'translate' (subckt.c).
 *-------------------------------------------------------------------------*/
char *
gettok_noparens(char **s)
{
    char buf[BSIZE_SP];
    int i = 0;
    char c;

    while ( isspace(**s) )
        (*s)++;   /* iterate over whitespace */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    while ((c = **s) && 
	   !isspace(c) && 
	   ( **s != '(' ) &&
	   ( **s != ')' ) &&
	   ( **s != ',') 
	  )  {
        buf[i++] = *(*s)++;
    }
    buf[i] = '\0';

    /* Now iterate up to next non-whitespace char */
    while ( isspace(**s) )
        (*s)++;  

    return (copy(buf));
}

/*-------------------------------------------------------------------------*
 * gettok_node was added by SDB on 12.3.2003
 * It acts like gettok, except that it treats parens and commas like
 * whitespace (i.e. it ignores them).  Use it when parsing through netnames
 * (node names) since they may be grouped using ( , ).
 *-------------------------------------------------------------------------*/
char *
gettok_node(char **s)
{
    char buf[BSIZE_SP];
    int i = 0;
    char c;

    while (isspace(**s) ||
           ( **s == '(' ) ||
           ( **s == ')' ) ||
           ( **s == ',')
          )
        (*s)++;   /* iterate over whitespace and ( , ) */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    while ((c = **s) && 
	   !isspace(c) && 
	   ( **s != '(' ) &&
	   ( **s != ')' ) &&
	   ( **s != ',') 
	   )  {           /* collect chars until whitespace or ( , ) */
        buf[i++] = *(*s)++;
    }
    buf[i] = '\0';

    /* Now iterate up to next non-whitespace char */
    while (isspace(**s) ||
           ( **s == '(' ) ||
           ( **s == ')' ) ||
           ( **s == ',')
          )
        (*s)++;   /* iterate over whitespace and ( , ) */

    return (copy(buf));
}

/*-------------------------------------------------------------------------*
 * get_l_paren iterates the pointer forward in a string until it hits
 * the position after the next left paren "(".  It returns 0 if it found a left 
 * paren, and 1 if no left paren is found.  It is called from 'translate'
 * (subckt.c).
 *-------------------------------------------------------------------------*/
int
get_l_paren(char **s)
{
    while (**s && ( **s != '(' ) )
        (*s)++;
    if (!**s)
        return (1);
    
    (*s)++;

    if (!**s)
        return (1);
    else
        return 0;
}


/*-------------------------------------------------------------------------*
 * get_r_paren iterates the pointer forward in a string until it hits
 * the position after the next right paren ")".  It returns 0 if it found a right 
 * paren, and 1 if no right paren is found.  It is called from 'translate'
 * (subckt.c).
 *-------------------------------------------------------------------------*/
int
get_r_paren(char **s)
{
    while (**s && ( **s != ')' ) )
        (*s)++;
    if (!**s)
        return (1);

    (*s)++;

    if (!**s)
        return (1);
    else 
        return 0;
}



#ifndef HAVE_BCOPY

#ifndef bcopy
void
bcopy(const void *vfrom, void *vto, size_t num)
{
    register const char *from=vfrom;
    register char *to=vto;
    while (num-- > 0)
        *to++ = *from++;
    return;
}
#endif

#ifndef bzero
/* can't declare void here, because we've already used it in this file */
/* and haven't declared it void before the use */
void
bzero(void *vptr, size_t num)
{
    register char *ptr=vptr;
    while (num-- > 0)
        *ptr++ = '\0';
    return;
}

#endif
#endif
