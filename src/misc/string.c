/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/*
 * String functions
 */

#include "ngspice/ngspice.h"
#include "ngspice/stringutil.h"
#include "ngspice/dstring.h"

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
copy(const char *str)
{
    char *p;
    
    if (!str)
        return NULL;

    if ((p = TMALLOC(char, strlen(str) + 1)) != NULL)
	    (void) strcpy(p, str);
    return(p);
}

char *
copy_substring(const char *str, const char *end)
{
    size_t n = (size_t) (end - str);
    char *p;

    if ((p = TMALLOC(char, n + 1)) != NULL) {
        (void) strncpy(p, str, n);
        p[n] = '\0';
    }
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
            if(isupper(*str))
                *str = (char) tolower(*str);
            str++;
        }
}

void
strtoupper(char *str)
{
    if (str)
        while (*str) {
            if(islower(*str))
                *str = (char) toupper(*str);
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
cinprefix(
register char *p, register char *s,
register int n)
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
cimatch(
register char *p, register char *s)
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
    char c;
    int paren;
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */

    paren = 0;
    while (isspace(**s))
        (*s)++;
    if (!**s)
        return (NULL);
    spice_dstring_init(&buf) ;
    while ((c = **s) != '\0' && !isspace(c)) {
	if (c == '('/*)*/)
	    paren += 1;
	else if (c == /*(*/')')
	    paren -= 1;
	else if (c == ',' && paren < 1)
	    break;
        spice_dstring_append_char( &buf, *(*s)++ ) ;
    }
    while (isspace(**s) || **s == ',')
        (*s)++;
    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
}

/*-------------------------------------------------------------------------*
 * gettok skips over whitespaces or '=' and returns the next token found, 
 * if the token is something like i(xxx), v(yyy), or v(xxx,yyy)
 *   -- h_vogt 10.07.2010.
 *-------------------------------------------------------------------------*/
char *
gettok_iv(char **s)
{
    char c;
    int paren;
    char *token ;             /* return token */
    SPICE_DSTRING buf ;	      /* allow any length string */

    paren = 0;
    while ((isspace(**s)) || (**s=='='))
        (*s)++;
    if ((!**s) || ((**s != 'v') && (**s != 'i') && (**s != 'V') && (**s != 'I')))
        return (NULL);
    // initialize string
    spice_dstring_init(&buf);
    // add v or i to buf
    spice_dstring_append_char( &buf, *(*s)++ ) ;
    while ((c = **s) != '\0') {
        if (c == '('/*)*/)
            paren += 1;
        else if (c == /*(*/')')
            paren -= 1;
        if (isspace(c)) 
            (*s)++;
        else {
            spice_dstring_append_char( &buf, *(*s)++ ) ;
            if (paren == 0) break;
        }
    }
    while (isspace(**s) || **s == ',')
        (*s)++;
    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
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
    char c;
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */

    while ( isspace(**s) )
        (*s)++;   /* iterate over whitespace */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    spice_dstring_init(&buf) ;
    while ((c = **s) != '\0' && 
	   !isspace(c) && 
	   ( **s != '(' ) &&
	   ( **s != ')' ) &&
	   ( **s != ',') 
	  )  {
        spice_dstring_append_char( &buf, *(*s)++ ) ;
    }

    /* Now iterate up to next non-whitespace char */
    while ( isspace(**s) )
        (*s)++;  

    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
}

char *
gettok_instance(char **s)
{
    char c;
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */

    while ( isspace(**s) )
        (*s)++;   /* iterate over whitespace */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    spice_dstring_init(&buf) ;
    while ((c = **s) != '\0' && 
         !isspace(c) && 
         ( **s != '(' ) &&
         ( **s != ')' )
        )  {
        spice_dstring_append_char( &buf, *(*s)++ ) ;
    }

    /* Now iterate up to next non-whitespace char */
    while ( isspace(**s) )
        (*s)++;  

    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
}

/* get the next token starting at next non white spice, stopping
   at p, if inc_p is true, then including p, else excluding p,
   return NULL if p is not found.
   If '}', ']'  or ')' and nested is true, find corresponding p

*/
char *
gettok_char(char **s, char p, bool inc_p, bool nested)
{
    char c;
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */

    while ( isspace(**s) )
        (*s)++;   /* iterate over whitespace */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    spice_dstring_init(&buf) ;
    if (nested && (( p == '}' ) || ( p == ')' ) || ( p == ']'))) {
        char q;
        int count = 0;
        /* find opening bracket */
        if ( p == '}' )
            q = '{';
        else if(p == ']' )
            q = '[';
        else
            q = '(';
        /* add string in front of q, excluding q */
        while ((c = **s) != '\0' && ( **s != q ))  {
            spice_dstring_append_char( &buf, *(*s)++ ) ;
        }
        /* return if nested bracket found, excluding its character */
        while ((c = **s) != '\0')  {
            if (c == q) count++;
            else if (c == p) count--;
            if (count == 0) {
                break;
            }
            spice_dstring_append_char( &buf, *(*s)++ ) ;
        }
    }
    else
        /* just look for p and return string, excluding p */
        while ((c = **s) != '\0' && ( **s != p ))  {
            spice_dstring_append_char( &buf, *(*s)++ ) ;
        }

    if (c == '\0')
        /* p not found */
        return (NULL);

    if (inc_p)
        /* add p */
        spice_dstring_append_char( &buf, *(*s)++ ) ;

    /* Now iterate up to next non-whitespace char */
    while ( isspace(**s) )
        (*s)++;  

    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
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
    char c;
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */

    while (isspace(**s) ||
           ( **s == '(' ) ||
           ( **s == ')' ) ||
           ( **s == ',')
          )
        (*s)++;   /* iterate over whitespace and ( , ) */

    if (!**s)
        return (NULL);  /* return NULL if we come to end of line */

    spice_dstring_init(&buf) ;
    while ((c = **s) != '\0' && 
	   !isspace(c) && 
	   ( **s != '(' ) &&
	   ( **s != ')' ) &&
	   ( **s != ',') 
	   )  {           /* collect chars until whitespace or ( , ) */
        spice_dstring_append_char( &buf, *(*s)++ ) ;
    }

    /* Now iterate up to next non-whitespace char */
    while (isspace(**s) ||
           ( **s == '(' ) ||
           ( **s == ')' ) ||
           ( **s == ',')
          )
        (*s)++;   /* iterate over whitespace and ( , ) */

    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
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

/*-------------------------------------------------------------------------*
 * this function strips all white space inside parens
 * is needed in gettoks (dotcards.c) for right processing of expressions
 * like ".plot v( 5,4) v(6)"
 *-------------------------------------------------------------------------*/
char *
stripWhiteSpacesInsideParens(char *str)
{
    char *token ;				/* return token */
    SPICE_DSTRING buf ;				/* allow any length string */
    int i = 0 ;					/* index into string */

    while ( (str[i] == ' ') || (str[i] == '\t') )
        i++;

    spice_dstring_init(&buf) ;
    for(i=i; str[i]!='\0'; i++) 
    {
        if ( str[i] != '(' ) {
	    spice_dstring_append_char( &buf, str[i] ) ;
        } else {
	    spice_dstring_append_char( &buf, str[i] ) ;
            while ( (str[i++] != ')') ) {
                if ( str[i] != ' ' ) spice_dstring_append_char( &buf, str[i] ) ;
            }
            i--;
        }
    }
    token = copy( spice_dstring_value(&buf) ) ;
    spice_dstring_free(&buf) ;
    return ( token ) ;
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
#endif

#ifndef HAVE_BZERO
#ifndef bzero
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


bool
isquote( char ch )
{
  return ( ch == '\'' || ch == '"' );
}

bool
is_arith_char( char c )
{
  if ( c == '+' || c == '-' || c == '*' || c == '/' || c == '(' || c == ')' || c == '<' ||
       c == '>' || c == '?' || c == '|' || c == '&' || c == '^')
    return TRUE;
  else
    return FALSE;
}

bool
str_has_arith_char( char *s )
{
  while ( *s && *s != '\0' ) {
    if ( is_arith_char(*s) ) return TRUE;
    s++;
  }
  return FALSE;
}

int
get_comma_separated_values( char *values[], char *str ) {
  int count = 0;
  char *ptr, *comma_ptr, keep;
  
  while ( ( comma_ptr = strstr( str, "," ) ) != NULL ) {
    ptr = comma_ptr - 1;
    while ( isspace(*ptr) ) ptr--;
    ptr++; keep = *ptr; *ptr = '\0';
    values[count++] = strdup(str);
    *ptr = keep;
    str = comma_ptr + 1;
    while ( isspace(*str) ) str++;
  }
  values[count++] = strdup(str);
  return count;
}
