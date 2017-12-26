/*       mystring.c                Copyright (C)  2002    Georg Post
 *
 *  This file is part of Numparam, see:  readme.txt
 *  Free software under the terms of the GNU Lesser General Public License
 */

#include "ngspice/ngspice.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>

#include "ngspice/config.h"
#include "ngspice/memory.h"
#ifdef HAS_WINGUI
#include "ngspice/wstdio.h"
#endif

#include "general.h"


/*
 * fetch a human answer to a y/n question from stdin
 * insist on a single non white-space char on a '\n' terminated line
 * return this char or '\n' or EOF
 * return '\0' if the answer doesn't fit this pattern
 */

int
yes_or_no(void)
{
    int first;

    do {
        first = getchar();
        if (first == '\n' || first == EOF)
            return first;
    } while (isspace(first));

    for (;;) {
        int c = getchar();
        if (c == EOF)
            return c;
        if (c == '\n')
            return tolower(first);
        if (!isspace(c))
            first = '\0';
    }
}


bool
ci_prefix(const char *p, const char *s)
{
    while (*p) {
        if ((isupper_c(*p) ? tolower_c(*p) : *p) !=
            (isupper_c(*s) ? tolower_c(*s) : *s))
            return (0);
        p++;
        s++;
    }

    return (1);
}


/*******  Strings ************
 *  are 0-terminated char arrays with a 2-byte trailer: max length.
 *  the string mini-library is "overflow-safe" under these conditions:
 *    use Str(n,s) macro: define and initialize a string s of maxlen n<255
 *    to allocate a string on the heap, use newstring(n).
 *    use maxlen() and length() to retrieve string max and actual length
 *    use: cadd, sadd, scopy, pscopy to manipulate them
 *    never put '\x0' characters inside strings !
 *
 *    the 'killer idea' is the following:
 *    on string overflow and/or on heap allocation failure, a program
 *    MUST die.   Now we only die on a heap failure as with dynamic
 *    string we cannot have a string overflow.
 */

/* -----------------------------------------------------------------
 * Function: add string t to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
void
sadd(SPICE_DSTRINGPTR dstr_p, const char *t)
{
    spice_dstring_append(dstr_p, t, -1);
}


/* -----------------------------------------------------------------
 * Function: add character c to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
void
cadd(SPICE_DSTRINGPTR dstr_p, char c)
{
    char tmp_str[2];
    tmp_str[0] = c;
    tmp_str[1] = '\0';
    spice_dstring_append(dstr_p, tmp_str, -1);
}


/* -----------------------------------------------------------------
 * Create copy of the dynamic string.  Dynamic strings are always NULL
 * terminated.
 * ----------------------------------------------------------------- */
void
scopyd(SPICE_DSTRINGPTR s, SPICE_DSTRINGPTR t)  /* returns success flag */
{
    spice_dstring_reinit(s);
    spice_dstring_append(s, spice_dstring_value(t), -1);
}


/* -----------------------------------------------------------------
 * Create copy of the string in the dynamic string.  Dynamic strings
 * are always NULLterminated.
 * ----------------------------------------------------------------- */
void
scopys(SPICE_DSTRINGPTR s, const char *t)     /* returns success flag */
{
    spice_dstring_reinit(s);
    spice_dstring_append(s, t, -1);
}


/* -----------------------------------------------------------------
 * Create an upper case copy of a string and store it in a dynamic string.
 * Dynamic strings are always NULL * terminated.
 * ----------------------------------------------------------------- */
void
scopy_up(SPICE_DSTRINGPTR dstr_p, const char *str)    /* returns success flag */
{
    char up[2];                 /* short string */
    const char *ptr;            /* position in string */

    spice_dstring_reinit(dstr_p);
    up[1] = '\0';
    for (ptr = str; ptr && *ptr; ptr++) {
        up[0] = toupper_c(*ptr);
        spice_dstring_append(dstr_p, up, 1);
    }
}


/* -----------------------------------------------------------------
 * Create a lower case copy of a string and store it in a dynamic string.
 * Dynamic strings are always NULL * terminated.
 * ----------------------------------------------------------------- */
void
scopy_lower(SPICE_DSTRINGPTR dstr_p, const char *str) /* returns success flag */
{
    char low[2];                /* short string */
    const char *ptr;            /* position in string */

    spice_dstring_reinit(dstr_p);
    low[1] = '\0';
    for (ptr = str; ptr && *ptr; ptr++) {
        low[0] = tolower_c(*ptr);
        spice_dstring_append(dstr_p, low, 1);
    }
}


char *
pscopy(SPICE_DSTRINGPTR dstr_p, const char *t, int leng)
/* partial string copy, with C-based start - Because we now have a 0 based
 * start and string may copy outselves, we may need to restore the first
 * character of the original dstring because resetting string will wipe
 * out first character. */
{
    int i;                      /* counter */
    int stop;                   /* stop value */
    char *s_p;                  /* value of dynamic string */

    stop = (int) strlen(t);

    if (0 < stop) {         /* nothing! */

        if (leng > stop)
            leng = stop;

        _spice_dstring_setlength(dstr_p, leng);
        s_p = spice_dstring_value(dstr_p);

        for (i = 0; i < leng; i++)
            s_p[i] = t[i];

        s_p[leng] = '\0';

    } else {

        s_p = spice_dstring_reinit(dstr_p);

    }

    return s_p;
}


char *
pscopy_up(SPICE_DSTRINGPTR dstr_p, const char *t, int leng)
/* partial string copy to upper case, with C convention for start. */
{
    int i;                      /* counter */
    int stop;                   /* stop value */
    char *s_p;                  /* value of dynamic string */

    stop = (int) strlen(t);

    if (0 < stop) {         /* nothing! */

        if (leng > stop)
            leng = stop;

        _spice_dstring_setlength(dstr_p, leng);
        s_p = spice_dstring_value(dstr_p);

        for (i = 0; i < leng; i++)
            s_p[i] = toupper_c(t[i]);

        s_p[leng] = '\0';

    } else {

        s_p = spice_dstring_reinit(dstr_p);

    }

    return s_p;
}


bool
alfa(char c)
{
    return
        ((c >= 'a') && (c <= 'z')) ||
        ((c >= 'A') && (c <= 'Z')) ||
        c == '_' || c == '[' || c == ']';
}


bool
alfanum(char c)
{
    return alfa(c) || ((c >= '0') && (c <= '9'));
}
