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
sadd(DSTRINGPTR dstr_p, const char *t)
{
    if (ds_cat_str(dstr_p, t) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not add string %s\n", t);
        controlled_exit(-1);
    }
}


/* -----------------------------------------------------------------
 * Function: add character c to dynamic string dstr_p.
 * ----------------------------------------------------------------- */
void
cadd(DSTRINGPTR dstr_p, char c)
{
    if (ds_cat_char(dstr_p, c) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not add character %c\n", c);
        controlled_exit(-1);
    }
}


/* -----------------------------------------------------------------
 * Create copy of the dynamic string.  Dynamic strings are always NULL
 * terminated.
 * ----------------------------------------------------------------- */
void
scopyd(DSTRINGPTR dst, const DSTRINGPTR src)  /* returns success flag */
{
    ds_clear(dst);
    if (ds_cat_ds(dst, src) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not copy string\n");
        controlled_exit(-1);
    }
}


/* -----------------------------------------------------------------
 * Create copy of the string in the dynamic string.  Dynamic strings
 * are always NULL terminated.
 * ----------------------------------------------------------------- */
void
scopys(DSTRINGPTR s, const char *t)     /* returns success flag */
{
    ds_clear(s);
    if (ds_cat_str(s, t) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not copy string %s\n", t);
        controlled_exit(-1);
    }
}


/* Concatentate until stop char (exclusive) or end of string if none given */

void
pscat(DSTRINGPTR dstr_p, const char *t, const char *stop)
{
    if (!stop) { /* locate end of string if no stop char given */
        stop = strchr(t, '\0');
    }
    if (ds_cat_mem(dstr_p, t, (size_t) (stop - t)) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not copy partially string %s\n", t);
        controlled_exit(-1);
    }
} /* end of function pscat */

/* Copy until stop char (exclusive) or end of string if none given */
void
pscopy(DSTRINGPTR dstr_p, const char *t, const char *stop)
{
    ds_clear(dstr_p);
    pscat(dstr_p, t, stop);
} /* end of function pscopy */


bool
alfa(char c)
{
    return
        ((c >= 'a') && (c <= 'z')) ||
        ((c >= 'A') && (c <= 'Z')) ||
        c == '_' || c == '[' || c == ']' || ((c) & 0200);
}


bool
alfanum(char c)
{
    return alfa(c) || ((c >= '0') && (c <= '9'));
}


/* Additionally '-' allowed in subckt name if ps compatible */
bool
alfanumps(char c)
{
    return alfa(c) || ((c >= '0') && (c <= '9')) || c == '-';
}



