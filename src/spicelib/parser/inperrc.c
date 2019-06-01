/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
  Concatenate two strings, which have to be defined on the heap.
  If either is NULL, the other is returned. If both are defined,
  a new string is malloced, they are combined, both input strings
  are freed, and the new string is returned.
 */
#include <stdio.h>
#include <string.h>

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "inpxx.h"

static char *INPcat(size_t n_a, const char *a, char sep_char,
        size_t n_b, const char *b);


/* This function returns the non-null string a or b if only one of them
 * is not null. Otherwise it returns NULL if both are null or
 * <a>'\n'<b> if both are non-null. */
char *INPerrCat(char *a, char *b)
{
    return INPstrCat(a, '\n', b);
} /* end of function INPerrCat */



/* This function returns the non-null string a or b if only one of them
 * is not null. Otherwise it returns NULL if both are null or
 * <a><seppchar><b> if both are non-null. */
char *INPstrCat(char *a, char sepchar, char *b)
{
    if (a != NULL) {
        if (b == NULL) { /* a valid, b null, return a */
            return a;
        }
        else { /* both valid  - hard work... */
            char *a_ch_b = INPcat(strlen(a), a, sepchar,
                    strlen(b), b);
            txfree(a);
            txfree(b);
            return a_ch_b;
        }
    }
    else { /* a null, so return b */
        return b;
    }
} /* end of function INPstrCat */



/* This function concatenates strings a and b with sep_char added
 * between them. Strings a and b need not be null-terminated. */
static char *INPcat(size_t n_a, const char *a, char sepchar,
        size_t n_b, const char *b)
{
    char *a_ch_b = TMALLOC(char, n_a + n_b + 2);

    /* Build string. Check not really requied since program exits
     * if allocation in TMALLOC fails but would be if this behavior
     * is changed. */
    if (a_ch_b != (char *) NULL) {
        char *p_cur = a_ch_b;
        (void) memcpy(p_cur, a, n_a);
        p_cur += n_a;
        *p_cur++ = sepchar;
        (void) memcpy(p_cur, b, n_b);
        p_cur += n_b;
        *p_cur = '\0';
    }

    return a_ch_b;
} /* end of function INPcat */



