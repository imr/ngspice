/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/


#include <string.h>

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"


/* Print a word  */
void
cp_printword(char *string, FILE *fp)
{
    char *s;

    if (string)
        for (s = string; *s; s++)
            (void) putc((*s), fp);
}


/* Create a copy of the input string removing the enclosing quotes,
 * if they are present */
char *
cp_unquote(const char *p_src)
{
    if (!p_src) { /* case of no string */
        return (char *) NULL;
    }

    const size_t len_src = strlen(p_src); /* input str length */
    size_t len_dst;

    /* If enclosed in quotes locate the source after the quote and
     * make the destination length 2 chars less */
    if (len_src >= 2 && *p_src == '"' && p_src[len_src - 1] == '"') {
        len_dst = len_src - 2;
        ++p_src; /* step past first quote */
    }
    else { /* not enclosed in quotes */
        len_dst = len_src;
    }

    /* Allocate string being returned and fill. */
    char * const p_dst = TMALLOC(char, len_dst + 1);
    strncpy(p_dst, p_src, len_dst);
    p_dst[len_dst] = '\0';

    return p_dst;
} /* end of function cp_unquote */



