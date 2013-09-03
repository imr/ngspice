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

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"


char *INPerrCat(char *a, char *b)
{
    if (a != NULL) {
        if (b == NULL) {        /* a valid, b null, return a */
            return (a);
        } else {                /* both valid  - hard work... */
            register char *errtmp;
            errtmp =
                TMALLOC(char, strlen(a) + strlen(b) + 2);
            (void) strcpy(errtmp, a);
            (void) strcat(errtmp, "\n");
            (void) strcat(errtmp, b);
            FREE(a);
            FREE(b);
            return (errtmp);
        }
    } else                      /* a null, so return b */
        return (b);
}


char *INPstrCat(char *a, char *b, char *c)
{
    if (a != NULL) {
        if (b == NULL) {        /* a valid, b null, return a */
            return (a);
        } else {                /* both valid  - hard work... */
            register char *strtmp;
            strtmp =
                TMALLOC(char, strlen(a) + strlen(b) + 2);
            (void) strcpy(strtmp, a);
            (void) strcat(strtmp, c); /* single character only! */
            (void) strcat(strtmp, b);
            FREE(a);
            FREE(b);
            return (strtmp);
        }
    } else                      /* a null, so return b */
        return (b);
}
