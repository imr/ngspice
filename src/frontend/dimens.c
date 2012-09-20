/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author: 1992 David A. Gates, U. C. Berkeley CAD Group
**********/

/*
 * Read and write dimension/index arrays via strings.
 */

#include "ngspice/ngspice.h"
#include "ngspice/dvec.h"               /* For MAXDIMS */
#include "dimens.h"


/*
 * Create a string of the form "12,1,10".
 */
void
dimstring(int *data, int length, char *retstring)
{
    int  i;
    char buf[BSIZE_SP];


    if (!data || length < 1)
        retstring = "";

    buf[0] = '\0';

    for (i = 0; i < length; i++)
        sprintf(buf + strlen(buf), "%d%s", data[i], (i < length - 1) ? "," : "");

    /* XXX Should I return a copy instead? */
    /* qui ci devo fare una copia */
    strcpy(retstring, buf);
}


/*
 * Create a string of the form "[12][1][10]".
 */
void
indexstring(int *data, int length, char *retstring)
{
    int  i;
    char buf[BSIZE_SP];

    if (!data || length < 1)
        retstring = "";

    buf[0] = '\0';

    for (i = 0; i < length; i++)
        sprintf(buf + strlen(buf), "[%d]", data[i]);

    strcpy(retstring, buf);
}


/*
 * Add one to anstrchr into an array with sizes in dims.
 * Return 1 when all counters overflow at once.
 */
int
incindex(int *counts, int numcounts, int *dims, int numdims)
{
    int i, start;

    if (!counts || numcounts < 1 || !dims || numdims < 1)
        return 0;

    start = numcounts - 1;

    for (i = start; i >= 0; i--)
        if (++counts[i] < dims[i])
            break;      /* This counter is not maxed out. */
        else
            counts[i] = 0;

    if (i == 0)
        return (1);
    else
        return (0);
}


/*
 * Count number of empty dimensions in an array.
 */
int
emptydims(int *data, int length)
{
    int i, numempty = 0;

    for (i = 0; i < length; i++)
        if (data[i] == 0)
            numempty++;

    return (numempty);
}


/*
 * Read a string of one of the following forms into a dimensions array:
 *  [12][1][10]
 *  [12,1,10]
 *  12,1,10
 *  12, 1, 10
 *  12 , 1 , 10
 *  Basically, we require that all brackets be matched, that all numbers
 *  be separated by commas or by "][", that all whitespace is ignored, and
 *  the beginning [ and end ] are ignored if they exist.  The only valid
 *  characters in the string are brackets, commas, spaces, and digits.
 *  If any dimension is blank, its entry in the array is set to 0.
 *
 *  Return 0 on success, 1 on failure.
 */
int
atodims(char *p, int *data, int *outlength)
{
    int length = 0;
    int state = 0;
    int err = 0;
    int needbracket = 0;
    char sep = '\0';

    if (!data || !outlength)
        return 1;

    if (!p) {
        *outlength = 0;
        return 0;
    }

    while (*p && isspace(*p))
        p++;

    if (*p == '[') {
        p++;
        while (*p && isspace(*p))
            p++;
        needbracket = 1;
    }

    while (*p && state != 3) {
        switch (state) {
        case 0: /* p just at or before a number */
            if (length >= MAXDIMS) {
                if (length == MAXDIMS)
                    printf("Error: maximum of %d dimensions allowed.\n",
                           MAXDIMS);
                length += 1;
            } else if (!isdigit(*p)) {
                data[length++] = 0;     /* This position was empty. */
            } else {
                data[length++] = atoi(p);
                while (isdigit(*p))
                    p++;
            }
            state = 1;
            break;

        case 1: /* p after a number, looking for ',' or ']' */
            if (sep == '\0')
                sep = *p;

            if (*p == ']' && *p == sep) {
                p++;
                state = 2;
            } else if (*p == ',' && *p == sep) {
                p++;
                state = 0;
            } else {  /* Funny character following a # */
                break;
            }
            break;

        case 2: /* p after a ']', either at the end or looking for '[' */
            if (*p == '[') {
                p++;
                state = 0;
            } else {
                state = 3;
            }
            break;
        }

        while (*p && isspace(*p))
            p++;
    }

    *outlength = length;

    if (length > MAXDIMS)
        return (1);

    if (state == 3) {  /* We finished with a closing bracket */
        err = !needbracket;
    } else if (*p) {   /* We finished by hitting a bad character after a # */
        err = 1;
    } else {           /* We finished by exhausting the string */
        err = needbracket;
    }

    if (err)
        *outlength = 0;

    return (err);
}


/*
 * Skip to the first character that cannot be part of a dimension string.
 */
char *
skipdims(char *p)
{
    if (!p)
        return NULL;

    while (*p && (*p == '[' || *p == ']' || *p == ',' || isspace(*p) || isdigit(*p)))
        p++;

    return (p);
}
