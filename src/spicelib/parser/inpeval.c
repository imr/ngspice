/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <ctype.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"

double INPevaluate(char **line, int *error, int gobble)
		/* non-zero to gobble rest of token, zero to leave it alone */
{
    char *token;
    char *here;
    double mantis;
    int expo1;
    int expo2;
    int sign;
    int expsgn;
    char *tmpline;

    /* setup */
    tmpline = *line;
    if (gobble == 1) {
    /* MW. INPgetUTok should be called with gobble=0 or it make
	 * errors in v(1,2) exp */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
        return (0.0);
    } else if (gobble == 2) {
        *error = INPgetU2Tok(line, &token, 0);
        if (*error)
            return (0.0);
    } else {
        token = *line;
        *error = 0;
    }
    mantis = 0;
    expo1 = 0;
    expo2 = 0;
    sign = 1;
    expsgn = 1;
    /* loop through all of the input token */
    here = token;
    if (*here == '+')
        here++;			/* plus, so do nothing except skip it */
    if (*here == '-') {		/* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }
    if ((*here == 0) || ((!(isdigit(*here))) && (*here != '.'))) {
	/* number looks like just a sign! */
        *error = 1;
	/* back out the 'gettok' operation */
        *line = tmpline;
        if (gobble) {
            FREE(token);
        } else {
            *line = here;
        }
        return (0);
    }
    while (isdigit(*here)) {
        /* digit, so accumulate it. */
        mantis = 10 * mantis + *here - '0';
        here++;
    }
    if (*here == 0) {
        /* reached the end of token - done. */
        if (gobble) {
            FREE(token);
        } else {
            *line = here;
        }
        return ((double) mantis * sign);
    }
    if (*here == ':') {
        /* ':' is no longer used for subcircuit node numbering
           but is part of ternary function a?b:c
           FIXME : subcircuit models still use ':' for model numbering
           Will this hurt somewhere? */
        if (gobble) {
            FREE(token);
        } else {
            *line = here;
        }
        return ((double) mantis * sign);
    }
    /* after decimal point! */
    if (*here == '.') {
        /* found a decimal point! */
        here++;			/* skip to next character */
        if (*here == 0) {
            /* number ends in the decimal point */
            if (gobble) {
                FREE(token);
            } else {
                *line = here;
            }
            return ((double) mantis * sign);
        }
        while (isdigit(*here)) {
            /* digit, so accumulate it. */
            mantis = 10 * mantis + *here - '0';
            expo1 = expo1 - 1;
            if (*here == 0) {
                /* reached the end of token - done. */
                if (gobble) {
                    FREE(token);
                } else {
                    *line = here;
                }
                return (mantis * sign * pow(10., (double) expo1));
            }
            here++;
        }
    }
    /* now look for "E","e",etc to indicate an exponent */
    if ((*here == 'E') || (*here == 'e') || (*here == 'D')
        || (*here == 'd')) {
        /* have an exponent, so skip the e */
        here++;
        /* now look for exponent sign */
        if (*here == '+')
            here++;		/* just skip + */
        if (*here == '-') {
            here++;		/* skip over minus sign */
            expsgn = -1;	/* and make a negative exponent */
            /* now look for the digits of the exponent */
        }
        while (isdigit(*here)) {
            expo2 = 10 * expo2 + *here - '0';
            here++;
        }
    }
    /* now we have all of the numeric part of the number, time to
     * look for the scale factor (alphabetic)
     */
    switch (*here) {
        case 't':
        case 'T':
	        expo1 = expo1 + 12;
            break;
        case 'g':
        case 'G':
            expo1 = expo1 + 9;
            break;
        case 'k':
        case 'K':
            expo1 = expo1 + 3;
            break;
        case 'u':
        case 'U':
            expo1 = expo1 - 6;
            break;
        case 'n':
        case 'N':
            expo1 = expo1 - 9;
            break;
        case 'p':
        case 'P':
            expo1 = expo1 - 12;
            break;
        case 'f':
        case 'F':
            expo1 = expo1 - 15;
            break;
        case 'm':
        case 'M':
        {
            /* special case for m - may be m or mil or meg */
            if (*(here + 1) != 0 && *(here + 2) != 0) {
                /* at least 2 characters, so check them. */
                if ((*(here + 1) == 'E') || (*(here + 1) == 'e')) {
                    if ((*(here + 2) == 'G') || (*(here + 2) == 'g')) {
                        expo1 = expo1 + 6;
                        if (gobble) {
                            FREE(token);
                        } else {
                            *line = here;
                        }
                        return (sign * mantis *
                            pow(10.0,
                            (double) (expo1 + expsgn * expo2)));
                    }
                } else if ((*(here + 1) == 'I') || (*(here + 1) == 'i')) {
                    if ((*(here + 2) == 'L') || (*(here + 2) == 'l')) {
                        expo1 = expo1 - 6;
                        mantis = mantis * 25.4;
                        if (gobble) {
                            FREE(token);
                        } else {
                            *line = here;
                        }
                        return (sign * mantis *
                            pow(10.0,
                            (double) (expo1 + expsgn * expo2)));
                    }
                }
            }
            /* not either special case, so just m => 1e-3 */
            expo1 = expo1 - 3;
	    }
	    break;
    default:
	    break;
    }
    if (gobble) {
        FREE(token);
    } else {
        *line = here;
    }
    return (sign * mantis *
        pow(10.0, (double) (expo1 + expsgn * expo2)));
}
