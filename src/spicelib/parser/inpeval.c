/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include <ctype.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"

/* Side channel for INPdevParse (inpdpar.c).  When INPevaluate's
 * HSPICE-compat path resolves a bare .param identifier, this points
 * at the resolved name.  INPdevParse reads it once after each
 * INPevaluate call and registers a parse-time binding for later
 * .dc-by-param sweeping; clears to NULL after reading.  All other
 * INPevaluate code paths leave this NULL (callers should still
 * defensively assume any non-NULL value belongs to the most recent
 * call). */
char *inpeval_last_param_name = NULL;


double
INPevaluate(char **line, int *error, int gobble)
/* gobble: non-zero to gobble rest of token, zero to leave it alone */
{
    char *token;
    char *here;
    int sign;
    char *tmpline;

    /* setup */
    tmpline = *line;

    if (gobble) {
        /* MW. INPgetUTok should be called with gobble=0 or it make
         * errors in v(1,2) exp */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
            return (0.0);
    } else {
        token = *line;
        *error = 0;
    }

    sign = 1;

    /* loop through all of the input token */
    here = token;

    if (*here == '+')
        here++;                 /* plus, so do nothing except skip it */
    else if (*here == '-') {    /* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }

    if ((*here == '\0') || ((!(isdigit_c(*here))) && (*here != '.'))) {
        /* HSPICE-compat: token doesn't look like a number — try to
         * resolve it as a bare .param identifier from the global
         * numparam dictionary.  Lets parser callers like
         *   vgs n2 n3 vgswp
         *   .dc vgswp -0.1 vgmax 0.01
         * accept bare `.param` names where they expect a numeric
         * value.  `{vgswp}` / `'vgswp'` already work via the
         * single-to-brace quote pass — this just removes the
         * "must be quoted" requirement.
         *
         * Identifier guard: starts with letter or underscore, then
         * letters / digits / underscore.  Any other character (e.g.
         * leading '$' or operator) falls through to the unchanged
         * error path.
         *
         * On success, also stash the resolved name in
         * `inpeval_last_param_name` so INPdevParse can register a
         * binding for later .dc-by-param sweeping. */
        char *id = here;
        if ((*id >= 'A' && *id <= 'Z') ||
            (*id >= 'a' && *id <= 'z') || *id == '_') {
            char *id_end = id;
            while ((*id_end >= 'A' && *id_end <= 'Z') ||
                   (*id_end >= 'a' && *id_end <= 'z') ||
                   (*id_end >= '0' && *id_end <= '9') ||
                   *id_end == '_')
                id_end++;
            if (*id_end == '\0') {
                extern int nupa_get_real(const char *name, double *value);
                double pv;
                if (nupa_get_real(id, &pv)) {
                    /* Stash a copy of the resolved name.  Static
                     * buffer is fine here: INPevaluate is the
                     * single producer, and consumers (INPdevParse)
                     * read it synchronously before the next call. */
                    static char last_name_buf[128];
                    size_t len = (size_t)(id_end - id);
                    if (len >= sizeof last_name_buf)
                        len = sizeof last_name_buf - 1;
                    memcpy(last_name_buf, id, len);
                    last_name_buf[len] = '\0';
                    extern char *inpeval_last_param_name;
                    inpeval_last_param_name = last_name_buf;
                    if (gobble) {
                        FREE(token);
                    } else {
                        *line = id_end;
                    }
                    return sign * pv;
                }
            }
        }
        /* number looks like just a sign! */
        *error = 1;
        if (gobble) {
            FREE(token);
            /* back out the 'gettok' operation */
            *line = tmpline;
        }
        return (0);
    }

    /* Parse the numeric magnitude with strtod so the result is the
     * correctly-rounded IEEE-754 value.  This matches formula()'s
     * fetchnumber() (xpressn.c, sscanf "%lG") to the bit and removes the
     * 1-ULP drift the old digit-by-digit accumulator + pow(10, expo)
     * produced for long mantissas.  `here` is positioned just past any
     * leading sign, on a digit or '.', so strtod sees the unsigned
     * magnitude and `sign` is applied separately.  A `:` (ternary) or any
     * other non-numeric character naturally terminates strtod, preserving
     * the old early-out behaviour. */
    {
        char *numend;
        double value = strtod(here, &numend);

        /* Fortran-style 'D'/'d' exponent (e.g. 1.5D3) — strtod stops at
         * the 'D', so finish the exponent by hand to keep legacy decks
         * working (the old code accepted D/d as an exponent marker). */
        if (numend != here && (*numend == 'D' || *numend == 'd')) {
            char *expend;
            long fexp = strtol(numend + 1, &expend, 10);
            if (expend != numend + 1) {
                value *= pow(10.0, (double) fexp);
                numend = expend;
            }
        }

        here = numend;          /* at the scale-factor suffix, or token end */

        /* Scale factor (alphabetic suffix).  As before, the suffix
         * character is NOT consumed from the line: `here` is left pointing
         * at it. */
        double scale = 1.0;
        switch (*here) {
        case 't': case 'T': scale = 1.0e12;  break;
        case 'g': case 'G': scale = 1.0e9;   break;
        case 'k': case 'K': scale = 1.0e3;   break;
        case 'u': case 'U': scale = 1.0e-6;  break;
        case 'n': case 'N': scale = 1.0e-9;  break;
        case 'p': case 'P': scale = 1.0e-12; break;
        case 'f': case 'F': scale = 1.0e-15; break;
        case 'a': case 'A': scale = 1.0e-18; break;
        case 'm': case 'M':
            if (((here[1] == 'E') || (here[1] == 'e')) &&
                ((here[2] == 'G') || (here[2] == 'g'))) {
                scale = 1.0e6;          /* Meg */
            } else if (((here[1] == 'I') || (here[1] == 'i')) &&
                       ((here[2] == 'L') || (here[2] == 'l'))) {
                scale = 25.4e-6;        /* Mil */
            } else {
                scale = 1.0e-3;         /* m, milli */
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

        return sign * value * scale;
    }
}


/* In addition to fcn INPevaluate() above, allow values like 4k7,
   similar to the RKM code (used by inp2r) */
double
INPevaluateRKM_R(char** line, int* error, int gobble)
/* gobble: non-zero to gobble rest of token, zero to leave it alone */
{
    char* token;
    char* here;
    double mantis;
    double deci;
    int expo1;
    int expo2;
    int expo3;
    int sign;
    int expsgn;
    char* tmpline;
    bool hasmulti = FALSE;

    /* setup */
    tmpline = *line;

    if (gobble) {
        /* MW. INPgetUTok should be called with gobble=0 or it leads to
         * errors in v(1,2) expression */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
            return (0.0);
    }
    else {
        token = *line;
        *error = 0;
    }

    mantis = 0;
    deci = 0;
    expo1 = 0;
    expo2 = 0;
    expo3 = 0;
    sign = 1;
    expsgn = 1;

    /* loop through all of the input token */
    here = token;

    if (*here == '+')
        here++;                 /* plus, so do nothing except skip it */
    else if (*here == '-') {    /* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }

    if ((*here == '\0') || ((!(isdigit_c(*here))) && (*here != '.') && (*here != 'r'))) {
        /* number looks like just a sign! */
        *error = 1;
        if (gobble) {
            FREE(token);
            /* back out the 'gettok' operation */
            *line = tmpline;
        }
        return (0);
    }

    while (isdigit_c(*here)) {
        /* digit, so accumulate it. */
        mantis = 10 * mantis + *here - '0';
        here++;
    }

    if (*here == '\0') {
        /* reached the end of token - done. */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    if (*here == ':') {
        /* ':' is no longer used for subcircuit node numbering
           but is part of ternary function a?b:c
           FIXME : subcircuit models still use ':' for model numbering
           Will this hurt somewhere? */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    /* after decimal point! */
    if (*here == '.') {
        /* found a decimal point! */
        here++;                 /* skip to next character */

        if (*here == '\0') {
            /* number ends in the decimal point */
            if (gobble) {
                FREE(token);
            }
            else {
                *line = here;
            }
            return ((double)mantis * sign);
        }

        while (isdigit_c(*here)) {
            /* digit, so accumulate it. */
            mantis = 10 * mantis + *here - '0';
            expo1 = expo1 - 1;
            here++;
        }
    }

    /* now look for "E","e",etc to indicate an exponent */
    if ((*here == 'E') || (*here == 'e') || (*here == 'D') || (*here == 'd')) {

        /* have an exponent, so skip the e */
        here++;

        /* now look for exponent sign */
        if (*here == '+')
            here++;             /* just skip + */
        else if (*here == '-') {
            here++;             /* skip over minus sign */
            expsgn = -1;        /* and make a negative exponent */
            /* now look for the digits of the exponent */
        }

        while (isdigit_c(*here)) {
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
        hasmulti = TRUE;
        break;
    case 'g':
    case 'G':
        expo1 = expo1 + 9;
        hasmulti = TRUE;
        break;
    case 'k':
    case 'K':
        expo1 = expo1 + 3;
        hasmulti = TRUE;
        break;
    case 'u':
    case 'U':
        expo1 = expo1 - 6;
        hasmulti = TRUE;
        break;
    case 'r':
    case 'R':
        /* This should be R150, i.e. R followed by an integer number */
        {
            int num;
            char ch;
            if (sscanf(here + 1, "%i%c", &num, &ch) == 1) {
                //expo1 = expo1;
                hasmulti = TRUE;
            }
            else {
                *error = 1;
                if (gobble) {
                    FREE(token);
                    /* back out the 'gettok' operation */
                    *line = tmpline;
                }
                return (0);
            }
        }
        break;
    case 'n':
    case 'N':
        expo1 = expo1 - 9;
        hasmulti = TRUE;
        break;
    case 'p':
    case 'P':
        expo1 = expo1 - 12;
        hasmulti = TRUE;
        break;
    case 'm':
    case 'M':
        if (((here[1] == 'E') || (here[1] == 'e')) &&
            ((here[2] == 'G') || (here[2] == 'g')))
        {
            expo1 = expo1 + 6;  /* Meg */
            here += 2;
            hasmulti = TRUE;
        }
        else if (((here[1] == 'I') || (here[1] == 'i')) &&
            ((here[2] == 'L') || (here[2] == 'l')))
        {
            expo1 = expo1 - 6;
            mantis *= 25.4;     /* Mil */
        }
        else {
            expo1 = expo1 - 3;  /* m, M for milli */
            hasmulti = TRUE;
        }
        break;
    case 'l':
    case 'L':
        expo1 = expo1 - 3;  /* m, milli */
        hasmulti = TRUE;
        break;
    default:
        break;
    }

    /* read a digit after multiplier */
    if (hasmulti) {
        here++;
        while (isdigit_c(*here)) {
            deci = 10 * deci + *here - '0';
            expo3 = expo3 - 1;
            here++;
        }
        mantis = mantis + deci * pow(10.0, (double)expo3);
    }

    if (gobble) {
        FREE(token);
    }
    else {
        *line = here;
    }

    return (sign * mantis *
        pow(10.0, (double)(expo1 + expsgn * expo2)));
}

/* In addition to fcn INPevaluate() above, allow values like 4k7,
   similar to the RKM code (used by inp2r) */
double
INPevaluateRKM_C(char** line, int* error, int gobble)
/* gobble: non-zero to gobble rest of token, zero to leave it alone */
{
    char* token;
    char* here;
    double mantis;
    double deci;
    int expo1;
    int expo2;
    int expo3;
    int sign;
    int expsgn;
    char* tmpline;
    bool hasmulti = FALSE;

    /* setup */
    tmpline = *line;

    if (gobble) {
        /* MW. INPgetUTok should be called with gobble=0 or it make
         * errors in v(1,2) exp */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
            return (0.0);
    }
    else {
        token = *line;
        *error = 0;
    }

    mantis = 0;
    deci = 0;
    expo1 = 0;
    expo2 = 0;
    expo3 = 0;
    sign = 1;
    expsgn = 1;

    /* loop through all of the input token */
    here = token;

    if (*here == '+')
        here++;                 /* plus, so do nothing except skip it */
    else if (*here == '-') {    /* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }

    if ((*here == '\0') || ((!(isdigit_c(*here))) && (*here != '.') && (*here != 'r'))) {
        /* number looks like just a sign! */
        *error = 1;
        if (gobble) {
            FREE(token);
            /* back out the 'gettok' operation */
            *line = tmpline;
        }
        return (0);
    }

    while (isdigit_c(*here)) {
        /* digit, so accumulate it. */
        mantis = 10 * mantis + *here - '0';
        here++;
    }

    if (*here == '\0') {
        /* reached the end of token - done. */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    if (*here == ':') {
        /* ':' is no longer used for subcircuit node numbering
           but is part of ternary function a?b:c
           FIXME : subcircuit models still use ':' for model numbering
           Will this hurt somewhere? */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    /* after decimal point! */
    if (*here == '.') {
        /* found a decimal point! */
        here++;                 /* skip to next character */

        if (*here == '\0') {
            /* number ends in the decimal point */
            if (gobble) {
                FREE(token);
            }
            else {
                *line = here;
            }
            return ((double)mantis * sign);
        }

        while (isdigit_c(*here)) {
            /* digit, so accumulate it. */
            mantis = 10 * mantis + *here - '0';
            expo1 = expo1 - 1;
            here++;
        }
    }

    /* now look for "E","e",etc to indicate an exponent */
    if ((*here == 'E') || (*here == 'e') || (*here == 'D') || (*here == 'd')) {

        /* have an exponent, so skip the e */
        here++;

        /* now look for exponent sign */
        if (*here == '+')
            here++;             /* just skip + */
        else if (*here == '-') {
            here++;             /* skip over minus sign */
            expsgn = -1;        /* and make a negative exponent */
            /* now look for the digits of the exponent */
        }

        while (isdigit_c(*here)) {
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
        hasmulti = TRUE;
        break;
    case 'g':
    case 'G':
        expo1 = expo1 + 9;
        hasmulti = TRUE;
        break;
    case 'k':
    case 'K':
        expo1 = expo1 + 3;
        hasmulti = TRUE;
        break;
    case 'u':
    case 'U':
        expo1 = expo1 - 6;
        hasmulti = TRUE;
        break;
    case 'r':
    case 'R':

        //expo1 = expo1;
        hasmulti = TRUE;
        break;
    case 'n':
    case 'N':
        expo1 = expo1 - 9;
        hasmulti = TRUE;
        break;
    case 'p':
    case 'P':
        expo1 = expo1 - 12;
        hasmulti = TRUE;
        break;
    case 'f':
    case 'F':
        expo1 = expo1 - 15;
        hasmulti = TRUE;
        break;
    case 'a':
    case 'A':
        expo1 = expo1 - 18;
        break;
    case 'm':
    case 'M':
        if (((here[1] == 'E') || (here[1] == 'e')) &&
            ((here[2] == 'G') || (here[2] == 'g')))
        {
            expo1 = expo1 + 6;  /* Meg */
            here += 2;
            hasmulti = TRUE;
        }
        else if (((here[1] == 'I') || (here[1] == 'i')) &&
            ((here[2] == 'L') || (here[2] == 'l')))
        {
            expo1 = expo1 - 6;
            mantis *= 25.4;     /* Mil */
        }
        else {
            expo1 = expo1 - 3;  /* Meg as well */
            hasmulti = TRUE;
        }
        break;
    case 'l':
    case 'L':
        expo1 = expo1 - 3;  /* m, milli */
        hasmulti = TRUE;
        break;
    default:
        break;
    }

    /* read a digit after multiplier */
    if (hasmulti) {
        here++;
        while (isdigit_c(*here)) {
            deci = 10 * deci + *here - '0';
            expo3 = expo3 - 1;
            here++;
        }
        mantis = mantis + deci * pow(10.0, (double)expo3);
    }

    if (gobble) {
        FREE(token);
    }
    else {
        *line = here;
    }

    return (sign * mantis *
        pow(10.0, (double)(expo1 + expsgn * expo2)));
}

/* In addition to fcn INPevaluate() above, allow values like 4k7,
   similar to the RKM code (used by inp2l) */
double
INPevaluateRKM_L(char** line, int* error, int gobble)
/* gobble: non-zero to gobble rest of token, zero to leave it alone */
{
    char* token;
    char* here;
    double mantis;
    double deci;
    int expo1;
    int expo2;
    int expo3;
    int sign;
    int expsgn;
    char* tmpline;
    bool hasmulti = FALSE;

    /* setup */
    tmpline = *line;

    if (gobble) {
        /* MW. INPgetUTok should be called with gobble=0 or it make
         * errors in v(1,2) exp */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
            return (0.0);
    }
    else {
        token = *line;
        *error = 0;
    }

    mantis = 0;
    deci = 0;
    expo1 = 0;
    expo2 = 0;
    expo3 = 0;
    sign = 1;
    expsgn = 1;

    /* loop through all of the input token */
    here = token;

    if (*here == '+')
        here++;                 /* plus, so do nothing except skip it */
    else if (*here == '-') {    /* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }

    if ((*here == '\0') || ((!(isdigit_c(*here))) && (*here != '.') && (*here != 'r'))) {
        /* number looks like just a sign! */
        *error = 1;
        if (gobble) {
            FREE(token);
            /* back out the 'gettok' operation */
            *line = tmpline;
        }
        return (0);
    }

    while (isdigit_c(*here)) {
        /* digit, so accumulate it. */
        mantis = 10 * mantis + *here - '0';
        here++;
    }

    if (*here == '\0') {
        /* reached the end of token - done. */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    if (*here == ':') {
        /* ':' is no longer used for subcircuit node numbering
           but is part of ternary function a?b:c
           FIXME : subcircuit models still use ':' for model numbering
           Will this hurt somewhere? */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    /* after decimal point! */
    if (*here == '.') {
        /* found a decimal point! */
        here++;                 /* skip to next character */

        if (*here == '\0') {
            /* number ends in the decimal point */
            if (gobble) {
                FREE(token);
            }
            else {
                *line = here;
            }
            return ((double)mantis * sign);
        }

        while (isdigit_c(*here)) {
            /* digit, so accumulate it. */
            mantis = 10 * mantis + *here - '0';
            expo1 = expo1 - 1;
            here++;
        }
    }

    /* now look for "E","e",etc to indicate an exponent */
    if ((*here == 'E') || (*here == 'e') || (*here == 'D') || (*here == 'd')) {

        /* have an exponent, so skip the e */
        here++;

        /* now look for exponent sign */
        if (*here == '+')
            here++;             /* just skip + */
        else if (*here == '-') {
            here++;             /* skip over minus sign */
            expsgn = -1;        /* and make a negative exponent */
            /* now look for the digits of the exponent */
        }

        while (isdigit_c(*here)) {
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
        hasmulti = TRUE;
        break;
    case 'g':
    case 'G':
        expo1 = expo1 + 9;
        hasmulti = TRUE;
        break;
    case 'k':
    case 'K':
        expo1 = expo1 + 3;
        hasmulti = TRUE;
        break;
    case 'u':
    case 'U':
        expo1 = expo1 - 6;
        hasmulti = TRUE;
        break;
    case 'r':
    case 'R':

        //expo1 = expo1;
        hasmulti = TRUE;
        break;
    case 'n':
    case 'N':
        expo1 = expo1 - 9;
        hasmulti = TRUE;
        break;
    case 'p':
    case 'P':
        expo1 = expo1 - 12;
        hasmulti = TRUE;
        break;
    case 'f':
    case 'F':
        expo1 = expo1 - 15;
        hasmulti = TRUE;
        break;
    case 'a':
    case 'A':
        expo1 = expo1 - 18;
        break;
    case 'm':
    case 'M':
        if (((here[1] == 'E') || (here[1] == 'e')) &&
            ((here[2] == 'G') || (here[2] == 'g')))
        {
            expo1 = expo1 + 6;  /* Meg */
            here += 2;
            hasmulti = TRUE;
        }
        else if (((here[1] == 'I') || (here[1] == 'i')) &&
            ((here[2] == 'L') || (here[2] == 'l')))
        {
            expo1 = expo1 - 6;
            mantis *= 25.4;     /* Mil */
        }
        else {
            expo1 = expo1 - 3;  /* Meg as well */
            hasmulti = TRUE;
        }
        break;
    case 'l':
    case 'L':
        expo1 = expo1 - 3;  /* m, milli */
        hasmulti = TRUE;
        break;
    default:
        break;
    }

    /* read a digit after multiplier */
    if (hasmulti) {
        here++;
        while (isdigit_c(*here)) {
            deci = 10 * deci + *here - '0';
            expo3 = expo3 - 1;
            here++;
        }
        mantis = mantis + deci * pow(10.0, (double)expo3);
    }

    if (gobble) {
        FREE(token);
    }
    else {
        *line = here;
    }

    return (sign * mantis *
        pow(10.0, (double)(expo1 + expsgn * expo2)));
}


/* This version will move past the scale factor for the rest of the token */
double
INPevaluate2(char** line, int* error, int gobble)
/* gobble: non-zero to gobble rest of token, zero to leave it alone */
{
    char* token;
    char* here;
    double mantis;
    int expo1;
    int expo2;
    int sign;
    int expsgn;
    char* tmpline;

    /* setup */
    tmpline = *line;

    if (gobble) {
        /* MW. INPgetUTok should be called with gobble=0 or it make
         * errors in v(1,2) exp */
        *error = INPgetUTok(line, &token, 0);
        if (*error)
            return (0.0);
    }
    else {
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
        here++;                 /* plus, so do nothing except skip it */
    else if (*here == '-') {    /* minus, so skip it, and change sign */
        here++;
        sign = -1;
    }

    if ((*here == '\0') || ((!(isdigit_c(*here))) && (*here != '.'))) {
        /* number looks like just a sign! */
        *error = 1;
        if (gobble) {
            FREE(token);
            /* back out the 'gettok' operation */
            *line = tmpline;
        }
        return (0);
    }

    while (isdigit_c(*here)) {
        /* digit, so accumulate it. */
        mantis = 10 * mantis + *here - '0';
        here++;
    }

    if (*here == '\0') {
        /* reached the end of token - done. */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    if (*here == ':') {
        /* ':' is no longer used for subcircuit node numbering
           but is part of ternary function a?b:c
           FIXME : subcircuit models still use ':' for model numbering
           Will this hurt somewhere? */
        if (gobble) {
            FREE(token);
        }
        else {
            *line = here;
        }
        return ((double)mantis * sign);
    }

    /* after decimal point! */
    if (*here == '.') {
        /* found a decimal point! */
        here++;                 /* skip to next character */

        if (*here == '\0') {
            /* number ends in the decimal point */
            if (gobble) {
                FREE(token);
            }
            else {
                *line = here;
            }
            return ((double)mantis * sign);
        }

        while (isdigit_c(*here)) {
            /* digit, so accumulate it. */
            mantis = 10 * mantis + *here - '0';
            expo1 = expo1 - 1;
            here++;
        }
    }

    /* now look for "E","e",etc to indicate an exponent */
    if ((*here == 'E') || (*here == 'e') || (*here == 'D') || (*here == 'd')) {

        /* have an exponent, so skip the e */
        here++;

        /* now look for exponent sign */
        if (*here == '+')
            here++;             /* just skip + */
        else if (*here == '-') {
            here++;             /* skip over minus sign */
            expsgn = -1;        /* and make a negative exponent */
            /* now look for the digits of the exponent */
        }

        while (isdigit_c(*here)) {
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
        here++;
        break;
    case 'g':
    case 'G':
        expo1 = expo1 + 9;
        here++;
        break;
    case 'k':
    case 'K':
        expo1 = expo1 + 3;
        here++;
        break;
    case 'u':
    case 'U':
        expo1 = expo1 - 6;
        here++;
        break;
    case 'n':
    case 'N':
        expo1 = expo1 - 9;
        here++;
        break;
    case 'p':
    case 'P':
        expo1 = expo1 - 12;
        here++;
        break;
    case 'f':
    case 'F':
        expo1 = expo1 - 15;
        here++;
        break;
    case 'a':
    case 'A':
        expo1 = expo1 - 18;
        here++;
        break;
    case 'm':
    case 'M':
        if (((here[1] == 'E') || (here[1] == 'e')) &&
            ((here[2] == 'G') || (here[2] == 'g')))
        {
            expo1 = expo1 + 6;  /* Meg */
            here += 3;
        }
        else if (((here[1] == 'I') || (here[1] == 'i')) &&
            ((here[2] == 'L') || (here[2] == 'l')))
        {
            expo1 = expo1 - 6;
            mantis *= 25.4;     /* Mil */
            here += 3;
        }
        else {
            expo1 = expo1 - 3;  /* m, milli */
            here++;
        }
        break;
    default:
        break;
    }

    if (gobble) {
        FREE(token);
    }
    else {
        *line = here;
    }

    return (sign * mantis *
        pow(10.0, (double)(expo1 + expsgn * expo2)));
}



