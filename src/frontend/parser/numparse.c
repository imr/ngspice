/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* This routine parses a number.  */

#include "ngspice/config.h"
#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/ftedefs.h"
#include "numparse.h"


static double
power10(double num)   /* Chris Inbody */
{
    double d = 1.0;

    while (num-- > 0)
        d *= 10.0;
    return (d);
}


bool ft_strictnumparse = FALSE;


/* Parse a number. This will handle things like 10M, etc... If the number
 * must not end before the end of the string, then whole is TRUE.
 * If whole is FALSE and there is more left to the number, the argument
 * is advanced to the end of the word. Returns NULL
 * if no number can be found or if there are trailing characters when
 * whole is TRUE.
 *
 * If ft_strictnumparse is TRUE, and whole is FALSE, the first of the
 * trailing characters must be a '_'.  */

double *
ft_numparse(char **s, bool whole)
{
    double mant = 0.0;
    int sign = 1, exsign = 1, p;
    double expo = 0.0;
    static double num;
    char *string = *s;

    /* See if the number begins with + or -. */
    if (*string == '+') {
        string++;
    } else if (*string == '-') {
        string++;
        sign = -1;
    }

    /* We don't want to recognise "P" as 0P, or .P as 0.0P... */
    if ((!isdigit(*string) && *string != '.') ||
        ((*string == '.') && !isdigit(string[1])))
        return (NULL);

    /* Now accumulate a number. Note ascii dependencies here... */
    while (isdigit(*string))
        mant = mant * 10.0 + (*string++ - '0');

    /* Now maybe a decimal point. */
    if (*string == '.') {
        string++;
        p = 1;
        while (isdigit(*string))
            mant += (*string++ - '0') / power10(p++);
    }

    /* Now look for the scale factor or the exponent (can't have both). */
    switch (*string) {
    case 'e':
    case 'E':
        /* Parse another number. */
        string++;
        if (*string == '+') {
            exsign = 1;
            string++;
        } else if (*string == '-') {
            exsign = -1;
            string++;
        }
        while (isdigit(*string))
            expo = expo * 10.0 + (*string++ - '0');
        if (*string == '.') {
            string++;
            p = 1;
            while (isdigit(*string))
                expo += (*string++ - '0') / power10(p++);
        }
        expo *= exsign;
        break;
    case 't':
    case 'T':
        expo = 12.0;
        string++;
        break;
    case 'g':
    case 'G':
        expo = 9.0;
        string++;
        break;
    case 'k':
    case 'K':
        expo = 3.0;
        string++;
        break;
    case 'u':
    case 'U':
        expo = -6.0;
        string++;
        break;
    case 'n':
    case 'N':
        expo = -9.0;
        string++;
        break;
    case 'p':
    case 'P':
        expo = -12.0;
        string++;
        break;
    case 'f':
    case 'F':
        expo = -15.0;
        string++;
        break;
    case 'm':
    case 'M':
        /* Can be either m, mil, or meg. */
        if (string[1] && string[2] &&
            ((string[1] == 'e') || (string[1] == 'E')) &&
            ((string[2] == 'g') || (string[2] == 'G')))
        {
            expo = 6.0;
            string += 3;
        } else if (string[1] && string[2] &&
                   ((string[1] == 'i') || (string[1] == 'I')) &&
                   ((string[2] == 'l') || (string[2] == 'L')))
        {
            expo = -6.0;
            mant *= 25.4;
            string += 3;
        } else {
            expo = -3.0;
            string++;
        }
        break;
    }

    if (whole && *string != '\0') {
        return (NULL);
    } else if (ft_strictnumparse && *string && isdigit(string[-1])) {
        if (*string == '_')
            while (isalpha(*string) || (*string == '_'))
                string++;
        else
            return (NULL);
    } else {
        while (isalpha(*string) || (*string == '_'))
            string++;
    }
    *s = string;
    num = sign * mant * pow(10.0, expo);
    if (ft_parsedb)
        fprintf(cp_err, "numparse: got %e, left = %s\n", num, *s);
    return (&num);
}
