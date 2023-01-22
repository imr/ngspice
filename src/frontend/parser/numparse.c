/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* This routine parses a number.  */
#include <ctype.h>
#include <limits.h>
#include <math.h>

#include "ngspice/ngspice.h"
#include "ngspice/bool.h"
#include "ngspice/ftedefs.h"
#include "numparse.h"


bool ft_strictnumparse = FALSE;


static int get_decimal_number(const char **p_str, double *p_val);


/* Parse a number. This will handle things like 10M, etc... If the number
 * must not end before the end of the string, then whole is TRUE.
 * If whole is FALSE and there is more left to the number, the argument
 * is advanced to the end of the word. Returns -1.
 * if no number can be found or if there are trailing characters when
 * whole is TRUE.
 *
 * If ft_strictnumparse is TRUE, and whole is FALSE, the first of the
 * trailing characters must be a '_'.
 *
 * Return codes
 * +1: String represented an integer number that was converted to a double
 *      but which can be stored as an int without loss of data
 * 0: String represented a non-integer number that was converted to a double
 *      that may not be expressed as an integer.
 * -1: Conversion failure
 */
int ft_numparse(char **p_str, bool whole, double *p_val)
{
    double mant;
    double expo;
    const char *p_cur = *p_str; /* position in string */

    /* Parse the mantissa (or decimal number if no exponent) */
    if (get_decimal_number(&p_cur, &mant) < 0) {
        return -1;
    }

    /* Now look for the scale factor or the exponent (can't have both). */
    switch (*p_cur) {
    case 'e':
    case 'E':
        /* Parse another number. Note that a decimal number such as 1.23
         * is allowed as the exponent */
        ++p_cur;
        if (get_decimal_number(&p_cur, &expo) < 0) {
            expo = 0.0;
            --p_cur; /* The "E" was not part of the number */
        }
        break;
    case 't':
    case 'T':
        expo = 12.0;
        ++p_cur;
        break;
    case 'g':
    case 'G':
        expo = 9.0;
        ++p_cur;
        break;
    case 'k':
    case 'K':
        expo = 3.0;
        ++p_cur;
        break;
    case 'u':
    case 'U':
        expo = -6.0;
        ++p_cur;
        break;
    case 'n':
    case 'N':
        expo = -9.0;
        ++p_cur;
        break;
    case 'p':
    case 'P':
        expo = -12.0;
        ++p_cur;
        break;
    case 'f':
    case 'F':
        expo = -15.0;
        ++p_cur;
        break;
    case 'a':
    case 'A':
        expo = -18.0;
        ++p_cur;
        break;
    case 'm':
    case 'M': {
        char ch_cur;

        /* Can be either m, mil, or meg. */
        if (((ch_cur = p_cur[1]) == 'e' || ch_cur == 'E') &&
                (((ch_cur = p_cur[2]) == 'g') || ch_cur == 'G')) {
            expo = 6.0;
            p_cur += 3;
        }
        else if (((ch_cur = p_cur[1]) == 'i' || ch_cur == 'I') &&
                (((ch_cur = p_cur[2]) == 'l') || ch_cur == 'L')) {
            expo = -6.0;
            mant *= 25.4;
            p_cur += 3;
        }
        else { /* plain m for milli */
            expo = -3.0;
            ++p_cur;
        }
        break;
    }
    default:
        expo = 0.0;
    }

    /* p_cur is now pointing to the fist char after the number */
    {
        /* If whole is true, it must be the end of the string */
        const char ch_cur = *p_cur;
        if (whole && ch_cur != '\0') {
            return -1;
        }

        /* If ft_strictnumparse is true, the first character after the
         * string representing the number, if any, must be '_' */
        if (ft_strictnumparse && ch_cur != '\0' && ch_cur != '_') {
            return -1;
        }
    }

    /* Remove the alpha and '_' characters after the number */
    for ( ; ; ++p_cur) {
        const char ch_cur = *p_cur;
        if (!isalpha(ch_cur) && ch_cur != '_') {
            break;
        }
    }

    /* Return results */
    {
       /* Value of number. Ternary operator used to prevent avoidable
        * calls to pow(). */
       const double val = *p_val = mant *
                (expo == 0.0 ? 1.0 : pow(10.0, expo));
        *p_str = (char *) p_cur; /* updated location in string */

        if (ft_parsedb) { /* diagnostics for parsing the number */
            fprintf(cp_err, "numparse: got %e, left = \"%s\"\n",
                    val, p_cur);
        }

        /* Test if the number can be represented as an intger */
        return (double) (int) val == val;
    }
} /* end of function ft_numparse */



/* This function converts the string form of a decimal number at *p_str to
 * its value and returns it in *p_val. The location in *p_str is advanced
 * to the first character after the number if the conversion is OK and
 * is unchanged otherwise.
 *
 * Return codes
 * -1: Conversion failure. *p_val is unchanged
 * 0: Conversion OK. The string was not the representation of an integer
 * +1: Conversion OK. The string was an integer */
static int get_decimal_number(const char **p_str, double *p_val)
{
    double sign = 1.0; /* default sign multiplier if missing is 1.0 */
    const char *p_cur = *p_str;
    char ch_cur = *p_cur; /* 1st char */
    bool f_is_integer = TRUE; /* assume integer */

    /* Test for a sign */
    if (ch_cur == '+') { /* Advance position in string. Sign unchanged */
        ch_cur = *++p_cur;
    }
    else if (ch_cur == '-') { /* Advance position in string. Sign = -1 */
        ch_cur = *++p_cur;
        sign = -1.0;
    }

    /* Ensure string either starts with a digit or a decimal point followed
     * by a digit */
    if ((!isdigit(ch_cur) && ch_cur != '.') ||
            ((ch_cur == '.') && !isdigit_c(p_cur[1]))) {
        return -1;
    }

    /* Parse and compute the number. Assuming 0-9 digits are contiguous and
     * increasing in char representation (true for ASCII and EBCDIC) */
    double val = 0.0;
    for ( ; ; p_cur++) {
        const unsigned int digit =
                (unsigned int) *p_cur - (unsigned int) '0';
        if (digit > 9) { /* not digit */
            break;
        }
        val = val * 10.0 + (double) digit;
    }

    /* Handle fraction, if any */
    if (*p_cur == '.') {
        const char *p0 = ++p_cur; /* start of fraction */
        double numerator = 0.0;

        /* Not an integer expression (even if no fraction after the '.') */
        f_is_integer = FALSE;

        /* Add the fractional part of the number */
        for ( ; ; p_cur++) {
            const unsigned int digit =
                    (unsigned int) *p_cur - (unsigned int) '0';
            if (digit > 9) { /* not digit */
                /* Add fractional part to intergral part from earlier */
                val += numerator * pow(10, (double) (p0 - p_cur));
                break;
            }
            numerator = numerator * 10.0 + (double) digit;
        }
    } /* end of case of fraction */

    /* Return the value and update the position in the string */
    *p_val = sign * val;
    *p_str = p_cur;
    return (int) f_is_integer;
} /* end of function get_decimal_number */



