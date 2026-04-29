/* Print a floating-point number in engineering notation.
   Documentation: http://www.cs.tut.fi/~jkorpela/c/eng.html
   BSD-style license */

#define PREFIX_START (-24)
/* Smallest power of ten for which there is a prefix defined.
   If the set of prefixes will be extended, change this constant
   and update the table "prefix". */

#include <stdio.h>
#include <math.h>
#include "ngspice/ngspice.h"

/* Print a floating-point number in engineering notation.
   Return string needs to be freed by the caller after its use.
   numeric selects e3, e6, e9  etc. or k, M, G etc.
   If flag bytes is set true, numeric is overwritten, bytes
   in multiples of 1024 are issued using k, M, G, T, P. */
char *eng(double value, int digits, bool numeric, bool bytes)
{
    static char *prefix[] = {
    "y", "z", "a", "f", "p", "n", "u", "m", "",
    "k", "M", "G", "T", "P", "E", "Z", "Y"
};
#define PREFIX_END (PREFIX_START+\
    (int)((sizeof(prefix)/sizeof(char *)-1)*3))


    double display, fract;
    int expof10;
    char *result, *sign;

    if (bytes) {
        int i = 0;

        // Divide by 1024 until the unit is reached
        while (value >= 1024. && i < 5) {
            value /= 1024.;
            i++;
        }
        result = tprintf("%.*g %s", digits - 1, value, prefix[i + 8]);
        return result;
    }

    if(value < 0.0) {
        sign = "-";
        value = -value;
    } else {
        sign = "";
    }

    // correctly round to desired precision
    expof10 = lrint( floor( log10(value) ) );
    value *= pow(10.0, digits - 1 - expof10);

    fract = modf(value, &display);
    if(fract >= 0.5) display += 1.0;

    value = display * pow(10.0, expof10 - digits + 1);


    if(expof10 > 0)
        expof10 = (expof10/3)*3;
    else
        expof10 = ((-expof10+3)/3)*(-3);

    value *= pow(10.0, -expof10);
    if (value >= 1000.0) {
        value /= 1000.0;
        expof10 += 3;
    }
    else if(value >= 100.0)
        digits -= 2;
    else if(value >= 10.0)
        digits -= 1;

    if(numeric || (expof10 < PREFIX_START) || (expof10 > PREFIX_END))
        if (expof10 == 0)
            result = tprintf("%s%.*f", sign, digits-1, value);
        else
            result = tprintf("%s%.*fe%d", sign, digits-1, value, expof10);
    else
        result = tprintf("%s%.*f %s", sign, digits-1, value, prefix[(expof10-PREFIX_START)/3]);

    return result;
}
