/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

/* Paolo Nenzi 2001: printnum  does not returns static data anymore. 
 * It is up to the caller to allocate space for strings.
 */

#include <stdio.h>

#include "ngspice/ngspice.h"
#include "printnum.h"

int cp_numdgt = -1;


static inline int get_num_width(double num)
{
    int n;

    if (cp_numdgt > 1) {
        n = cp_numdgt;
    }
    else {
        n = 6;
    }
    if (num < 0.0 && n > 1) {
        n--;
    }

    return n;
} /* end of function get_num_width */



/* This funtion writes num to buf. It can cause buffer overruns. The size of
 * buf is unknown, so cp_numdgt can be large enough to cause sprintf()
 * to write past the end of the array. */
void printnum(char *buf, double num)
{
    int n = get_num_width(num);
    (void) sprintf(buf, "%.*e", n, num);
} /* end of function printnum */



int printnum_ds(DSTRING *p_ds, double num)
{
    const int n = get_num_width(num);
    return ds_cat_printf(p_ds, "%.*e", n, num);
} /* end of function printnum_ds */



