/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified: 2001 Paolo Nenzi
**********/

/* Paolo Nenzi 2001: printnum  does not returns static data anymore. 
 * It is up to the caller to allocate space for strings.
 */

#include "ngspice.h"
#include "printnum.h"
#include <stdio.h>

int cp_numdgt = -1;

void printnum(char *buf, double num)
{
    int n;

    if (cp_numdgt > 1)
        n = cp_numdgt;
    else
        n = 6;
    if (num < 0.0)
        n--;

    (void) sprintf(buf, "%.*e", n, num);
    
}
