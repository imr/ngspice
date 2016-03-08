/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include <ctype.h>
#include "ngspice/fteext.h"
#include "ngspice/inpdefs.h"
#include "inpxx.h"


void INPcaseFix(register char *string)
{

    while (*string) {
	if (isupper_c(*string)) {
	    *string = tolower_c(*string);
	}
	string++;
    }
}
