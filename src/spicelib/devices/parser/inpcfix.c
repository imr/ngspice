/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <ctype.h>
#include "fteext.h"
#include "inpdefs.h"
#include "inp.h"


void INPcaseFix(register char *string)
{

    while (*string) {
	if (isupper(*string)) {
	    *string = tolower(*string);
	}
	string++;
    }
}
