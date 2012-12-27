/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "inpxx.h"


char *INPmkTemp(char *string)
{
    size_t len;
    char *temp;

    len = strlen(string);
    temp = TMALLOC(char, len + 1);
    if (temp != NULL)
	(void) strcpy(temp, string);
    return (temp);

}
