/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "inp.h"


char *INPmkTemp(char *string)
{
    int len;
    char *temp;

    len = strlen(string);
    temp = MALLOC(len + 1);
    if (temp != (char *) NULL)
	(void) strcpy(temp, string);
    return (temp);

}
