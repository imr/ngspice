/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* 
 *  provide the error message appropriate for the given error code
 */

#include "ngspice.h"
#include <stdio.h>
#include "fteext.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "sperror.h"
#include "inp.h"

char *INPerror(int type)
{
    char *val;
    char *ebuf;

    val = SPerror(type);

    if (!val)
	return (val);

    if (errRtn)
	asprintf(&ebuf, "%s detected in routine \"%s\"\n", val, errRtn);
    else
	asprintf(&ebuf, "%s\n", val);

    return ebuf;
}
