/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 *  provide the error message appropriate for the given error code
 */

#include "ngspice/ngspice.h"

#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/sperror.h"
#include "inpxx.h"

char *INPerror(int type)
{
    char *val;
    char *ebuf;

    /*CDHW Lots of things set errMsg but it is never used  so let's hack it in CDHW*/
    if ( errMsg ) {
        val = errMsg;
        errMsg = NULL;
    } else
        /*CDHW end of hack CDHW*/
        val = copy(SPerror(type));

    if (!val)
        return NULL;

    if (errRtn)
        ebuf = tprintf("%s detected in routine \"%s\"\n", val, errRtn);
    else
        ebuf = tprintf("%s\n", val);

    tfree(val);
    return ebuf;
}
