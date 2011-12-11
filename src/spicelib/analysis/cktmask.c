/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTmodAsk
     *  Ask questions about a specified device.
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"


/* ARGSUSED */
int
CKTmodAsk(CKTcircuit *ckt, GENmodel *modfast, int which, IFvalue *value, IFvalue *selector)
{
    int type = modfast->GENmodType;

    NG_IGNORE(selector);

    if(DEVices[type]->DEVmodAsk) {
        return( DEVices[type]->DEVmodAsk (ckt,
                modfast, which, value) );
    }
    return(E_BADPARM);
}
