/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * CKTpName()
 *
 *  Take a parameter by Name and set it on the specified device
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/sperror.h"


/* the name of the parameter to set */
/* the parameter union containing the value to set */
/* the circuit this device is a member of */
/* the device type code to the device being parsed */
/* the name of the device being parsed */
/* direct pointer to device being parsed */

int
CKTpName(char *parm, IFvalue *val, CKTcircuit *ckt, int dev, char *name, GENinstance **fast)
{
    IFdevice *device = &(DEVices[dev]->DEVpublic);

    IFparm *p = device->instanceParms;
    IFparm *p_end = p + *(device->numInstanceParms);

    NG_IGNORE(name);

    for (; p < p_end; p++)
        if (!strcmp(parm, p->keyword))
            return CKTparam(ckt, *fast, p->id, val, NULL);

    return E_BADPARM;
}
