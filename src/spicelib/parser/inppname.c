/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     * INPpName()
     *
     *  Take a parameter by Name and set it on the specified device 
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "inpxx.h"

int INPpName(char *parm, IFvalue * val, CKTcircuit *ckt, int dev, GENinstance *fast)
		    /* the name of the parameter to set */
		    /* the parameter union containing the value to set */
		    /* the circuit this device is a member of */
		    /* the device type code to the device being parsed */
		    /* direct pointer to device being parsed */
{
    int error;			/* int to store evaluate error return codes in */
    int i;

    for (i = 0; i < *(ft_sim->devices[dev]->numInstanceParms); i++) {
	if (strcmp(parm, ft_sim->devices[dev]->instanceParms[i].keyword) == 0) {
	    error =
		ft_sim->setInstanceParm (ckt, fast,
					 ft_sim->devices[dev]->instanceParms[i].id,
					 val, NULL);
	    if (error)
		return (error);
	    break;
	}
    }
    if (i == *(ft_sim->devices[dev]->numInstanceParms)) {
	return (E_BADPARM);
    }
    return (OK);
}
