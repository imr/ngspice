/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     * INPpName()
     *
     *  Take a parameter by Name and set it on the specified device 
     */

#include "ngspice.h"
#include <stdio.h>
#include "cpdefs.h"
#include "fteext.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "inp.h"

int INPpName(char *parm, IFvalue * val, void *ckt, int dev, void *fast)
		    /* the name of the parameter to set */
		    /* the parameter union containing the value to set */
		    /* the circuit this device is a member of */
		    /* the device type code to the device being parsed */
		    /* direct pointer to device being parsed */
{
    int error;			/* int to store evaluate error return codes in */
    int i;

    for (i = 0; i < (*(*(ft_sim->devices)[dev]).numInstanceParms); i++) {
	if (strcmp(parm,
		   ((*(ft_sim->devices)[dev]).instanceParms[i].keyword)) ==
	    0) {
	    error =
		(*(ft_sim->setInstanceParm)) (ckt, fast,
					      (*(ft_sim->devices)[dev]).
					      instanceParms[i].id, val,
					      (IFvalue *) NULL);
	    if (error)
		return (error);
	    break;
	}
    }
    if (i == (*(*(ft_sim->devices)[dev]).numInstanceParms)) {
	return (E_BADPARM);
    }
    return (OK);
}
