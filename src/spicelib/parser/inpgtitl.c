/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*  INPgetTitle(ckt,data)
     *      get the title card from the specified data deck and pass
     *      it through to SPICE-3.
     */

#include "ngspice.h"
#include <stdio.h>
#include "inpdefs.h"
#include "iferrmsg.h"
#include "cpstd.h"
#include "fteext.h"
#include "inp.h"

int INPgetTitle(void **ckt, card ** data)
{
    int error;

    error = (*(ft_sim->newCircuit)) (ckt);
    if (error)
	return (error);
    *data = (*data)->nextcard;
    return (OK);
}
