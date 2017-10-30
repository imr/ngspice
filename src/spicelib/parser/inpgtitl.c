/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*  INPgetTitle(ckt,data)
     *      get the title card from the specified data deck and pass
     *      it through to SPICE-3.
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpstd.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

int INPgetTitle(CKTcircuit **ckt, struct card ** data)
{
    int error;

    error = ft_sim->newCircuit (ckt);
    if (error)
	return (error);
    *data = (*data)->nextcard;
    return (OK);
}
