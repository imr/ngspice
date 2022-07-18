/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

int
INPapName(CKTcircuit *ckt, int type, JOB *analPtr, char *parmname,
	  IFvalue * value)
{
    IFparm *if_parm;

    if (!parmname)
        return (E_BADPARM);

    if (!ft_sim->analyses[type])
        return (E_BADPARM);

    if_parm = ft_find_analysis_parm(type, parmname);

    if (!if_parm) {
        fprintf(cp_err, "\n%s\n", parmname);
        return (E_BADPARM);
    }

    return ft_sim->setAnalysisParm (ckt, analPtr,
				    if_parm->id,
				    value,
				    NULL);
}
