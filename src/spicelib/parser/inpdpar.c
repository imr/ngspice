/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*
 * INPdevParse()
 *
 *  parse a given input according to the standard rules - look
 *  for the parameters given in the parmlists, In addition,
 *  an optional leading numeric parameter is handled.
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"

char *
INPdevParse(char **line, CKTcircuit *ckt, int dev, GENinstance *fast,
            double *leading, int *waslead, INPtables *tab)
/* the line to parse */
/* the circuit this device is a member of */
/* the device type code to the device being parsed */
/* direct pointer to device being parsed */
/* the optional leading numeric parameter */
/* flag - 1 if leading double given, 0 otherwise */
{
    int error;                  /* int to store evaluate error return codes in */
    char *parm = NULL;
    char *errbuf;
    int i;
    IFvalue *val;
    char *rtn = NULL;

    /* check for leading value */
    *waslead = 0;
    *leading = INPevaluate(line, &error, 1);

    if (error == 0)             /* found a good leading number */
        *waslead = 1;
    else
        *leading = 0.0;

    while (**line != '\0') {
        error = INPgetTok(line, &parm, 1);
        if (!*parm) {
            FREE(parm);
            continue;
        }
        if (error) {
            rtn  = INPerror(error);
            goto quit;
        }
        for (i = 0; i < *(ft_sim->devices[dev]->numInstanceParms); i++) {
            if (strcmp(parm, ft_sim->devices[dev]->instanceParms[i].keyword) == 0) {

                int type;

                val = INPgetValue(ckt, line,
                                  ft_sim->devices[dev]->instanceParms[i].dataType,
                                  tab);
                if (!val) {
                    rtn = INPerror(E_PARMVAL);
                    goto quit;
                }
                error = ft_sim->setInstanceParm (ckt, fast,
                                                 ft_sim->devices[dev]->instanceParms[i].id,
                                                 val, NULL);
                if (error) {
                    rtn = INPerror(error);
                    goto quit;
                }

                /* delete the union val */
                type = ft_sim->devices[dev]->instanceParms[i].dataType;
                type &= IF_VARTYPES;
                if (type == IF_REALVEC)
                    tfree(val->v.vec.rVec);
                else if (type == IF_INTVEC)
                    tfree(val->v.vec.iVec);

                break;
            }
        }
        if (i == *(ft_sim->devices[dev]->numInstanceParms)) {
            errbuf = tprintf(" unknown parameter (%s) \n", parm);
            rtn = errbuf;
            goto quit;
        }
        FREE(parm);
    }

 quit:
    FREE(parm);
    return rtn;
}
