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

static IFparm *
find_instance_parameter(char *name, IFdevice *device)
{
    IFparm *p = device->instanceParms;
    IFparm *p_end = p + *(device->numInstanceParms);

    for (; p < p_end; p++)
        if (strcmp(name, p->keyword) == 0)
            return p;
    return NULL;
}


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
    IFdevice *device = ft_sim->devices[dev];

    int error;                  /* int to store evaluate error return codes in */
    char *parm = NULL;
    char *errbuf;
    IFvalue *val;
    char *rtn = NULL;

    /* check for leading value */
    *waslead = 0;
    *leading = INPevaluate(line, &error, 1);

    if (error == 0)             /* found a good leading number */
        *waslead = 1;
    else
        *leading = 0.0;

    wordlist *x = fast->GENmodPtr->defaults;
    for (; x; x = x->wl_next->wl_next) {
        char *parameter = x->wl_word;
        char *value = x->wl_next->wl_word;

        IFparm *p = find_instance_parameter(parameter, device);

        if (!p) {
            if (cieq(parameter, "$")) {
                errbuf = copy("  unknown parameter ($). Check the compatibility flag!\n");
            }
            else {
                errbuf = tprintf("  unknown parameter (%s) \n", parameter);
            }
            rtn = errbuf;
            goto quit;
        }

        val = INPgetValue(ckt, &value, p->dataType, tab);
        if (!val) {
            rtn = INPerror(E_PARMVAL);
            goto quit;
        }

        error = ft_sim->setInstanceParm (ckt, fast, p->id, val, NULL);
        if (error) {
            rtn = INPerror(error);
            if (rtn && error == E_BADPARM) {
                /* add the parameter name to error message */
                char* extended_rtn = tprintf("%s: %s", p->keyword, rtn);
                tfree(rtn);
                rtn = extended_rtn;
            }
            goto quit;
        }

        /* delete the union val */
        switch (p->dataType & IF_VARTYPES) {
        case IF_REALVEC:
            tfree(val->v.vec.rVec);
            break;
        case IF_INTVEC:
            tfree(val->v.vec.iVec);
            break;
        default:
            break;
        }
    }

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

        IFparm *p = find_instance_parameter(parm, device);

        if (!p) {
            if (eq(parm, "$")) {
                errbuf = copy("  unknown parameter ($). Check the compatibility flag!\n");
            }
            else {
                errbuf = tprintf("  unknown parameter (%s) \n", parm);
            }
            rtn = errbuf;
            goto quit;
        }

        val = INPgetValue(ckt, line, p->dataType, tab);
        if (!val) {
            rtn = INPerror(E_PARMVAL);
            goto quit;
        }
        error = ft_sim->setInstanceParm (ckt, fast, p->id, val, NULL);
        if (error) {
            rtn = INPerror(error);
            goto quit;
        }

        /* delete the union val */
        switch (p->dataType & IF_VARTYPES) {
        case IF_REALVEC:
            tfree(val->v.vec.rVec);
            break;
        case IF_INTVEC:
            tfree(val->v.vec.iVec);
            break;
        default:
            break;
        }

        FREE(parm);
    }

 quit:
    FREE(parm);
    return rtn;
}
