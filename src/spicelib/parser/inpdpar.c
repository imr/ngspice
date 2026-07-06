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

/* Side-channel from INPevaluate (inpeval.c) — when INPevaluate
 * resolves a bare identifier as a .param value (HSPICE-compat path),
 * it stashes the parameter's name here so INPdevParse can record a
 * binding for later .dc-by-param sweeping.  Cleared by INPevaluate
 * itself at every entry, so non-bare-param paths never see a stale
 * value.  Read once and ignore otherwise. */
extern char *inpeval_last_param_name;

/* dpar_param_binding records "this V/I source's DC value was set
 * from .param paramName at parse time" — used by the .dc analysis
 * loop to push a swept value through to dependent sources.
 *
 * Owned strings: param_name + dev_name (allocated by us).
 * Lifetime: file-scope list; never freed in practice (one entry per
 * V/I source that uses a bare-param leading value — typically a
 * handful per deck). */
typedef struct dpar_param_binding {
    char *param_name;
    char *dev_name;
    int   dev_type;    /* CKTtypelook code: VSRC or ISRC */
    struct dpar_param_binding *next;
} dpar_param_binding_t;

static dpar_param_binding_t *dpar_binding_head = NULL;

void dpar_register_binding(const char *param_name,
                           const char *dev_name, int dev_type) {
    if (!param_name || !dev_name) return;
    for (dpar_param_binding_t *p = dpar_binding_head; p; p = p->next) {
        if (p->dev_type == dev_type &&
            strcmp(p->param_name, param_name) == 0 &&
            strcmp(p->dev_name, dev_name) == 0)
            return;  /* already registered — netlist re-parse */
    }
    dpar_param_binding_t *b = TMALLOC(dpar_param_binding_t, 1);
    b->param_name = copy(param_name);
    b->dev_name = copy(dev_name);
    b->dev_type = dev_type;
    b->next = dpar_binding_head;
    dpar_binding_head = b;
}

/* Public accessor — called by dctrcurv.c.  Caller iterates bindings
 * for a given param_name and updates the source's DC value field
 * directly. */
dpar_param_binding_t *dpar_first_param_binding(void) {
    return dpar_binding_head;
}

const char *dpar_binding_param_name(const dpar_param_binding_t *b) {
    return b ? b->param_name : NULL;
}
const char *dpar_binding_dev_name(const dpar_param_binding_t *b) {
    return b ? b->dev_name : NULL;
}
int dpar_binding_dev_type(const dpar_param_binding_t *b) {
    return b ? b->dev_type : -1;
}
const dpar_param_binding_t *dpar_binding_next(const dpar_param_binding_t *b) {
    return b ? b->next : NULL;
}

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
    inpeval_last_param_name = NULL;
    *leading = INPevaluate(line, &error, 1);

    if (error == 0)             /* found a good leading number */
        *waslead = 1;
    else
        *leading = 0.0;

    /* If INPevaluate resolved the leading value from a bare .param
     * identifier (HSPICE-compat path), register a binding so the
     * .dc analysis loop can push a swept value through to this
     * device's DC field.  fast->GENname carries the device's
     * canonical instance name (lowercased, e.g. "vgs"). */
    if (*waslead && inpeval_last_param_name && fast && fast->GENname) {
        dpar_register_binding(inpeval_last_param_name, fast->GENname, dev);
    }
    inpeval_last_param_name = NULL;

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
                errbuf = tprintf("  unknown instance parameter (%s) \n", parameter);
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
                rtn = errbuf;
                goto quit;
            }
            /* OSDI models may receive extra instance parameters from PDK
             * subcircuits (e.g. 'total' from TSMC nch_mac) that the model
             * does not define.  Skip the parameter and its value rather than
             * aborting; this matches HSPICE/Spectre behaviour. */
            if (device->registry_entry != NULL) {
                while (isspace((unsigned char)**line)) (*line)++;
                if (**line == '=') {
                    (*line)++;
                    while (isspace((unsigned char)**line)) (*line)++;
                    char *endp;
                    strtod(*line, &endp);
                    if (endp > *line)
                        *line = endp;
                    else
                        while (**line && !isspace((unsigned char)**line)) (*line)++;
                }
                FREE(parm);
                continue;
            }
            errbuf = tprintf("  unknown parameter (%s) \n", parm);
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
