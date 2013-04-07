/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/sen2defs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/suffix.h"
#include "../spicelib/analysis/analysis.h"


int
SENsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    SENstruct *job = (SENstruct *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case SEN_DC:
        if (value->iValue)
            job->SENmode |= DCSEN;
        break;

    case SEN_AC:
        if (value->iValue)
            job->SENmode |= ACSEN;
        break;

    case SEN_TRAN:
        if (value->iValue)
            job->SENmode |= TRANSEN;
        break;

    case SEN_DEV:
        job->SENnumVal += 1;
        if (!job->SENdevices) {
            job->SENdevices = TMALLOC(char *, job->SENnumVal);
            if (job->SENdevices == NULL)
                return E_NOMEM;
            job->SENparmNames = TMALLOC(char *, job->SENnumVal);
            if (job->SENparmNames == NULL)
                return E_NOMEM;
        } else {
            job->SENdevices = TREALLOC(char *, job->SENdevices, job->SENnumVal);
            if (job->SENdevices == NULL)
                return E_NOMEM;
            job->SENparmNames = TREALLOC(char *, job->SENparmNames, job->SENnumVal);
            if (job->SENparmNames == NULL)
                return E_NOMEM;
        }
        job->SENdevices [job->SENnumVal - 1] = value->sValue;
        break;

    case SEN_PARM:
        job->SENparmNames [job->SENnumVal - 1] = value->sValue;
        break;

    default:
        return E_BADPARM;
    }

    return OK;
}


static IFparm SENparms[] = {
    { "dc",     SEN_DC,    IF_SET|IF_FLAG,     "sensitivity in DC analysis" },
    { "op",     SEN_DC,    IF_SET|IF_FLAG,     "sensitivity in DCop analysis" },
    { "ac",     SEN_AC,    IF_SET|IF_FLAG,     "sensitivity in AC analysis" },
    { "tran",   SEN_TRAN,  IF_SET|IF_FLAG,     "sensitivity in transient analysis"},
    { "dev",    SEN_DEV,   IF_SET|IF_INSTANCE, "instance with design param." },
    { "parm",   SEN_PARM,  IF_SET|IF_STRING,   "name of design parameter" },
};

SPICEanalysis SEN2info = {
    {
        "SENS2",
        "Sensitivity analysis",

        NUMELEMS(SENparms),
        SENparms
    },
    sizeof(SENstruct),
    NODOMAIN,
    0,
    SENsetParm,
    SENaskQuest,
    NULL,
    SENstartup
};
