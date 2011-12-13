/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/tfdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
TFsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    TFan *job = (TFan *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case TF_OUTPOS:
        job->TFoutPos = value->nValue;
        job->TFoutIsV = TRUE;
        job->TFoutIsI = FALSE;
        break;
    case TF_OUTNEG:
        job->TFoutNeg = value->nValue;
        job->TFoutIsV = TRUE;
        job->TFoutIsI = FALSE;
        break;
    case TF_OUTNAME:
        job->TFoutName = value->sValue;
        break;
    case TF_OUTSRC:
        job->TFoutSrc = value->uValue;
        job->TFoutIsV = FALSE;
        job->TFoutIsI = TRUE;
        break;
    case TF_INSRC:
        job->TFinSrc = value->uValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm TFparms[] = {
    { "outpos",      TF_OUTPOS,     IF_SET|IF_NODE, "Positive output node" },
    { "outneg",      TF_OUTNEG,     IF_SET|IF_NODE, "Negative output node" },
    { "outname",     TF_OUTNAME,    IF_SET|IF_STRING,"Name of output variable"},
    { "outsrc",      TF_OUTSRC,     IF_SET|IF_INSTANCE, "Output source" },
    { "insrc",       TF_INSRC,      IF_SET|IF_INSTANCE, "Input source" }
};

SPICEanalysis TFinfo  = {
    { 
        "TF",
        "transfer function analysis",

        NUMELEMS(TFparms),
        TFparms
    },
    sizeof(TFan),
    NODOMAIN,
    0,
    TFsetParm,
    TFaskQuest,
    NULL,
    TFanal
};
