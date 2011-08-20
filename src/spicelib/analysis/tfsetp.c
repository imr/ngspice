/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include <ngspice/ngspice.h>
#include <ngspice/ifsim.h>
#include <ngspice/iferrmsg.h>
#include <ngspice/tfdefs.h>
#include <ngspice/cktdefs.h>

#include "analysis.h"

/* ARGSUSED */
int 
TFsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NG_IGNORE(ckt);

    switch(which) {

    case TF_OUTPOS:
        ((TFan *)anal)->TFoutPos = value->nValue;
        ((TFan *)anal)->TFoutIsV = TRUE;
        ((TFan *)anal)->TFoutIsI = FALSE;
        break;
    case TF_OUTNEG:
        ((TFan *)anal)->TFoutNeg = value->nValue;
        ((TFan *)anal)->TFoutIsV = TRUE;
        ((TFan *)anal)->TFoutIsI = FALSE;
        break;
    case TF_OUTNAME:
        ((TFan *)anal)->TFoutName = value->sValue;
        break;
    case TF_OUTSRC:
        ((TFan *)anal)->TFoutSrc = value->uValue;
        ((TFan *)anal)->TFoutIsV = FALSE;
        ((TFan *)anal)->TFoutIsI = TRUE;
        break;
    case TF_INSRC:
        ((TFan *)anal)->TFinSrc = value->uValue;
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
