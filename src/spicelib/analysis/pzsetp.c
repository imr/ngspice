/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "pzdefs.h"
#include "cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
PZsetParm(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case PZ_NODEI:
        ((PZAN*)anal)->PZin_pos = ((CKTnode*)value->nValue)->number;
        break;

    case PZ_NODEG:
        ((PZAN*)anal)->PZin_neg = ((CKTnode*)value->nValue)->number;
        break;

    case PZ_NODEJ:
        ((PZAN*)anal)->PZout_pos = ((CKTnode*)value->nValue)->number;
        break;

    case PZ_NODEK:
        ((PZAN*)anal)->PZout_neg = ((CKTnode*)value->nValue)->number;
        break;

    case PZ_V:
        if(value->iValue) {
            ((PZAN*)anal)->PZinput_type = PZ_IN_VOL;
        }
        break;

    case PZ_I:
        if(value->iValue) {
            ((PZAN*)anal)->PZinput_type = PZ_IN_CUR;
        }
        break;

    case PZ_POL:
        if(value->iValue) {
            ((PZAN*)anal)->PZwhich = PZ_DO_POLES;
        }
        break;

    case PZ_ZER:
        if(value->iValue) {
            ((PZAN*)anal)->PZwhich = PZ_DO_ZEROS;
        }
        break;

    case PZ_PZ:
        if(value->iValue) {
            ((PZAN*)anal)->PZwhich = PZ_DO_POLES | PZ_DO_ZEROS;
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm PZparms[] = {
    { "nodei",  PZ_NODEI,   IF_SET|IF_ASK|IF_NODE,  "" },
    { "nodeg",  PZ_NODEG,   IF_SET|IF_ASK|IF_NODE,  "" },
    { "nodej",  PZ_NODEJ,   IF_SET|IF_ASK|IF_NODE,  "" },
    { "nodek",  PZ_NODEK,   IF_SET|IF_ASK|IF_NODE,  "" },
    { "vol",    PZ_V,       IF_SET|IF_ASK|IF_FLAG,  "" },
    { "cur",    PZ_I,       IF_SET|IF_ASK|IF_FLAG,  "" },
    { "pol",    PZ_POL,     IF_SET|IF_ASK|IF_FLAG,  "" },
    { "zer",    PZ_ZER,     IF_SET|IF_ASK|IF_FLAG,  "" },
    { "pz",     PZ_PZ,      IF_SET|IF_ASK|IF_FLAG,  "" }
};

SPICEanalysis PZinfo  = {
    { 
        "PZ",
        "pole-zero analysis",

        sizeof(PZparms)/sizeof(IFparm),
        PZparms
    },
    sizeof(PZAN),
    NODOMAIN,
    1,
    PZsetParm,
    PZaskQuest,
    NULL,
    PZan
};
