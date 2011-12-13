/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/pzdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
PZsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    PZAN *job = (PZAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case PZ_NODEI:
        job->PZin_pos = value->nValue->number;
        break;

    case PZ_NODEG:
        job->PZin_neg = value->nValue->number;
        break;

    case PZ_NODEJ:
        job->PZout_pos = value->nValue->number;
        break;

    case PZ_NODEK:
        job->PZout_neg = value->nValue->number;
        break;

    case PZ_V:
        if(value->iValue) {
            job->PZinput_type = PZ_IN_VOL;
        }
        break;

    case PZ_I:
        if(value->iValue) {
            job->PZinput_type = PZ_IN_CUR;
        }
        break;

    case PZ_POL:
        if(value->iValue) {
            job->PZwhich = PZ_DO_POLES;
        }
        break;

    case PZ_ZER:
        if(value->iValue) {
            job->PZwhich = PZ_DO_ZEROS;
        }
        break;

    case PZ_PZ:
        if(value->iValue) {
            job->PZwhich = PZ_DO_POLES | PZ_DO_ZEROS;
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

        NUMELEMS(PZparms),
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
