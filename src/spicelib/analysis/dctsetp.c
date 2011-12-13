/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/trcvdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
DCTsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    TRCV *job = (TRCV *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case DCT_START1:
        job->TRCVvStart[0] = value->rValue;
        job->TRCVnestLevel = MAX(0, job->TRCVnestLevel);
        job->TRCVset[0] = TRUE;
        break;

    case DCT_STOP1:
        job->TRCVvStop[0] = value->rValue;
        job->TRCVnestLevel = MAX(0, job->TRCVnestLevel);
        job->TRCVset[0] = TRUE;
        break;

    case DCT_STEP1:
        job->TRCVvStep[0] = value->rValue;
        job->TRCVnestLevel = MAX(0, job->TRCVnestLevel);
        job->TRCVset[0] = TRUE;
        break;

    case DCT_START2:
        job->TRCVvStart[1] = value->rValue;
        job->TRCVnestLevel = MAX(1, job->TRCVnestLevel);
        job->TRCVset[1] = TRUE;
        break;

    case DCT_STOP2:
        job->TRCVvStop[1] = value->rValue;
        job->TRCVnestLevel = MAX(1, job->TRCVnestLevel);
        job->TRCVset[1] = TRUE;
        break;

    case DCT_STEP2:
        job->TRCVvStep[1] = value->rValue;
        job->TRCVnestLevel = MAX(1, job->TRCVnestLevel);
        job->TRCVset[1] = TRUE;
        break;
    
    case DCT_NAME1:
        job->TRCVvName[0] = value->uValue;
        job->TRCVnestLevel = MAX(0, job->TRCVnestLevel);
        job->TRCVset[0] = TRUE;
        break;

    case DCT_NAME2:
        job->TRCVvName[1] = value->uValue;
        job->TRCVnestLevel = MAX(1, job->TRCVnestLevel);
        job->TRCVset[1] = TRUE;
        break;

    case DCT_TYPE1:
        job->TRCVvType[0] = value->iValue;
        job->TRCVnestLevel = MAX(0, job->TRCVnestLevel);
        job->TRCVset[0] = TRUE;
        break;

    case DCT_TYPE2:
        job->TRCVvType[1] = value->iValue;
        job->TRCVnestLevel = MAX(1, job->TRCVnestLevel);
        job->TRCVset[1] = TRUE;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm DCTparms[] = {
    { "start1",     DCT_START1, IF_SET|IF_REAL,     "starting voltage/current"},
    { "stop1",      DCT_STOP1,  IF_SET|IF_REAL,     "ending voltage/current" },
    { "step1",      DCT_STEP1,  IF_SET|IF_REAL,     "voltage/current step" },
    { "start2",     DCT_START2, IF_SET|IF_REAL,     "starting voltage/current"},
    { "stop2",      DCT_STOP2,  IF_SET|IF_REAL,     "ending voltage/current" },
    { "step2",      DCT_STEP2,  IF_SET|IF_REAL,     "voltage/current step" },
    { "name1",      DCT_NAME1,  IF_SET|IF_INSTANCE, "name of source to step" },
    { "name2",      DCT_NAME2,  IF_SET|IF_INSTANCE, "name of source to step" },
    { "type1",      DCT_TYPE1,  IF_SET|IF_INTEGER,  "type of source to step" },
    { "type2",      DCT_TYPE2,  IF_SET|IF_INTEGER,  "type of source to step" }
};

SPICEanalysis DCTinfo  = {
    { 
        "DC",
        "D.C. Transfer curve analysis",

        NUMELEMS(DCTparms),
        DCTparms
    },
    sizeof(TRCV),
    SWEEPDOMAIN,
    1,
    DCTsetParm,
    DCTaskQuest,
    NULL,
    DCtrCurv
};
