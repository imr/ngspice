/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "trcvdefs.h"
#include "cktdefs.h"

#include "analysis.h"

/* ARGSUSED */
int 
DCTsetParm(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    TRCV* cv= (TRCV*)anal;
    switch(which) {

    case DCT_START1:
        cv->TRCVvStart[0] = value->rValue;
        cv->TRCVnestLevel = MAX(0,cv->TRCVnestLevel);
        cv->TRCVset[0]=TRUE;
        break;

    case DCT_STOP1:
        cv->TRCVvStop[0] = value->rValue;
        cv->TRCVnestLevel = MAX(0,cv->TRCVnestLevel);
        cv->TRCVset[0]=TRUE;
        break;

    case DCT_STEP1:
        cv->TRCVvStep[0] = value->rValue;
        cv->TRCVnestLevel = MAX(0,cv->TRCVnestLevel);
        cv->TRCVset[0]=TRUE;
        break;

    case DCT_START2:
        cv->TRCVvStart[1] = value->rValue;
        cv->TRCVnestLevel = MAX(1,cv->TRCVnestLevel);
        cv->TRCVset[1]=TRUE;
        break;

    case DCT_STOP2:
        cv->TRCVvStop[1] = value->rValue;
        cv->TRCVnestLevel = MAX(1,cv->TRCVnestLevel);
        cv->TRCVset[1]=TRUE;
        break;

    case DCT_STEP2:
        cv->TRCVvStep[1] = value->rValue;
        cv->TRCVnestLevel = MAX(1,cv->TRCVnestLevel);
        cv->TRCVset[1]=TRUE;
        break;
    
    case DCT_NAME1:
        cv->TRCVvName[0] = value->uValue;
        cv->TRCVnestLevel = MAX(0,cv->TRCVnestLevel);
        cv->TRCVset[0]=TRUE;
        break;

    case DCT_NAME2:
        cv->TRCVvName[1] = value->uValue;
        cv->TRCVnestLevel = MAX(1,cv->TRCVnestLevel);
        cv->TRCVset[1]=TRUE;
        break;

    case DCT_TYPE1:
        cv->TRCVvType[0] = value->iValue;
        cv->TRCVnestLevel = MAX(0,cv->TRCVnestLevel);
        cv->TRCVset[0]=TRUE;
        break;

    case DCT_TYPE2:
        cv->TRCVvType[1] = value->iValue;
        cv->TRCVnestLevel = MAX(1,cv->TRCVnestLevel);
        cv->TRCVset[1]=TRUE;
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

        sizeof(DCTparms)/sizeof(IFparm),
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
