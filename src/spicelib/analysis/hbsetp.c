/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/hbardefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

#ifdef WITH_HB

/* ARGSUSED */
int
HBsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    HBAN *job = (HBAN *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case HB_F1:
        if (value->rValue < 0.0) {
            errMsg = copy("Frequency 1 less than 0 is invalid for HB");
            job->HBFreq1 = 1.0;
            return(E_PARMVAL);
        }

        job->HBFreq1 = value->rValue;
        break;


    case HB_F2:
        if (value->rValue < 0.0) {
            errMsg = copy("Frequency 2 less than 0 is invalid for HB");
            job->HBFreq2 = 1.0;
            return(E_PARMVAL);
        }

        job->HBFreq2 = value->rValue;
        break;



    default:
        return(E_BADPARM);
    }
    return(OK);
}


static IFparm HBparms[] = {
    { "f1",    HB_F1,   IF_SET|IF_ASK|IF_REAL, "fundamental frequency" },
    { "f2",    HB_F2,   IF_SET|IF_ASK|IF_REAL, "second frequency" }
};

SPICEanalysis HBinfo  = {
    {
        "HB",
        "Harmonic Balance analysis",

        NUMELEMS(HBparms),
        HBparms
    },
    sizeof(HBAN),
    FREQUENCYDOMAIN,
    1,
    HBsetParm,
    HBaskQuest,
    NULL,
    HBan
};
#endif
