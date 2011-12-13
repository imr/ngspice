/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/pzdefs.h"
#include "ngspice/cktdefs.h"


/* ARGSUSED */
int 
PZaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    PZAN *job = (PZAN *) anal;

    switch(which) {

    case PZ_NODEI:
        value->nValue = CKTnum2nod(ckt, job->PZin_pos);
        break;

    case PZ_NODEG:
        value->nValue = CKTnum2nod(ckt, job->PZin_neg);
        break;

    case PZ_NODEJ:
        value->nValue = CKTnum2nod(ckt, job->PZout_pos);
        break;

    case PZ_NODEK:
        value->nValue = CKTnum2nod(ckt, job->PZout_neg);
        break;

    case PZ_V:
        if (job->PZinput_type == PZ_IN_VOL) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_I:
        if (job->PZinput_type == PZ_IN_CUR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_POL:
        if (job->PZwhich == PZ_DO_POLES) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_ZER:
        if (job->PZwhich == PZ_DO_ZEROS) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_PZ:
        if (job->PZwhich == (PZ_DO_POLES | PZ_DO_ZEROS)) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
