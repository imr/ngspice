/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/


#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "pzdefs.h"
#include "cktdefs.h"


/* ARGSUSED */
int 
PZaskQuest(CKTcircuit *ckt, void *anal, int which, IFvalue *value)
{
    switch(which) {

    case PZ_NODEI:
        value->nValue = (IFnode)CKTnum2nod(ckt,((PZAN*)anal)->PZin_pos);
        break;

    case PZ_NODEG:
        value->nValue = (IFnode)CKTnum2nod(ckt,((PZAN*)anal)->PZin_neg);
        break;

    case PZ_NODEJ:
        value->nValue = (IFnode)CKTnum2nod(ckt,((PZAN*)anal)->PZout_pos);
        break;

    case PZ_NODEK:
        value->nValue = (IFnode)CKTnum2nod(ckt,((PZAN*)anal)->PZout_neg);
        break;

    case PZ_V:
        if( ((PZAN*)anal)->PZinput_type == PZ_IN_VOL) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_I:
        if( ((PZAN*)anal)->PZinput_type == PZ_IN_CUR) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_POL:
        if( ((PZAN*)anal)->PZwhich == PZ_DO_POLES) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_ZER:
        if( ((PZAN*)anal)->PZwhich == PZ_DO_ZEROS) {
            value->iValue=1;
        } else {
            value->iValue=0;
        }
        break;

    case PZ_PZ:
        if( ((PZAN*)anal)->PZwhich == (PZ_DO_POLES | PZ_DO_ZEROS)) {
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
