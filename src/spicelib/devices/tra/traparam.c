/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "tradefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
TRAparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    TRAinstance *here = (TRAinstance *)inst;
    switch(param) {
        case TRA_RELTOL:
            here->TRAreltol = value->rValue;
            here->TRAreltolGiven = TRUE;
            break;
        case TRA_ABSTOL:
            here->TRAabstol = value->rValue;
            here->TRAabstolGiven = TRUE;
            break;
        case TRA_Z0:
            here->TRAimped = value->rValue;
            here->TRAimpedGiven = TRUE;
            break;
        case TRA_TD:
            here->TRAtd = value->rValue;
            here->TRAtdGiven = TRUE;
            break;
        case TRA_NL:
            here->TRAnl= value->rValue;
            here->TRAnlGiven = TRUE;
            break;
        case TRA_FREQ:
            here->TRAf= value->rValue;
            here->TRAfGiven = TRUE;
            break;
        case TRA_V1:
            here->TRAinitVolt1 = value->rValue;
            here->TRAicV1Given = TRUE;
            break;
        case TRA_I1:
            here->TRAinitCur1 = value->rValue;
            here->TRAicC1Given = TRUE;
            break;
        case TRA_V2:
            here->TRAinitVolt2 = value->rValue;
            here->TRAicV2Given = TRUE;
            break;
        case TRA_I2:
            here->TRAinitCur2 = value->rValue;
            here->TRAicC2Given = TRUE;
            break;
        case TRA_IC:
            switch(value->v.numValue){
                case 4:
                    here->TRAinitCur2 = *(value->v.vec.rVec+3);
                    break;
                case 3:
                    here->TRAinitVolt2 =  *(value->v.vec.rVec+2);
                    break;
                case 2:
                    here->TRAinitCur1 = *(value->v.vec.rVec+1);
                    break;
                case 1:
                    here->TRAinitVolt1 = *(value->v.vec.rVec);
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
