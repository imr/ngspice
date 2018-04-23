/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device parameters
 * of voltage controlled SWitch
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    SWinstance *here = (SWinstance *) inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch (which) {
    case SW_POS_NODE:
        value->iValue = here->SWposNode;
        return OK;
    case SW_NEG_NODE:
        value->iValue = here->SWnegNode;
        return OK;
    case SW_POS_CONT_NODE:
        value->iValue = here->SWposCntrlNode;
        return OK;
    case SW_NEG_CONT_NODE:
        value->iValue = here->SWnegCntrlNode;
        return OK;
    case SW_CURRENT:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = TMALLOC(char, strlen(msg) + 1);
            errRtn = "SWask";
            strcpy(errMsg, msg);
            return E_ASKCURRENT;
        } else {
            value->rValue =
                (ckt->CKTrhsOld[here->SWposNode] -
                 ckt->CKTrhsOld[here->SWnegNode]) *
                here->SWcond;
        }
        return OK;
    case SW_POWER:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = TMALLOC(char, strlen(msg) + 1);
            errRtn = "SWask";
            strcpy(errMsg, msg);
            return E_ASKPOWER;
        } else {
            value->rValue =
                (ckt->CKTrhsOld[here->SWposNode] -
                 ckt->CKTrhsOld[here->SWnegNode]) *
                (ckt->CKTrhsOld[here->SWposNode] -
                 ckt->CKTrhsOld[here->SWnegNode]) *
                here->SWcond;
        }
        return OK;
    default:
        return E_BADPARM;
    }
}
