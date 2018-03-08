/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal device parameters
 * of Current controlled SWitch
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "cswdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CSWask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    CSWinstance *here = (CSWinstance *) inst;
    static char *msg = "Current and power not available in ac analysis";

    NG_IGNORE(select);

    switch (which) {
    case CSW_CONTROL:
        value->uValue = here->CSWcontName;
        return OK;
    case CSW_POS_NODE:
        value->iValue = here->CSWposNode;
        return OK;
    case CSW_NEG_NODE:
        value->iValue = here->CSWnegNode;
        return OK;
    case CSW_CURRENT:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = TMALLOC(char, strlen(msg) + 1);
            errRtn = "CSWask";
            strcpy(errMsg, msg);
            return E_ASKCURRENT;
        } else {
            value->rValue =
                (ckt->CKTrhsOld[here->CSWposNode] -
                 ckt->CKTrhsOld[here->CSWnegNode]) *
                here->CSWcond;
        }
        return OK;
    case CSW_POWER:
        if (ckt->CKTcurrentAnalysis & DOING_AC) {
            errMsg = TMALLOC(char, strlen(msg) + 1);
            errRtn = "CSWask";
            strcpy(errMsg, msg);
            return E_ASKPOWER;
        } else {
            value->rValue =
                (ckt->CKTrhsOld[here->CSWposNode] -
                 ckt->CKTrhsOld[here->CSWnegNode]) *
                (ckt->CKTrhsOld[here->CSWposNode] -
                 ckt->CKTrhsOld[here->CSWnegNode]) *
                here->CSWcond;
        }
        return OK;
    default:
        return E_BADPARM;
    }
}
