/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*
 * This routine gives access to the internal device parameters
 * of Current Controlled Voltage Source
 */

int
ASRCask(CKTcircuit *ckt, GENinstance *instPtr, int which, IFvalue *value, IFvalue *select)
{
    ASRCinstance *here = (ASRCinstance*) instPtr;

    NG_IGNORE(select);

    switch(which) {
    case ASRC_TEMP:
        value->rValue = here->ASRCtemp - CONSTCtoK;
        return(OK);
    case ASRC_DTEMP:
        value->rValue = here->ASRCdtemp;
        return(OK);
    case ASRC_TC1:
        value->rValue = here->ASRCtc1;
        return(OK);
    case ASRC_TC2:
        value->rValue = here->ASRCtc2;
        return(OK);
    case ASRC_M:
        value->rValue = here->ASRCm;
        return(OK);
    case ASRC_CURRENT:
        value->tValue =
            (here->ASRCtype == ASRC_CURRENT) ? here->ASRCtree : NULL;
        return(OK);
    case ASRC_VOLTAGE:
        value->tValue =
            (here->ASRCtype == ASRC_VOLTAGE) ? here->ASRCtree : NULL;
        return(OK);
    case ASRC_POS_NODE:
        value->iValue = here->ASRCposNode;
        return(OK);
    case ASRC_NEG_NODE:
        value->iValue = here->ASRCnegNode;
        return(OK);
    case ASRC_OUTPUTCURRENT:
        if (here->ASRCtype == ASRC_VOLTAGE)
            value->rValue = ckt->CKTrhsOld[here->ASRCbranch];
        else
            value->rValue = here->ASRCprev_value;
        return(OK);
    case ASRC_OUTPUTVOLTAGE:
        value->rValue =
            ckt->CKTrhsOld[here->ASRCposNode] -
            ckt->CKTrhsOld[here->ASRCnegNode];
        return(OK);
    default:
        return(E_BADPARM);
    }
}
