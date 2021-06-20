/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Kanwar Jit Singh
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "asrcdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
ASRCparam(int param, IFvalue *value, GENinstance *fast, IFvalue *select)
{
    ASRCinstance *here = (ASRCinstance*) fast;

    NG_IGNORE(select);

    switch (param) {
    case ASRC_VOLTAGE:
        here->ASRCtype = ASRC_VOLTAGE;
        here->ASRCtree = value->tValue;
        break;
    case ASRC_CURRENT:
        here->ASRCtype = ASRC_CURRENT;
        here->ASRCtree = value->tValue;
        break;
    case ASRC_TC1:
        here->ASRCtc1 = value->rValue;
        here->ASRCtc1Given = TRUE;
        break;
    case ASRC_TC2:
        here->ASRCtc2 = value->rValue;
        here->ASRCtc2Given = TRUE;
        break;
    case ASRC_M:
        here->ASRCm = value->rValue;
        here->ASRCmGiven = TRUE;
        break;
    case ASRC_RTC:
        here->ASRCreciproctc = value->iValue;
        here->ASRCreciproctcGiven = TRUE;
        break;
    case ASRC_RM:
        here->ASRCreciprocm = value->iValue;
        here->ASRCreciprocmGiven = TRUE;
        break;
    case ASRC_TEMP:
        here->ASRCtemp = value->rValue + CONSTCtoK;
        here->ASRCtempGiven = TRUE;
        break;
    case ASRC_DTEMP:
        here->ASRCdtemp = value->rValue;
        here->ASRCdtempGiven = TRUE;
        break;
    default:
        return(E_BADPARM);
    }

    return(OK);
}
