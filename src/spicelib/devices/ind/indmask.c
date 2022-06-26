/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


 /* ARGSUSED */
int
INDmAsk(CKTcircuit* ckt, GENmodel* inst, int which, IFvalue* value)
{
    INDmodel* here = (INDmodel*)inst;

    NG_IGNORE(ckt);

    switch (which) {
    case IND_MOD_IND:
        value->rValue = here->INDmInd;
        return(OK);
    case IND_MOD_TNOM:
        value->rValue = here->INDtnom - CONSTCtoK;
        return(OK);
    case IND_MOD_TC1:
        value->rValue = here->INDtempCoeff1;
        return(OK);
    case IND_MOD_TC2:
        value->rValue = here->INDtempCoeff2;
        return(OK);
    case IND_MOD_CSECT:
        value->rValue = here->INDcsect;
        return(OK);
    case IND_MOD_DIA:
        value->rValue = here->INDdia;
        return(OK);
    case IND_MOD_LENGTH:
        value->rValue = here->INDlength;
        return(OK);
    case IND_MOD_NT:
        value->rValue = here->INDmodNt;
        return(OK);
    case IND_MOD_MU:
        value->rValue = here->INDmu;
        return(OK);
    default:
        return(E_BADPARM);
    }
}
