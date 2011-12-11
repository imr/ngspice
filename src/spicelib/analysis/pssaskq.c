/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"
#include "ngspice/pssdefs.h"

/* ARGSUSED */
int
PSSaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value)
{
    NG_IGNORE(ckt);

    switch(which) {

    case GUESSED_FREQ:
        value->rValue = ((PSSan *)anal)->PSSguessedFreq;
        break;
    case OSC_NODE:
        value->nValue = ((PSSan *)anal)->PSSoscNode;
        break;
    case STAB_TIME:
        value->rValue = ((PSSan *)anal)->PSSstabTime;
        break;
    case PSS_UIC:
        if(((PSSan *)anal)->PSSmode & MODEUIC) {
            value->iValue = 1;
        } else {
            value->iValue = 0;
        }
        break;
    case PSS_POINTS:
        value->iValue = ((PSSan *)anal)->PSSpoints;
        break;
    case PSS_HARMS:
        value->iValue = ((PSSan *)anal)->PSSharms;
        break;
    case SC_ITER:
        value->iValue = ((PSSan *)anal)->sc_iter;
        break;
    case STEADY_COEFF:
        value->rValue = ((PSSan *)anal)->steady_coeff;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
