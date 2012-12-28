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
    PSSan *job = (PSSan *) anal;

    NG_IGNORE(ckt);

    switch(which) {

    case GUESSED_FREQ:
        value->rValue = job->PSSguessedFreq;
        break;
    case OSC_NODE:
        value->nValue = job->PSSoscNode;
        break;
    case STAB_TIME:
        value->rValue = job->PSSstabTime;
        break;
    case PSS_UIC:
        if (job->PSSmode & MODEUIC) {
            value->iValue = 1;
        } else {
            value->iValue = 0;
        }
        break;
    case PSS_POINTS:
        value->iValue = (int)job->PSSpoints;
        break;
    case PSS_HARMS:
        value->iValue = job->PSSharms;
        break;
    case SC_ITER:
        value->iValue = job->sc_iter;
        break;
    case STEADY_COEFF:
        value->rValue = job->steady_coeff;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
