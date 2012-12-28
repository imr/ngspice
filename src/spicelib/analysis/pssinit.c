/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/pssdefs.h"
#include "ngspice/iferrmsg.h"

int PSSinit(CKTcircuit *ckt, JOB *anal)
{
    PSSan *job = (PSSan *) anal;

    /* Step is chosen empirically to be 1% of PSSguessedFreq */
    ckt->CKTstep = 0.01 * (1/job->PSSguessedFreq);
    /* Init time should be always zero */
    ckt->CKTinitTime = 0;
    /* MaxStep should not exceed Nyquist criterion */
    ckt->CKTmaxStep = 0.5*(1/job->PSSguessedFreq);
    ckt->CKTdelmin = 1e-9*ckt->CKTmaxStep;
    ckt->CKTmode = job->PSSmode;
    /* modified CKTdefs.h  for the following - 100609 */
    ckt->CKTstabTime     = job->PSSstabTime;
    ckt->CKTguessedFreq  = job->PSSguessedFreq;
    ckt->CKTharms        = job->PSSharms;
    ckt->CKTpsspoints    = job->PSSpoints;
    ckt->CKTsc_iter      = job->sc_iter;
    ckt->CKTsteady_coeff = job->steady_coeff;

    return OK;
}
