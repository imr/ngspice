/**********
Author: 2010-05 Stefano Perticaroli ``spertica''
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/trandefs.h"
#include "ngspice/pssdefs.h"
#include "ngspice/iferrmsg.h"

int PSSinit(CKTcircuit *ckt, JOB *job)
{
    /* Final time depends on stabilization time requested for PSS
	and on at least one more oscillation period */
    ckt->CKTfinalTime = ((PSSan*)job)->PSSstabTime + 2/((PSSan*)job)->PSSguessedFreq;
    /* Step is chosen empirically to be 1% of PSSguessedFreq */
    ckt->CKTstep = 0.01 * (1/((PSSan*)job)->PSSguessedFreq);
    /* Init time should be always zero */
    ckt->CKTinitTime = 0;
    /* MaxStep should not exceed Nyquist criterion */
    ckt->CKTmaxStep = 0.5*(1/((PSSan*)job)->PSSguessedFreq);
    ckt->CKTdelmin = 1e-9*ckt->CKTmaxStep;
    ckt->CKTmode = ((PSSan*)job)->PSSmode;
    /* modified CKTdefs.h  for the following - 100609 */
    ckt->CKTstabTime = ((PSSan*)job)->PSSstabTime;
    ckt->CKTguessedFreq = ((PSSan*)job)->PSSguessedFreq;
    ckt->CKTharms = ((PSSan*)job)->PSSharms;
    ckt->CKTpsspoints = ((PSSan*)job)->PSSpoints;
    ckt->CKTsc_iter = ((PSSan*)job)->sc_iter;
    ckt->CKTsteady_coeff = ((PSSan*)job)->steady_coeff;

    return OK;
}
