/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include <stdio.h>
#include "sperror.h"
#include "trandefs.h"


extern SPICEanalysis *analInfo[];
extern int ANALmaxnum;

int
CKTdoJob(void *inCkt, int reset, void *inTask)
{
    CKTcircuit	*ckt = (CKTcircuit *)inCkt;
    TSKtask	*task = (TSKtask *)inTask;
    JOB		*job;
    double	startTime;
    int		error, i, error2;

#ifdef WANT_SENSE2
    int		senflag;
    static int	sens_num = -1;

    /* Sensitivity is special */
    if (sens_num < 0) {
	for (i = 0; i <  ANALmaxnum; i++)
	    if (!strcmp("SENS2", analInfo[i]->public.name))
		break;
	sens_num = i;
    }
#endif

    startTime = (*(SPfrontEnd->IFseconds))( );

    ckt->CKTtemp  = task->TSKtemp;
    ckt->CKTnomTemp  = task->TSKnomTemp;
    ckt->CKTmaxOrder  = task->TSKmaxOrder;
    ckt->CKTintegrateMethod  = task->TSKintegrateMethod;
    ckt->CKTbypass  = task->TSKbypass;
    ckt->CKTdcMaxIter  = task->TSKdcMaxIter;
    ckt->CKTdcTrcvMaxIter  = task->TSKdcTrcvMaxIter;
    ckt->CKTtranMaxIter  = task->TSKtranMaxIter;
    ckt->CKTnumSrcSteps  = task->TSKnumSrcSteps;
    ckt->CKTnumGminSteps  = task->TSKnumGminSteps;
    ckt->CKTgminFactor  = task->TSKgminFactor;
    ckt->CKTminBreak  = task->TSKminBreak;
    ckt->CKTabstol  = task->TSKabstol;
    ckt->CKTpivotAbsTol  = task->TSKpivotAbsTol;
    ckt->CKTpivotRelTol  = task->TSKpivotRelTol;
    ckt->CKTreltol  = task->TSKreltol;
    ckt->CKTchgtol  = task->TSKchgtol;
    ckt->CKTvoltTol  = task->TSKvoltTol;
    ckt->CKTgmin  = task->TSKgmin;
    ckt->CKTgshunt  = task->TSKgshunt;
    ckt->CKTdelmin  = task->TSKdelmin;
    ckt->CKTtrtol  = task->TSKtrtol;
    ckt->CKTdefaultMosM  = task->TSKdefaultMosM;
    ckt->CKTdefaultMosL  = task->TSKdefaultMosL;
    ckt->CKTdefaultMosW  = task->TSKdefaultMosW;
    ckt->CKTdefaultMosAD  = task->TSKdefaultMosAD;
    ckt->CKTdefaultMosAS  = task->TSKdefaultMosAS;
    ckt->CKTfixLimit  = task->TSKfixLimit;
    ckt->CKTnoOpIter  = task->TSKnoOpIter;
    ckt->CKTtryToCompact = task->TSKtryToCompact;
    ckt->CKTbadMos3 = task->TSKbadMos3;
    ckt->CKTkeepOpInfo = task->TSKkeepOpInfo;
    ckt->CKTcopyNodesets = task->TSKcopyNodesets;
    ckt->CKTnodeDamping = task->TSKnodeDamping;
    ckt->CKTabsDv = task->TSKabsDv;
    ckt->CKTrelDv = task->TSKrelDv;
    ckt->CKTtroubleNode  = 0;
    ckt->CKTtroubleElt  = NULL;
#ifdef NEWTRUNC
    ckt->CKTlteReltol = task->TSKlteReltol;
    ckt->CKTlteAbstol = task->TSKlteAbstol;
#endif /* NEWTRUNC */

    error = 0;

    if (reset) {

	ckt->CKTdelta = 0.0;
	ckt->CKTtime = 0.0;
	ckt->CKTcurrentAnalysis = 0;

#ifdef WANT_SENSE2
	senflag = 0;
	if (sens_num < ANALmaxnum)
	    for (job = task->jobs; !error && job; job = job->JOBnextJob) {
		if (job->JOBtype == sens_num) {
		    senflag = 1;
		    ckt->CKTcurJob = job;
		    ckt->CKTsenInfo = (SENstruct *) job;
		    error = (*(analInfo[sens_num]->an_func))(ckt, reset);
		}
	    }

	if (ckt->CKTsenInfo && (!senflag || error))
	    FREE(ckt->CKTsenInfo);
#endif

	/* normal reset */
	if (!error)
	    error = CKTunsetup(ckt);
	if (!error)
	    error = CKTsetup(ckt);
	if (!error)
	    error = CKTtemp(ckt);
	if (error)
	    return error;
    }

    error2 = OK;

    /* Analysis order is important */
    for (i = 0; i < ANALmaxnum; i++) {

#ifdef WANT_SENSE2
	if (i == sens_num)
	    continue;
#endif

	for (job = task->jobs; job; job = job->JOBnextJob) {
	    if (job->JOBtype == i) {
                ckt->CKTcurJob=job;
		error = OK;
		if (analInfo[i]->an_init)
		    error = (*(analInfo[i]->an_init))(ckt, job);
		if (!error && analInfo[i]->do_ic)
		    error = CKTic(ckt);
		if (!error)
		    error = (*(analInfo[i]->an_func))(ckt, reset);
		if (error)
		    error2 = error;
	    }
	}
    }

    ckt->CKTstat->STATtotAnalTime += (*(SPfrontEnd->IFseconds))( ) - startTime;

#ifdef WANT_SENSE2
    if (ckt->CKTsenInfo)
	SENdestroy(ckt->CKTsenInfo);
#endif

    return(error2);
}

