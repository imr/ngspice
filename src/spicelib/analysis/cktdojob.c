/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/trandefs.h"
#include "ngspice/cpextern.h"
#include "ngspice/fteext.h"

#include "analysis.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/26/90 - add include for MIF and EVT global data */
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
/* gtri - end - wbk - 11/26/90 */
/* gtri - add - 12/12/90 - wbk - include ipc stuff */
#include "ngspice/ipctiein.h"
/* gtri - end - 12/12/90 */
#endif

extern SPICEanalysis* analInfo[];

int
CKTdoJob(CKTcircuit* ckt, int reset, TSKtask* task)
{
    JOB* job;
    double	startTime;
    int		error, i, error2;

    int         ANALmaxnum = spice_num_analysis();

#ifdef WANT_SENSE2
    int		senflag;
    static int	sens_num = -1;

    /* Sensitivity is special */
    if (sens_num < 0) {
        for (i = 0; i < ANALmaxnum; i++)
            if (!strcmp("SENS2", analInfo[i]->if_analysis.name))
                break;
        sens_num = i;
    }
#endif

    startTime = SPfrontEnd->IFseconds();

    ckt->CKTtemp = task->TSKtemp;
    ckt->CKTnomTemp = task->TSKnomTemp;
    ckt->CKTmaxOrder = task->TSKmaxOrder;
    ckt->CKTintegrateMethod = task->TSKintegrateMethod;
    ckt->CKTindverbosity = task->TSKindverbosity;
    ckt->CKTxmu = task->TSKxmu;
    ckt->CKTbypass = task->TSKbypass;
    ckt->CKTdcMaxIter = task->TSKdcMaxIter;
    ckt->CKTdcTrcvMaxIter = task->TSKdcTrcvMaxIter;
    ckt->CKTtranMaxIter = task->TSKtranMaxIter;
    ckt->CKTnumSrcSteps = task->TSKnumSrcSteps;
    ckt->CKTnumGminSteps = task->TSKnumGminSteps;
    ckt->CKTgminFactor = task->TSKgminFactor;
    ckt->CKTminBreak = task->TSKminBreak;
    ckt->CKTabstol = task->TSKabstol;
    ckt->CKTpivotAbsTol = task->TSKpivotAbsTol;
    ckt->CKTpivotRelTol = task->TSKpivotRelTol;
    ckt->CKTreltol = task->TSKreltol;
    ckt->CKTchgtol = task->TSKchgtol;
    ckt->CKTvoltTol = task->TSKvoltTol;
    ckt->CKTgmin = task->TSKgmin;
    ckt->CKTgshunt = task->TSKgshunt;
    ckt->CKTcshunt = task->TSKcshunt;
    ckt->CKTdelmin = task->TSKdelmin;
    ckt->CKTtrtol = task->TSKtrtol;
#ifdef XSPICE
    /* Lower value of trtol to give smaller stepsize and more accuracy,
       but only if there are 'A' devices in the circuit,
       may be overridden by 'set xtrtol=newval' */
    if (ckt->CKTadevFlag && (ckt->CKTtrtol > 1)) {
        int newtol;
        if (cp_getvar("xtrtol", CP_NUM, &newtol, 0)) {
            printf("Override trtol to %d for xspice 'A' devices\n", newtol);
            ckt->CKTtrtol = newtol;
        }
        else {
            printf("Reducing trtol to 1 for xspice 'A' devices\n");
            ckt->CKTtrtol = 1;
        }
    }
#endif
    ckt->CKTdefaultMosM = task->TSKdefaultMosM;
    ckt->CKTdefaultMosL = task->TSKdefaultMosL;
    ckt->CKTdefaultMosW = task->TSKdefaultMosW;
    ckt->CKTdefaultMosAD = task->TSKdefaultMosAD;
    ckt->CKTdefaultMosAS = task->TSKdefaultMosAS;
    ckt->CKTfixLimit = task->TSKfixLimit;
    ckt->CKTnoOpIter = task->TSKnoOpIter;
    ckt->CKTtryToCompact = task->TSKtryToCompact;
    ckt->CKTbadMos3 = task->TSKbadMos3;
    ckt->CKTkeepOpInfo = task->TSKkeepOpInfo;
    ckt->CKTcopyNodesets = task->TSKcopyNodesets;
    ckt->CKTnodeDamping = task->TSKnodeDamping;
    ckt->CKTabsDv = task->TSKabsDv;
    ckt->CKTrelDv = task->TSKrelDv;
    ckt->CKTtroubleNode = 0;
    ckt->CKTtroubleElt = NULL;
    ckt->CKTnoopac = task->TSKnoopac && ckt->CKTisLinear;
    ckt->CKTepsmin = task->TSKepsmin;
#ifdef NEWTRUNC
    ckt->CKTlteReltol = task->TSKlteReltol;
    ckt->CKTlteAbstol = task->TSKlteAbstol;
#endif /* NEWTRUNC */

    fprintf(stdout, "Doing analysis at TEMP = %f and TNOM = %f\n\n",
        ckt->CKTtemp - CONSTCtoK, ckt->CKTnomTemp - CONSTCtoK);

    /* call altermod and alter on device and model parameters assembled in
       devtlist and modtlist (if using temper) because we have a new temperature */
    inp_evaluate_temper(ft_curckt);

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
                    ckt->CKTsenInfo = (SENstruct*)job;
                    error = analInfo[sens_num]->an_func(ckt, reset);
                }
            }

        if (ckt->CKTsenInfo && (!senflag || error))
            FREE(ckt->CKTsenInfo);
#endif

        /* make sure this is either up do date or NULL */
        ckt->CKTcurJob = NULL;

        /* normal reset */
        if (!error)
            error = CKTunsetup(ckt);

#ifdef XSPICE
        /* gtri - add - 12/12/90 - wbk - set ipc syntax error flag */
        if (error)   g_ipc.syntax_error = IPC_TRUE;
        /* gtri - end - 12/12/90 */
#endif

        if (!error)
            error = CKTsetup(ckt);

#ifdef XSPICE
        /* gtri - add - 12/12/90 - wbk - set ipc syntax error flag */
        if (error)   g_ipc.syntax_error = IPC_TRUE;
        /* gtri - end - 12/12/90 */
#endif

        if (!error)
            error = CKTtemp(ckt);

#ifdef XSPICE
        /* gtri - add - 12/12/90 - wbk - set ipc syntax error flag */
        if (error)   g_ipc.syntax_error = IPC_TRUE;
        /* gtri - end - 12/12/90 */
#endif

        if (error) {

#ifdef XSPICE
            /* gtri - add - 12/12/90 - wbk - return if syntax errors from parsing */
            if (g_ipc.enabled) {
                if (g_ipc.syntax_error)
                    ;
                else {
                    /* else, send (GO) errchk status if we got this far */
                    /* Caller is responsible for sending NOGO status if we returned earlier */
                    ipc_send_errchk();
                }
            }
            /* gtri - end - 12/12/90 */
#endif


            return error;


        }/* if error  */
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
                ckt->CKTcurJob = job;
                error = OK;
                if (analInfo[i]->an_init)
                    error = analInfo[i]->an_init(ckt, job);
                if (!error && analInfo[i]->do_ic)
                    error = CKTic(ckt);
                if (!error) {
#ifdef XSPICE
                    if (reset) {
                        /* gtri - begin - 6/10/91 - wbk - Setup event-driven data */
                        error = EVTsetup(ckt);
                        if (error) {
                            ckt->CKTstat->STATtotAnalTime +=
                                SPfrontEnd->IFseconds() - startTime;
                            return(error);
                        }
                        /* gtri - end - 6/10/91 - wbk - Setup event-driven data */
                    }
#endif
                    error = analInfo[i]->an_func(ckt, reset);
                    /* txl, cpl addition */
                    if (error == 1111) break;
                }
                if (error)
                    error2 = error;
            }
        }
    }

    ckt->CKTstat->STATtotAnalTime += SPfrontEnd->IFseconds() - startTime;

#ifdef WANT_SENSE2
    if (ckt->CKTsenInfo)
        SENdestroy(ckt->CKTsenInfo);
#endif

    return(error2);
}

