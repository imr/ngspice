/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/tskdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"


/* CDHW See notes in spiceif.c for an explanation of these fixes CDHW */

int
CKTnewTask(CKTcircuit *ckt, TSKtask **taskPtr, IFuid taskName, TSKtask **defPtr)
{
    TSKtask *tsk, *def = NULL;

    NG_IGNORE(ckt);

    FREE(*taskPtr); /* clear old task upon repeated calls to tran etc.*/
    *taskPtr = TMALLOC(TSKtask, 1);
    if (*taskPtr == NULL)
        return(E_NOMEM);

    tsk = *taskPtr;
    tsk->TSKname = taskName;

#if (1) /*CDHW*/

    if(defPtr)
        def = *defPtr;

    if ((strcmp(taskName,"special") == 0) && def)  {
        /* create options by copying the circuit's defaults */
        tsk->TSKtemp            = def->TSKtemp;
        tsk->TSKnomTemp         = def->TSKnomTemp;
        tsk->TSKmaxOrder        = def->TSKmaxOrder;
        tsk->TSKintegrateMethod = def->TSKintegrateMethod;
        tsk->TSKindverbosity    = def->TSKindverbosity;
        tsk->TSKxmu             = def->TSKxmu;
        tsk->TSKbypass          = def->TSKbypass;
        tsk->TSKdcMaxIter       = def->TSKdcMaxIter;
        tsk->TSKdcTrcvMaxIter   = def->TSKdcTrcvMaxIter;
        tsk->TSKtranMaxIter     = def->TSKtranMaxIter;
        tsk->TSKnumSrcSteps     = def->TSKnumSrcSteps;
        tsk->TSKnumGminSteps    = def->TSKnumGminSteps;
        tsk->TSKgminFactor      = def->TSKgminFactor;
        /* minBreak */
        tsk->TSKabstol          = def->TSKabstol;
        tsk->TSKpivotAbsTol     = def->TSKpivotAbsTol;
        tsk->TSKpivotRelTol     = def->TSKpivotRelTol;
        tsk->TSKreltol          = def->TSKreltol;
        tsk->TSKchgtol          = def->TSKchgtol;
        tsk->TSKvoltTol         = def->TSKvoltTol;
        tsk->TSKgmin            = def->TSKgmin;
        tsk->TSKgshunt          = def->TSKgshunt;
        tsk->TSKcshunt          = def->TSKcshunt;
        /* delmin */
        tsk->TSKtrtol           = def->TSKtrtol;
        tsk->TSKdefaultMosM     = def->TSKdefaultMosM;
        tsk->TSKdefaultMosL     = def->TSKdefaultMosL;
        tsk->TSKdefaultMosW     = def->TSKdefaultMosW;
        tsk->TSKdefaultMosAD    = def->TSKdefaultMosAD;
        tsk->TSKdefaultMosAS    = def->TSKdefaultMosAS;
        /* fixLimit */
        tsk->TSKnoOpIter        = def->TSKnoOpIter;
        tsk->TSKtryToCompact    = def->TSKtryToCompact;
        tsk->TSKbadMos3         = def->TSKbadMos3;
        tsk->TSKkeepOpInfo      = def->TSKkeepOpInfo;
        tsk->TSKcopyNodesets    = def->TSKcopyNodesets;
        tsk->TSKnodeDamping     = def->TSKnodeDamping;
        tsk->TSKabsDv           = def->TSKabsDv;
        tsk->TSKrelDv           = def->TSKrelDv;
        tsk->TSKnoopac          = def->TSKnoopac;
        tsk->TSKepsmin          = def->TSKepsmin;
#ifdef NEWTRUNC
        tsk->TSKlteReltol       = def->TSKlteReltol;
        tsk->TSKlteAbstol       = def->TSKlteAbstol;
#endif

    } else {
#endif /*CDHW*/

        /* use the application defaults */
        tsk->TSKgmin            = 1e-12;
        tsk->TSKgshunt          = 0;
        tsk->TSKcshunt          = -1;
        tsk->TSKabstol          = 1e-12;
        tsk->TSKreltol          = 1e-3;
        tsk->TSKchgtol          = 1e-14;
        tsk->TSKvoltTol         = 1e-6;
#ifdef NEWTRUNC
        tsk->TSKlteReltol       = 1e-3;
        tsk->TSKlteAbstol       = 1e-6;
#endif
        tsk->TSKtrtol           = 7;
        tsk->TSKbypass          = 0;
        tsk->TSKtranMaxIter     = 10;
        tsk->TSKdcMaxIter       = 100;
        tsk->TSKdcTrcvMaxIter   = 50;
        tsk->TSKintegrateMethod = TRAPEZOIDAL;
        tsk->TSKmaxOrder        = 2;
        /* full check, and full verbosity */
        tsk->TSKindverbosity    = 2;
        /*
         * when using trapezoidal method
         *   xmu=0:    Backward Euler
         *   xmu=0.5:  trapezoidal (standard)
         *   xmu=0.49: good damping of current ringing, e.g. in R.O.s.
         */
        tsk->TSKxmu             = 0.5;
        tsk->TSKnumSrcSteps     = 1;
        tsk->TSKnumGminSteps    = 1;
        tsk->TSKgminFactor      = 10;
        tsk->TSKpivotAbsTol     = 1e-13;
        tsk->TSKpivotRelTol     = 1e-3;
        tsk->TSKtemp            = 300.15;
        tsk->TSKnomTemp         = 300.15;
        tsk->TSKdefaultMosM     = 1;
        tsk->TSKdefaultMosL     = 1e-4;
        tsk->TSKdefaultMosW     = 1e-4;
        tsk->TSKdefaultMosAD    = 0;
        tsk->TSKdefaultMosAS    = 0;
        tsk->TSKnoOpIter        = 0;
        tsk->TSKtryToCompact    = 0;
        tsk->TSKbadMos3         = 0;
        tsk->TSKkeepOpInfo      = 0;
        tsk->TSKcopyNodesets    = 0;
        tsk->TSKnodeDamping     = 0;
        tsk->TSKabsDv           = 0.5;
        tsk->TSKrelDv           = 2.0;
        tsk->TSKepsmin          = 1e-28;

#if (1) /*CDHW*/
    }
#endif

    return(OK);
}
