/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice.h"
#include "tskdefs.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "iferrmsg.h"


/* ARGSUSED */
int
CKTnewTask(void *ckt, void **taskPtr, IFuid taskName)
{
    TSKtask *tsk;

    *taskPtr = (void *)MALLOC(sizeof(TSKtask));
    if(*taskPtr==NULL) return(E_NOMEM);
    tsk = *(TSKtask **)taskPtr;
    tsk->TSKname = taskName;
    tsk->TSKgmin = 1e-12;
    tsk->TSKgshunt = 0;
    tsk->TSKabstol = 1e-12;
    tsk->TSKreltol = 1e-3;
    tsk->TSKchgtol = 1e-14;
    tsk->TSKvoltTol = 1e-6;
#ifdef NEWTRUNC
    tsk->TSKlteReltol = 1e-3;
    tsk->TSKlteAbstol = 1e-6;
#endif /* NEWTRUNC */
    tsk->TSKtrtol = 7;
    tsk->TSKbypass = 0;
    tsk->TSKtranMaxIter = 10;
    tsk->TSKdcMaxIter = 100;
    tsk->TSKdcTrcvMaxIter = 50;
    tsk->TSKintegrateMethod = TRAPEZOIDAL;
    tsk->TSKmaxOrder = 2;
    tsk->TSKnumSrcSteps = 1; 
    tsk->TSKnumGminSteps = 1; 
    tsk->TSKgminFactor = 10;  
    tsk->TSKpivotAbsTol = 1e-13;
    tsk->TSKpivotRelTol = 1e-3;
    tsk->TSKtemp = 300.15;
    tsk->TSKnomTemp = 300.15;
    tsk->TSKdefaultMosM = 1; 
    tsk->TSKdefaultMosL = 1e-4;
    tsk->TSKdefaultMosW = 1e-4;
    tsk->TSKdefaultMosAD = 0;
    tsk->TSKdefaultMosAS = 0;
    tsk->TSKnoOpIter=0;
    tsk->TSKtryToCompact=0;
    tsk->TSKbadMos3=0;
    tsk->TSKkeepOpInfo=0;
    tsk->TSKcopyNodesets=0; 
    tsk->TSKnodeDamping=0;  
    tsk->TSKabsDv=0.5;      
    tsk->TSKrelDv=2.0;     
    return(OK);
}
