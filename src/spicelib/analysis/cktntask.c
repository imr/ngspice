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
/*
int
CKTnewTask(void *ckt, void **taskPtr, IFuid taskName)
 */
int
CKTnewTask(void *ckt, void **taskPtr, IFuid taskName, void **defPtr)
/*CDHW See notes in spiceif.c for an explanation of these fixes CDHW*/
{
    TSKtask *tsk, *def=NULL;

    *taskPtr = (void *)tmalloc(sizeof(TSKtask));
    if(*taskPtr==NULL) return(E_NOMEM);
    tsk = *(TSKtask **)taskPtr;
    tsk->TSKname = taskName;
#if (1) /*CDHW*/
    if(defPtr) 
       def = *(TSKtask **)defPtr;
    if ((strcmp(taskName,"special")==0) && def )  {
    /* create options by copying the circuit's defaults */  
        tsk->TSKtemp = def->TSKtemp;    
	tsk->TSKnomTemp = def->TSKnomTemp;
	tsk->TSKmaxOrder = def->TSKmaxOrder;
	tsk->TSKintegrateMethod = def->TSKintegrateMethod;
	tsk->TSKbypass = def->TSKbypass;
	tsk->TSKdcMaxIter = def->TSKdcMaxIter;
	tsk->TSKdcTrcvMaxIter = def->TSKdcTrcvMaxIter;
	tsk->TSKtranMaxIter = def->TSKtranMaxIter;
	tsk->TSKnumSrcSteps = def->TSKnumSrcSteps;
	tsk->TSKnumGminSteps = def->TSKnumGminSteps;
	tsk->TSKgminFactor   = def->TSKgminFactor;
	/* minBreak */
	tsk->TSKabstol = def->TSKabstol;
	tsk->TSKpivotAbsTol = def->TSKpivotAbsTol;
        tsk->TSKpivotRelTol = def->TSKpivotRelTol;
	tsk->TSKreltol = def->TSKreltol;
        tsk->TSKchgtol = def->TSKchgtol;
	tsk->TSKvoltTol = def->TSKvoltTol;
        tsk->TSKgmin = def->TSKgmin;
	tsk->TSKgshunt = def->TSKgshunt;
        /* delmin */
        tsk->TSKtrtol = def->TSKtrtol;
	tsk->TSKdefaultMosM = def->TSKdefaultMosM;
        tsk->TSKdefaultMosL = def->TSKdefaultMosL;
        tsk->TSKdefaultMosW = def->TSKdefaultMosW;
        tsk->TSKdefaultMosAD = def->TSKdefaultMosAD;
        tsk->TSKdefaultMosAS = def->TSKdefaultMosAS;
	/* fixLimit */
	tsk->TSKnoOpIter= def->TSKnoOpIter;
	tsk->TSKtryToCompact = def->TSKtryToCompact;
	tsk->TSKbadMos3 = def->TSKbadMos3;
        tsk->TSKkeepOpInfo = def->TSKkeepOpInfo;
	tsk->TSKcopyNodesets = def->TSKcopyNodesets;
	tsk->TSKnodeDamping = def->TSKnodeDamping;
	tsk->TSKabsDv = def->TSKabsDv;
        tsk->TSKrelDv = def->TSKrelDv;
#ifdef NEWTRUNC
        tsk->TSKlteReltol = def->TSKlteReltol;
        tsk->TSKlteAbstol = def->TSKlteAbstol;
#endif /* NEWTRUNC */       
        } else {
     /* use the application defaults */
#endif /*CDHW*/	
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

/* gtri - modify - 4/17/91 - wbk - Change trtol default */
#ifdef XSPICE
/* Lower default value of trtol to give more accuracy */
/*    tsk->TSKtrtol = 7;  */
    tsk->TSKtrtol = 1;
/* gtri - modify - 4/17/91 - wbk - Change trtol default */
#else
        tsk->TSKtrtol = 7;
#endif /* XSPICE */

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
#if (1) /*CDHW*/
        }
#endif	     
    return(OK);
}
