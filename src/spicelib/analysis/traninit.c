/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "trandefs.h"
#include "iferrmsg.h"

/*
 * this used to be in setup, but we need it here now
 * (must be done after mode is set as below)
 */

int TRANinit(CKTcircuit	*ckt, JOB *job)
{
    ckt->CKTfinalTime = ((TRANan*)job)->TRANfinalTime;
    ckt->CKTstep = ((TRANan*)job)->TRANstep;
    ckt->CKTinitTime = ((TRANan*)job)->TRANinitTime;
    ckt->CKTmaxStep = ((TRANan*)job)->TRANmaxStep;
   
   
    
   /*  The following code has been taken from macspice 3f4 (A. Wilson) 
       in the file traninit.new.c - Seems interesting */
    if(ckt->CKTmaxStep == 0) 
      {
       if (ckt->CKTstep < ( ckt->CKTfinalTime - ckt->CKTinitTime )/50.0)
    	      {
    		  ckt->CKTmaxStep = ckt->CKTstep;
       } 
       else
    	      {
    		ckt->CKTmaxStep = ( ckt->CKTfinalTime - ckt->CKTinitTime )/50.0;
    	 } 
}

   
    
    ckt->CKTdelmin = 1e-11*ckt->CKTmaxStep;	/* XXX */
    ckt->CKTmode = ((TRANan*)job)->TRANmode;

    return OK;
}
