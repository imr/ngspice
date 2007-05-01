/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ndevdefs.h"
#include "complex.h"
#include "sperror.h"
#include "suffix.h"



int NDEVaccept(CKTcircuit *ckt, GENmodel *inModel)
{
  NDEVmodel *model = (NDEVmodel *)inModel;
  NDEVinstance *here;
  /*  loop through all the ndev models */
  for( ; model != NULL; model = model->NDEVnextModel ) 
  {
        /* loop through all the instances of the model */
        for (here = model->NDEVinstances; here != NULL ; here=here->NDEVnextInstance) 
	{ 
	     
	     if (here->NDEVowner != ARCHme) continue;
	     
	     /* set ckt accept_flag */
	     here->CKTInfo.DEV_CALL = NDEV_ACCEPT; 
             here->CKTInfo.CKTmode  = ckt->CKTmode;
             here->CKTInfo.time     = ckt->CKTtime;
             here->CKTInfo.dt       = ckt->CKTdelta;
             here->CKTInfo.dt_old   = ckt->CKTdeltaOld[0];
             here->CKTInfo.accept_flag = 1;
	     send(model->sock,&here->CKTInfo,sizeof(sCKTinfo),0);
        }
  } 
  return (OK);
  /* NOTREACHED */
}


int NDEVconvTest(GENmodel *inModel, CKTcircuit *ckt)
{
    NDEVmodel *model = (NDEVmodel *)inModel;
    NDEVinstance *here;
 
    for( ; model != NULL; model = model->NDEVnextModel) {
        for(here=model->NDEVinstances;here!=NULL;here = here->NDEVnextInstance){
	    if (here->NDEVowner != ARCHme) continue;
            /*
             *   get convergence information from ndev
             */
	   here->CKTInfo.DEV_CALL = NDEV_CONVERGINCE_TEST; 
	   send(model->sock,&here->CKTInfo,sizeof(sCKTinfo),0); 
	   recv(model->sock,&here->CKTInfo,sizeof(sCKTinfo),MSG_WAITALL);
   
            if (here->CKTInfo.convergence_flag<0) {
	        /* no reason to continue - we've failed... */
                ckt->CKTnoncon++;
		ckt->CKTtroubleElt = (GENinstance *) here;
                return(OK); 
            } 
        }
    }
    return(OK);
}
