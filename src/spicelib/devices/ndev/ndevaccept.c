/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ndevdefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"



int NDEVaccept(CKTcircuit *ckt, GENmodel *inModel)
{
  NDEVmodel *model = (NDEVmodel *)inModel;
  NDEVinstance *here;
  /*  loop through all the ndev models */
  for( ; model != NULL; model = NDEVnextModel(model)) 
  {
        /* loop through all the instances of the model */
        for (here = NDEVinstances(model); here != NULL ; here=NDEVnextInstance(here)) 
	{
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
 
    for( ; model != NULL; model = NDEVnextModel(model)) {
        for(here=NDEVinstances(model);here!=NULL;here = NDEVnextInstance(here)){

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
