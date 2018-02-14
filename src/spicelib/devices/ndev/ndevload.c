/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "ndevdefs.h"
#include "ngspice/numenum.h"
#include "ngspice/trandefs.h"
#include "ngspice/complex.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
NDEVload(GENmodel * inModel, CKTcircuit * ckt)
{
  NDEVmodel *model = (NDEVmodel *)inModel;
  NDEVinstance *here;
  int i,j;
    
    /*  loop through all the ndev models */
    for( ; model != NULL; model = NDEVnextModel(model)) 
    {

        /* loop through all the instances of the model */
        for (here = NDEVinstances(model); here != NULL ; here=NDEVnextInstance(here)) 
	{
	     /* sent ckt information to device simulator */
	     here->CKTInfo.DEV_CALL = NDEV_LOAD; 
             here->CKTInfo.CKTmode  = ckt->CKTmode;
             here->CKTInfo.time     = ckt->CKTtime;
             here->CKTInfo.dt       = ckt->CKTdelta;
             here->CKTInfo.dt_old   = ckt->CKTdeltaOld[0];
	     here->CKTInfo.accept_flag = 0;
             send(model->sock,&here->CKTInfo,sizeof(sCKTinfo),0);
    
	     /* send terminal voltage to device simulator */
             for(i=0;i<here->term;i++)
	     {
	       here->PINinfos[i].V_old = here->PINinfos[i].V;
	       here->PINinfos[i].V = *(ckt->CKTrhsOld+here->pin[i]);
	       send(model->sock,&here->PINinfos[i],sizeof(here->PINinfos[i]),0);
	     }
        }
    } 

    /*  loop through all the ndev models */
    
    for(model = (NDEVmodel *)inModel; model != NULL; model = NDEVnextModel(model)) 
    {
	
        /* loop through all the instances of the model */
        for (here = NDEVinstances(model); here != NULL ; here=NDEVnextInstance(here)) 
	{
             /* reveive terminal current and conductional matrix from device simulator */
	     for(i=0;i<here->term;i++)  
	     {
	       recv(model->sock,&here->PINinfos[i],sizeof(here->PINinfos[i]),MSG_WAITALL);
	       *(ckt->CKTrhs+here->pin[i]) += here->PINinfos[i].I;
	       for(j=0;j<here->term;j++) 
	         *(here->mat_pointer[i*here->term+j]) += here->PINinfos[i].dI_dV[j];
	     }
        }
    }

    return(OK);
}

int NDEVgetic(GENmodel *inModel, CKTcircuit *ckt)
{
    NDEVmodel *model = (NDEVmodel*)inModel;
    NDEVinstance *here;

    NG_IGNORE(ckt);

    /*
     * grab initial conditions out of rhs array.   User specified, so use
     * external nodes to get values
     */

    printf("set ic\n");
     
    for( ; model ; model = NDEVnextModel(model)) {
        for(here = NDEVinstances(model); here ; here = NDEVnextInstance(here)) {
/*
            if(!here->DIOinitCondGiven) {
                here->DIOinitCond = 
                        *(ckt->CKTrhs + here->DIOposNode) - 
                        *(ckt->CKTrhs + here->DIOnegNode);

            }
*/				    
        }
    }
    return(OK);
}
