/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/

#include "ngspice.h"

#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "res/resdefs.h"

#include "cktdefs.h"
#include "const.h"
#include "sperror.h"

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add headers */
#include "mif.h"
#include "evtproto.h"
#include "ipctiein.h"
/* gtri - end - wbk */
#endif

#include <devdefs.h>
extern SPICEdev **DEVices;

int
DCtrCurv(CKTcircuit *ckt, int restart) 
                
                /* forced restart flag */
{
    TRCV* cv = (TRCV*)ckt->CKTcurJob; /* Where we get the job to do */ 
    int i;
    double *temp;
    int converged;
    int rcode;     
    int vcode;
    int icode;
    int j;
    int error;
    IFuid varUid;
    IFuid *nameList;
    int numNames;
    int firstTime=1;
    static void *plot=NULL;

#ifdef WANT_SENSE2
#ifdef SENSDEBUG
    if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&DCSEN) ){
        printf("\nDC Sensitivity Results\n\n");
        CKTsenPrint(ckt);
    }
#endif /* SENSDEBUG */
#endif


    rcode = CKTtypelook("Resistor");
    vcode = CKTtypelook("Vsource");
    icode = CKTtypelook("Isource");
    if(!restart && cv->TRCVnestState >= 0) {
        /* continuing */
        i = cv->TRCVnestState;
	/* resume to work? saj*/
	error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
		      (void*)ckt->CKTcurJob, ckt->CKTcurJob->JOBname,
	              varUid,IF_REAL,666,nameList, 666,&plot);	
        goto resume;
    }
    ckt->CKTtime = 0;
    ckt->CKTdelta = cv->TRCVvStep[0];
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT ;
    ckt->CKTorder=1;

    
    /* Save the state of the circuit */
    for(i=0;i<7;i++) {
        ckt->CKTdeltaOld[i]=ckt->CKTdelta;
    }
    
    for(i=0;i<=cv->TRCVnestLevel;i++) {
        if(rcode >= 0) {
            /* resistances are in this version, so use them */
            RESinstance *here;
            RESmodel *model;

            for(model = (RESmodel *)ckt->CKThead[rcode];model != NULL;
                    model=model->RESnextModel){
                for(here=model->RESinstances;here!=NULL;
                        here=here->RESnextInstance) {
                    if(here->RESname == cv->TRCVvName[i]) {
                        cv->TRCVvElt[i]   = (GENinstance *)here;
                        cv->TRCVvSave[i]  = here->RESresist;
			cv->TRCVgSave[i]  = here->RESresGiven;
                        cv->TRCVvType[i]  = rcode;
                        here->RESresist   = cv->TRCVvStart[i];
                        here->RESresGiven = 1;
			CKTtemp(ckt);
                        goto found;
                    }
                }
            }
        }
        if(vcode >= 0) {
            /* voltage sources are in this version, so use them */
            VSRCinstance *here;
            VSRCmodel *model;

            for(model = (VSRCmodel *)ckt->CKThead[vcode];model != NULL;
                    model=model->VSRCnextModel){
                for(here=model->VSRCinstances;here!=NULL;
                        here=here->VSRCnextInstance) {
                    if(here->VSRCname == cv->TRCVvName[i]) {
                        cv->TRCVvElt[i]   = (GENinstance *)here;
                        cv->TRCVvSave[i]  = here->VSRCdcValue;
			cv->TRCVgSave[i]  = here->VSRCdcGiven;
                        cv->TRCVvType[i]  = vcode;
                        here->VSRCdcValue = cv->TRCVvStart[i];
                        here->VSRCdcGiven = 1;
                        goto found;
                    }
                }
            }
        }
        if(icode >= 0 ) {
            /* current sources are in this version, so use them */
            ISRCinstance *here;
            ISRCmodel *model;

            for(model= (ISRCmodel *)ckt->CKThead[icode];model != NULL;
                    model=model->ISRCnextModel){
                for(here=model->ISRCinstances;here!=NULL;
                        here=here->ISRCnextInstance) {
                    if(here->ISRCname == cv->TRCVvName[i]) {
                        cv->TRCVvElt[i]   = (GENinstance *)here;
                        cv->TRCVvSave[i]  = here->ISRCdcValue;
			cv->TRCVgSave[i]  = here->ISRCdcGiven;
                        cv->TRCVvType[i]  = icode;
                        here->ISRCdcValue = cv->TRCVvStart[i];
                        here->ISRCdcGiven = 1;
                        goto found;
                    }
                }
            }
        }
	
        if(!strcmp(cv->TRCVvName[i], "temp"))
        {
            cv->TRCVvSave[i]=ckt->CKTtemp; /* Saves the old circuit temperature */
            cv->TRCVvType[i]=TEMP_CODE;    /* Set the sweep type code */
            ckt->CKTtemp = cv->TRCVvStart[i] + CONSTCtoK; /* Set the new circuit temp */
	    CKTtemp(ckt);
            goto found;
        }
	
        (*(SPfrontEnd->IFerror))(ERR_FATAL, 
                "DCtrCurv: source / resistor %s not in circuit", &(cv->TRCVvName[i]));
        return(E_NODEV);

found:;
    }

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and anal_init and anal_type */

        /* Tell the beginPlot routine what mode we're in */
        g_ipc.anal_type = IPC_ANAL_DCTRCURVE;

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_DC;

        g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif

    i--; /* PN: This seems to do nothing ??? */ 
    
    error = CKTnames(ckt,&numNames,&nameList);
    if(error) return(error);
    
    
        if (cv->TRCVvType[i]==vcode)
    	   (*(SPfrontEnd->IFnewUid))((void *)ckt,&varUid,(IFuid )NULL,
            "v-sweep", UID_OTHER, (void **)NULL);
        
	else {
	      if (cv->TRCVvType[i]==icode)
    	         (*(SPfrontEnd->IFnewUid))((void *)ckt,&varUid,(IFuid )NULL,
                 "i-sweep", UID_OTHER, (void **)NULL);
                     
                 else {
		       if (cv->TRCVvType[i]==TEMP_CODE)
    	                  (*(SPfrontEnd->IFnewUid))((void *)ckt,&varUid,(IFuid )NULL,
                          "temp-sweep", UID_OTHER, (void **)NULL);
	    
                           else {
			         if (cv->TRCVvType[i]==rcode)
    	                            (*(SPfrontEnd->IFnewUid))((void *)ckt,&varUid,(IFuid )NULL,
                                    "res-sweep", UID_OTHER, (void **)NULL);
                                
				    else
    	                                  (*(SPfrontEnd->IFnewUid))((void *)ckt,&varUid,(IFuid )NULL,
                                          "?-sweep", UID_OTHER, (void **)NULL);	    
                           } /* icode */
                 } /* TEMP_CODE */
        } /* rcode*/
    
    error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
	(void*)ckt->CKTcurJob, ckt->CKTcurJob->JOBname,
	varUid,IF_REAL,numNames,nameList, IF_REAL,&plot);
    
    if(error) return(error);
    /* now have finished the initialization - can start doing hard part */
    
    i = 0;

resume:
    
    for(;;) {

        if(cv->TRCVvType[i]==vcode) { /* voltage source */
            if((((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcValue)*
                    SIGN(1.,cv->TRCVvStep[i]) - 
                    SIGN(1.,cv->TRCVvStep[i]) * cv->TRCVvStop[i] >
		    DBL_EPSILON*1e+03)
                { 
                    i++ ; 
                    firstTime=1;
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | 
                            MODEDCTRANCURVE | MODEINITJCT ;
                    if (i > cv->TRCVnestLevel ) break ; 
                    goto nextstep;
                }
        } else if(cv->TRCVvType[i]==icode) { /* current source */
            if((((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcValue)*
                    SIGN(1.,cv->TRCVvStep[i]) -
                    SIGN(1.,cv->TRCVvStep[i]) * cv->TRCVvStop[i] >
		    DBL_EPSILON*1e+03)
                { 
                    i++ ; 
                    firstTime=1;
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | 
                            MODEDCTRANCURVE | MODEINITJCT ;
                    if (i > cv->TRCVnestLevel ) break ; 
                    goto nextstep;
                } 
		
	} else if(cv->TRCVvType[i]==rcode) { /* resistance */
            if((((RESinstance*)(cv->TRCVvElt[i]))->RESresist)*
                    SIGN(1.,cv->TRCVvStep[i]) -
                    SIGN(1.,cv->TRCVvStep[i]) * cv->TRCVvStop[i] 
		    > DBL_EPSILON*1e+03)
                { 
                    i++ ; 
                    firstTime=1;
                    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | 
                            MODEDCTRANCURVE | MODEINITJCT ;
                    if (i > cv->TRCVnestLevel ) break ; 
                    goto nextstep;
                } 
        } else if(cv->TRCVvType[i]==TEMP_CODE) { /* temp sweep */
            if(((ckt->CKTtemp) - CONSTCtoK) * SIGN(1.,cv->TRCVvStep[i]) -
	            SIGN(1.,cv->TRCVvStep[i]) * cv->TRCVvStop[i] >
		    DBL_EPSILON*1e+03)
		  {
		     i++ ;
		     firstTime=1;
		     ckt->CKTmode = (ckt->CKTmode & MODEUIC) |
		            MODEDCTRANCURVE | MODEINITJCT ;
		     if (i > cv->TRCVnestLevel ) break ;
		     goto nextstep;
		  
		  }
	   
        } /* else  not possible */
        while (i > 0) { 
            /* init(i); */
            i--; 
            if(cv->TRCVvType[i]==vcode) { /* voltage source */
                ((VSRCinstance *)(cv->TRCVvElt[i]))->VSRCdcValue =
                        cv->TRCVvStart[i];
			
            } else if(cv->TRCVvType[i]==icode) { /* current source */
                ((ISRCinstance *)(cv->TRCVvElt[i]))->ISRCdcValue =
                        cv->TRCVvStart[i];
			
            } else if(cv->TRCVvType[i]==TEMP_CODE) { 
                ckt->CKTtemp = cv->TRCVvStart[i] + CONSTCtoK;
                CKTtemp(ckt); 
	    
	    } else if(cv->TRCVvType[i]==rcode) { 
                ((RESinstance *)(cv->TRCVvElt[i]))->RESresist =
                        cv->TRCVvStart[i];
		((RESinstance *)(cv->TRCVvElt[i]))->RESconduct =	
		  1/(((RESinstance *)(cv->TRCVvElt[i]))->RESresist); 
                                                     /* Note: changing the resistance does nothing */
		                                     /* changing the conductance 1/r instead */
		DEVices[rcode]->DEVload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt); 		
		
		/*
		 * RESload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt); 
		 */
	     
	     
	     } /* else not possible */
        }

        /* Rotate state vectors. */
        temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
        for(j=ckt->CKTmaxOrder;j>=0;j--) {
            ckt->CKTstates[j+1] = ckt->CKTstates[j];
        }
        ckt->CKTstate0 = temp;

        /* do operation */
#ifdef XSPICE
/* gtri - begin - wbk - Do EVTop if event instances exist */
    if(ckt->evt->counts.num_insts == 0) {
        /* If no event-driven instances, do what SPICE normally does */
#endif
        converged = NIiter(ckt,ckt->CKTdcTrcvMaxIter);
        if(converged != 0) {
            converged = CKTop(ckt,
                (ckt->CKTmode&MODEUIC)|MODEDCTRANCURVE | MODEINITJCT,
                (ckt->CKTmode&MODEUIC)|MODEDCTRANCURVE | MODEINITFLOAT,
                ckt->CKTdcMaxIter);
            if(converged != 0) {
                return(converged);
            }
        }
#ifdef XSPICE
    }
    else {
        /* else do new algorithm */

        /* first get the current step in the analysis */
        if(cv->TRCVvType[0] == vcode) {
            g_mif_info.circuit.evt_step = ((VSRCinstance *)(cv->TRCVvElt[i]))
                    ->VSRCdcValue ;
        } else if(cv->TRCVvType[0] == icode) {
            g_mif_info.circuit.evt_step = ((ISRCinstance *)(cv->TRCVvElt[i]))
                    ->ISRCdcValue ;
        } else if(cv->TRCVvType[0] == rcode) {
            g_mif_info.circuit.evt_step =  ((RESinstance*)(cv->TRCVvElt[i]->GENmodPtr))
                    ->RESresist;
        } else if(cv->TRCVvType[0] == TEMP_CODE) {
            g_mif_info.circuit.evt_step =  ckt->CKTtemp - CONSTCtoK;
        }

        /* if first time through, call EVTop immediately and save event results */
        if(firstTime) {
            converged = EVTop(ckt,
                        (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                        (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                        ckt->CKTdcMaxIter,
                        MIF_TRUE);
            EVTdump(ckt, IPC_ANAL_DCOP, g_mif_info.circuit.evt_step);
            EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
            if(converged != 0)
                return(converged);
        }
        /* else, call NIiter first with mode = MODEINITPRED */
        /* to attempt quick analog solution.  Then call all hybrids and call */
        /* EVTop only if event outputs have changed, or if non-converged */
        else {
            converged = NIiter(ckt,ckt->CKTdcTrcvMaxIter);
            EVTcall_hybrids(ckt);
            if((converged != 0) || (ckt->evt->queue.output.num_changed != 0)) {
                converged = EVTop(ckt,
                            (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                            (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                            ckt->CKTdcMaxIter,
                            MIF_FALSE);
                EVTdump(ckt, IPC_ANAL_DCTRCURVE, g_mif_info.circuit.evt_step);
                EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
                if(converged != 0)
                    return(converged);
            }
        }
    }
/* gtri - end - wbk - Do EVTop if event instances exist */
#endif

        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODEDCTRANCURVE | MODEINITPRED ;
        if(cv->TRCVvType[0] == vcode) {
            ckt->CKTtime = ((VSRCinstance *)(cv->TRCVvElt[i]))
                    ->VSRCdcValue ;
        } else if(cv->TRCVvType[0] == icode) {
            ckt->CKTtime = ((ISRCinstance *)(cv->TRCVvElt[i]))
                    ->ISRCdcValue ;
        } else if(cv->TRCVvType[0] == rcode) {
            ckt->CKTtime = ((RESinstance *)(cv->TRCVvElt[i]))
                    ->RESresist;
        } 
        /* PN Temp sweep */
	else 
        {
	ckt->CKTtime = ckt->CKTtemp - CONSTCtoK ; 
        }

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

        /* If first time through, call CKTdump to output Operating Point info */
        /* for Mspice compatibility */

        if(g_ipc.enabled && firstTime) {
            ipc_send_dcop_prefix();
            CKTdump(ckt,(double) 0,plot);
            ipc_send_dcop_suffix();
        }

/* gtri - end - wbk */
#endif

#ifdef WANT_SENSE2
/*
        if(!ckt->CKTsenInfo) printf("sensitivity structure does not exist\n");
    */
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode&DCSEN) ){
	    int senmode;

#ifdef SENSDEBUG
            if(cv->TRCVvType[i]==vcode) { /* voltage source */
                printf("Voltage Source Value : %.5e V\n",
                        ((VSRCinstance*) (cv->TRCVvElt[i]))->VSRCdcValue);
            }
            if(cv->TRCVvType[i]==icode) { /* current source */
                printf("Current Source Value : %.5e A\n",
                        ((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcValue);
            }
	    if(cv->TRCVvType[i]==rcode) { /* resistance */
                printf("Current Resistance Value : %.5e Ohm\n",
                        ((RESinstance*)(cv->TRCVvElt[i]->GENmodPtr))->RESresist);
            }
	    if(cv->TRCVvType[i]==TEMP_CODE) { /* Temperature */
                printf("Current Circuit Temperature : %.5e C\n",
                        ckt-CKTtemp - CONSTCtoK);
            }
	    
#endif /* SENSDEBUG */

            senmode = ckt->CKTsenInfo->SENmode;
            save = ckt->CKTmode;
            ckt->CKTsenInfo->SENmode = DCSEN;
            if(error = CKTsenDCtran(ckt)) return (error);
            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = senmode;

        }
#endif

#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC delimiters */

        if(g_ipc.enabled)
            ipc_send_data_prefix(ckt->CKTtime);
#endif

        CKTdump(ckt,ckt->CKTtime,plot);

#ifdef XSPICE
        if(g_ipc.enabled)
            ipc_send_data_suffix();

/* gtri - end - wbk */
#endif

        if(firstTime) {
            firstTime=0;
            bcopy((char *)ckt->CKTstate0,(char *)ckt->CKTstate1,
                    ckt->CKTnumStates*sizeof(double));
        }

nextstep:;
        if(cv->TRCVvType[i]==vcode) { /* voltage source */
            ((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcValue +=
                    cv->TRCVvStep[i];
        } else if(cv->TRCVvType[i]==icode) { /* current source */
            ((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcValue +=
                    cv->TRCVvStep[i];
	} else if(cv->TRCVvType[i]==rcode) { /* resistance */
            ((RESinstance*)(cv->TRCVvElt[i]))->RESresist +=
                    cv->TRCVvStep[i];
	    /* This code should update resistance and conductance */    
	    ((RESinstance*)(cv->TRCVvElt[i]))->RESconduct =
	    1/(((RESinstance*)(cv->TRCVvElt[i]))->RESresist);
            DEVices[rcode]->DEVload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt); 	    
            /*
	     * RESload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt);
	     */ 
	}
	/* PN Temp Sweep - serban */
        else if (cv->TRCVvType[i]==TEMP_CODE)
        {
    	    ckt->CKTtemp += cv->TRCVvStep[i];
            CKTtemp(ckt);	    
        } /* else not possible */
        
	if( (*(SPfrontEnd->IFpauseTest))() ) {
            /* user asked us to pause, so save state */
            cv->TRCVnestState = i;
            return(E_PAUSE);
        }
    }

    /* all done, lets put everything back */

    for(i=0;i<=cv->TRCVnestLevel;i++) {
        if(cv->TRCVvType[i] == vcode) {   /* voltage source */
            ((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcValue = 
                    cv->TRCVvSave[i];
            ((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcGiven = cv->TRCVgSave[i];
        } else  if(cv->TRCVvType[i] == icode) /*current source */ {
            ((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcValue = 
                    cv->TRCVvSave[i];
            ((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcGiven = cv->TRCVgSave[i];
        } else  if(cv->TRCVvType[i] == rcode) /* Resistance */ {
            ((RESinstance*)(cv->TRCVvElt[i]))->RESresist = 
                    cv->TRCVvSave[i];
	    /* We restore both resistance and conductance */
	    ((RESinstance*)(cv->TRCVvElt[i]))->RESconduct =
	    1/(((RESinstance*)(cv->TRCVvElt[i]))->RESresist);
	    
            ((RESinstance*)(cv->TRCVvElt[i]))->RESresGiven = cv->TRCVgSave[i];
	    DEVices[rcode]->DEVload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt); 
	    
	    /*
	     * RESload((GENmodel*)(cv->TRCVvElt[i]->GENmodPtr),ckt);
	     */ 
        }
	 else if(cv->TRCVvType[i] == TEMP_CODE) {
            ckt->CKTtemp = cv->TRCVvSave[i];
	    CKTtemp(ckt);
	} /* else not possible */
    }
    (*(SPfrontEnd->OUTendPlot))(plot);

    return(OK);
}
