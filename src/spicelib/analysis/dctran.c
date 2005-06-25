/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000  AlansFixes
**********/

/* subroutine to do DC TRANSIENT analysis    
        --- ONLY, unlike spice2 routine with the same name! */

#include<ngspice.h>
#include <config.h>
#include <cktdefs.h>
#include <cktaccept.h>
#include <trandefs.h>
#include <sperror.h>

#ifdef XSPICE
/* gtri - add - wbk - Add headers */
#include "miftypes.h" 
/* gtri - end - wbk - Add headers */

/* gtri - add - wbk - Add headers */
#include "evt.h"
#include "mif.h"
#include "evtproto.h"
#include "ipctiein.h"
/* gtri - end - wbk - Add headers */
#endif

#ifdef CLUSTER
#include "cluster.h"
#endif

#ifdef HAS_WINDOWS    /* hvogt 10.03.99, nach W. Mues */
void SetAnalyse( char * Analyse, int Percent);
#endif


int
DCtran(CKTcircuit *ckt,
       int restart)		/* forced restart flag */
{
    int i;
    double olddelta;
    double delta;
    double new;
    double *temp;
    double startdTime;
    double startsTime;
    double startlTime;
    double startcTime;
    double startkTime;
    double startTime;
    int startIters;
    int converged;
    int firsttime;
    int error;
#ifdef WANT_SENSE2
#ifdef SENSDEBUG
    FILE *outsen;
#endif /* SENSDEBUG */
#endif
    int save_order;
    long save_mode;
    IFuid timeUid;
    IFuid *nameList;
    int numNames;
    double maxstepsize=0.0;

    int ltra_num;
    CKTnode *node;
#ifdef PARALLEL_ARCH
    long type = MT_TRANAN, length = 1;
#endif /* PARALLEL_ARCH */
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */
    Ipc_Boolean_t  ipc_firsttime = IPC_TRUE;
    Ipc_Boolean_t  ipc_secondtime = IPC_FALSE;
    Ipc_Boolean_t  ipc_delta_cut = IPC_FALSE;
    double         ipc_last_time = 0.0;
    double         ipc_last_delta = 0.0;
/* gtri - end - wbk - 12/19/90 - Add IPC stuff */
#endif
#ifdef CLUSTER
    int redostep;
#endif
    if(restart || ckt->CKTtime == 0) {
        delta=MIN(ckt->CKTfinalTime/200,ckt->CKTstep)/10;

	/* begin LTRA code addition */
	if (ckt->CKTtimePoints != NULL)
	    FREE(ckt->CKTtimePoints);

	if (ckt->CKTstep >= ckt->CKTmaxStep)
	    maxstepsize = ckt->CKTstep;
	else
	    maxstepsize = ckt->CKTmaxStep;

	ckt->CKTsizeIncr = 10;
	ckt->CKTtimeIndex = -1; /* before the DC soln has been stored */
	ckt->CKTtimeListSize = ckt->CKTfinalTime / maxstepsize + 0.5;
	ltra_num = CKTtypelook("LTRA");
	if (ltra_num >= 0 && ckt->CKThead[ltra_num] != NULL)
	    ckt->CKTtimePoints = NEWN(double, ckt->CKTtimeListSize);
	/* end LTRA code addition */

        if(ckt->CKTbreaks) FREE(ckt->CKTbreaks);
        ckt->CKTbreaks=(double *)MALLOC(2*sizeof(double));
        if(ckt->CKTbreaks == (double *)NULL) return(E_NOMEM);
        *(ckt->CKTbreaks)=0;
        *(ckt->CKTbreaks+1)=ckt->CKTfinalTime;
        ckt->CKTbreakSize=2;

#ifdef XSPICE
/* gtri - begin - wbk - 12/19/90 - Modify setting of CKTminBreak */
/*      if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5; */
        /* Set to 10 times delmin for ATESSE 1 compatibity */
        if(ckt->CKTminBreak==0) ckt->CKTminBreak = 10.0 * ckt->CKTdelmin;
/* gtri - end - wbk - 12/19/90 - Modify setting of CKTminBreak */
#else
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
#endif

#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff and set anal_init and anal_type */

        /* Tell the beginPlot routine what mode we're in */
        g_ipc.anal_type = IPC_ANAL_TRAN;

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_DC;

        g_mif_info.circuit.anal_init = MIF_TRUE;

/* gtri - end - wbk */
#endif
        error = CKTnames(ckt,&numNames,&nameList);
        if(error) return(error);
        (*(SPfrontEnd->IFnewUid))((void *)ckt,&timeUid,(IFuid)NULL,
                "time", UID_OTHER, (void **)NULL);
        error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
		(void*)ckt->CKTcurJob,
                ckt->CKTcurJob->JOBname,timeUid,IF_REAL,numNames,nameList,
                IF_REAL,&(((TRANan*)ckt->CKTcurJob)->TRANplot));
	tfree(nameList);
        if(error) return(error);

        ckt->CKTtime = 0;
        ckt->CKTdelta = 0;
        ckt->CKTbreak=1;
        firsttime = 1;
        save_mode = (ckt->CKTmode&MODEUIC)|MODETRANOP | MODEINITJCT;
        save_order = ckt->CKTorder;
#ifdef XSPICE
/* gtri - begin - wbk - set a breakpoint at end of supply ramping time */
        /* must do this after CKTtime set to 0 above */
        if(ckt->enh->ramp.ramptime > 0.0)
            CKTsetBreak(ckt, ckt->enh->ramp.ramptime);
/* gtri - end - wbk - set a breakpoint at end of supply ramping time */

/* gtri - begin - wbk - Call EVTop if event-driven instances exist */
	if(ckt->evt->counts.num_insts != 0){
	  /* use new DCOP algorithm */
	  converged = EVTop(ckt,
			    (ckt->CKTmode & MODEUIC) | MODETRANOP | MODEINITJCT,
			    (ckt->CKTmode & MODEUIC)|MODETRANOP| MODEINITFLOAT,
			    ckt->CKTdcMaxIter,
			    MIF_TRUE);
	  EVTdump(ckt, IPC_ANAL_DCOP, 0.0);
	  
	  EVTop_save(ckt, MIF_FALSE, 0.0);
	  
	  /* gtri - end - wbk - Call EVTop if event-driven instances exist */
	} else
#endif		
        converged = CKTop(ckt,
                (ckt->CKTmode & MODEUIC)|MODETRANOP| MODEINITJCT,
                (ckt->CKTmode & MODEUIC)|MODETRANOP| MODEINITFLOAT,
                ckt->CKTdcMaxIter);
                
         if(converged != 0) {
			fprintf(stdout,"\nTransient solution failed -\n");
			CKTncDump(ckt);
     /*      CKTnode *node;
           double new, old, tol;
           int i=1;

           fprintf(stdout,"\nTransient solution failed -\n\n");
           fprintf(stdout,"Last Node Voltages\n");
           fprintf(stdout,"------------------\n\n");
           fprintf(stdout,"%-30s %20s %20s\n", "Node", "Last Voltage",
                                                              "Previous Iter");
           fprintf(stdout,"%-30s %20s %20s\n", "----", "------------",
                                                              "-------------");
           for(node=ckt->CKTnodes->next;node;node=node->next) {
             if (strstr(node->name, "#branch") || !strstr(node->name, "#")) {
               new =  *((ckt->CKTrhsOld) + i ) ;
               old =  *((ckt->CKTrhs) + i ) ;
               fprintf(stdout,"%-30s %20g %20g", node->name, new, old);
               if(node->type == 3) {
                   tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                           ckt->CKTvoltTol;
               } else {
                   tol =  ckt->CKTreltol * (MAX(fabs(old),fabs(new))) +
                           ckt->CKTabstol;
               }
               if (fabs(new-old) >tol ) {
                    fprintf(stdout," *");
               }
               fprintf(stdout,"\n");
             };
             i++;
           } */
        } else {
           fprintf(stdout,"\nInitial Transient Solution\n");
           fprintf(stdout,"--------------------------\n\n");
           fprintf(stdout,"%-30s %15s\n", "Node", "Voltage");
           fprintf(stdout,"%-30s %15s\n", "----", "-------");
           for(node=ckt->CKTnodes->next;node;node=node->next) {
               if (strstr(node->name, "#branch") || !strstr(node->name, "#"))
                 fprintf(stdout,"%-30s %15g\n", node->name,
                                             *(ckt->CKTrhsOld+node->number));
           };
        };
        fprintf(stdout,"\n");
        fflush(stdout);
        if(converged != 0) return(converged);
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

        /* Send the operating point results for Mspice compatibility */
        if(g_ipc.enabled) {
            ipc_send_dcop_prefix();
            CKTdump(ckt,(double)0,(((TRANan*)ckt->CKTcurJob)->TRANplot));
            ipc_send_dcop_suffix();
        }

/* gtri - end - wbk */


/* gtri - add - wbk - 12/19/90 - set anal_init and anal_type */

        g_mif_info.circuit.anal_init = MIF_TRUE;

        /* Tell the code models what mode we're in */
        g_mif_info.circuit.anal_type = MIF_TRAN;

/* gtri - end - wbk */

/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Initialize the temporary breakpoint variables to infinity */
        g_mif_info.breakpoint.current = 1.0e30;
        g_mif_info.breakpoint.last    = 1.0e30;

/* gtri - end - wbk - Add Breakpoint stuff */
#endif	
        ckt->CKTstat->STATtimePts ++;
        ckt->CKTorder=1;
        for(i=0;i<7;i++) {
            ckt->CKTdeltaOld[i]=ckt->CKTmaxStep;
        }
        ckt->CKTdelta = delta;
#ifdef STEPDEBUG
        (void)printf("delta initialized to %g\n",ckt->CKTdelta);
#endif
        ckt->CKTsaveDelta = ckt->CKTfinalTime/50;

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
#ifdef SENSDEBUG
            printf("\nTransient Sensitivity Results\n\n");
            CKTsenPrint(ckt);
#endif /* SENSDEBUG */
            save = ckt->CKTsenInfo->SENmode;
            ckt->CKTsenInfo->SENmode = TRANSEN;
            save1 = ckt->CKTmode;
            save2 = ckt->CKTorder;
            ckt->CKTmode = save_mode;
            ckt->CKTorder = save_order;
            if(error = CKTsenDCtran(ckt)) return(error);
            ckt->CKTmode = save1;
            ckt->CKTorder = save2;
        }
#endif

        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
		/* modeinittran set here */
        ckt->CKTag[0]=ckt->CKTag[1]=0;
        bcopy((char *)ckt->CKTstate0,(char *)ckt->CKTstate1,
                ckt->CKTnumStates*sizeof(double));

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
            size = SMPmatSize(ckt->CKTmatrix);
            for(i = 1; i<=size ; i++)
                *(ckt->CKTrhsOp + i) = *(ckt->CKTrhsOld + i);
        }
#endif

        startTime=(*(SPfrontEnd->IFseconds))();
        startIters = ckt->CKTstat->STATnumIter;
        startdTime = ckt->CKTstat->STATdecompTime;
        startsTime = ckt->CKTstat->STATsolveTime;
        startlTime = ckt->CKTstat->STATloadTime;
        startcTime = ckt->CKTstat->STATcombineTime;
        startkTime = ckt->CKTstat->STATsyncTime;
#ifdef CLUSTER
	CLUsetup(ckt);
#endif	
    } else {
	/*saj As traninit resets CKTmode */
    	ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
	/* saj */    
        startTime=(*(SPfrontEnd->IFseconds))();
        startIters = ckt->CKTstat->STATnumIter;
        startdTime = ckt->CKTstat->STATdecompTime;
        startsTime = ckt->CKTstat->STATsolveTime;
        startlTime = ckt->CKTstat->STATloadTime;
        startcTime = ckt->CKTstat->STATcombineTime;
        startkTime = ckt->CKTstat->STATsyncTime;
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
        firsttime=0;
	/* To get rawfile working saj*/
	error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt, (void*)ckt->CKTcurJob,
					       ckt->CKTcurJob->JOBname,timeUid,IF_REAL,666,nameList,
					       666,&(((TRANan*)ckt->CKTcurJob)->TRANplot));/*magic 666 nums as flags */
	if(error) {
	  fprintf(stderr, "Couldn't relink rawfile\n");
	  return error;
	}
	/*end saj*/	
        goto resume;
    }

/* 650 */
nextTime:

    /* begin LTRA code addition */
    if (ckt->CKTtimePoints) {
	ckt->CKTtimeIndex++;
	if (ckt->CKTtimeIndex >= ckt->CKTtimeListSize) {
	    /* need more space */
	    int need;
	    need = 0.5 + (ckt->CKTfinalTime - ckt->CKTtime) / maxstepsize;
	    if (need < ckt->CKTsizeIncr)
		need = ckt->CKTsizeIncr;
	    ckt->CKTtimeListSize += need;
	    ckt->CKTtimePoints = (double *) REALLOC( (char *)
		 ckt->CKTtimePoints, sizeof(double) * ckt->CKTtimeListSize);
	    ckt->CKTsizeIncr *= 1.4;
	}
	*(ckt->CKTtimePoints + ckt->CKTtimeIndex) = ckt->CKTtime;
    }
    /* end LTRA code addition */

    error = CKTaccept(ckt);
    /* check if current breakpoint is outdated; if so, clear */
    if (ckt->CKTtime > *(ckt->CKTbreaks)) CKTclrBreak(ckt);

/*
 * Breakpoint handling scheme:
 * When a timepoint t is accepted (by CKTaccept), clear all previous
 * breakpoints, because they will never be needed again.
 *
 * t may itself be a breakpoint, or indistinguishably close. DON'T
 * clear t itself; recognise it as a breakpoint and act accordingly
 *
 * if t is not a breakpoint, limit the timestep so that the next
 * breakpoint is not crossed
 */

#ifdef STEPDEBUG
   printf("Delta %g accepted at time %g\n",ckt->CKTdelta,ckt->CKTtime);
   fflush(stdout);
#endif /* STEPDEBUG */
    ckt->CKTstat->STATaccepted ++;
    ckt->CKTbreak=0;
    /* XXX Error will cause single process to bail. */
    if(error)  {
        ckt->CKTcurrentAnalysis = DOING_TRAN;
        ckt->CKTstat->STATtranTime += (*(SPfrontEnd->IFseconds))()-startTime;
        ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters;
        ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime -
                startdTime;
        ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime -
                startsTime;
        ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime -
                startlTime;
        ckt->CKTstat->STATtranCombTime += ckt->CKTstat->STATcombineTime -
                startcTime;
        ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime -
                startkTime;
        return(error);
    }
#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

    if(g_ipc.enabled) {

        /* Send event-driven results */
        EVTdump(ckt, IPC_ANAL_TRAN, 0.0);

        /* Then follow with analog results... */

        /* Test to see if delta was cut by a breakpoint, */
        /* a non-convergence, or a too large truncation error */
        if(ipc_firsttime)
            ipc_delta_cut = IPC_FALSE;
        else if(ckt->CKTtime < (ipc_last_time + (0.999 * ipc_last_delta)))
            ipc_delta_cut = IPC_TRUE;
        else
            ipc_delta_cut = IPC_FALSE;

        /* Record the data required to check for delta cuts */
        ipc_last_time = ckt->CKTtime;
        ipc_last_delta = MIN(ckt->CKTdelta, ckt->CKTmaxStep);

        /* Send results data if time since last dump is greater */
        /* than 'mintime', or if first or second timepoints, */
        /* or if delta was cut */
        if( (ckt->CKTtime >= (g_ipc.mintime + g_ipc.last_time)) ||
            ipc_firsttime || ipc_secondtime || ipc_delta_cut ) {

            ipc_send_data_prefix(ckt->CKTtime);
            CKTdump(ckt,ckt->CKTtime,
                  (((TRANan*)ckt->CKTcurJob)->TRANplot));
            ipc_send_data_suffix();

            if(ipc_firsttime) {
                ipc_firsttime = IPC_FALSE;
                ipc_secondtime = IPC_TRUE;
            }
            else if(ipc_secondtime)
                ipc_secondtime = IPC_FALSE;

            g_ipc.last_time = ckt->CKTtime;
        }
    }
    else  
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#endif
#ifdef CLUSTER
      CLUoutput(ckt);
#endif    
    if(ckt->CKTtime >= ckt->CKTinitTime) CKTdump(ckt,ckt->CKTtime,
            (((TRANan*)ckt->CKTcurJob)->TRANplot));
#ifdef XSPICE
/* gtri - begin - wbk - Update event queues/data for accepted timepoint */

    /* Note: this must be done AFTER sending results to SI so it can't */
    /* go next to CKTaccept() above */
    if(ckt->evt->counts.num_insts > 0)
        EVTaccept(ckt, ckt->CKTtime);
/* gtri - end - wbk - Update event queues/data for accepted timepoint */
#endif	    
    ckt->CKTstat->STAToldIter = ckt->CKTstat->STATnumIter;
    if(fabs(ckt->CKTtime - ckt->CKTfinalTime) < ckt->CKTminBreak) {
        /*printf(" done:  time is %g, final time is %g, and tol is %g\n",*/
        /*ckt->CKTtime,ckt->CKTfinalTime,ckt->CKTminBreak);*/
        (*(SPfrontEnd->OUTendPlot))( (((TRANan*)ckt->CKTcurJob)->TRANplot));
        ckt->CKTcurrentAnalysis = 0;
        ckt->CKTstat->STATtranTime += (*(SPfrontEnd->IFseconds))()-startTime;
        ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters;
        ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime -
                startdTime;
        ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime -
                startsTime;
        ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime -
                startlTime;
        ckt->CKTstat->STATtranCombTime += ckt->CKTstat->STATcombineTime -
                startcTime;
        ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime -
                startkTime;
#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
            ckt->CKTsenInfo->SENmode = save;
#ifdef SENSDEBUG
            fclose(outsen);
#endif /* SENSDEBUG */
        }
#endif
        return(OK);
    }
    if( (*(SPfrontEnd->IFpauseTest))() ) {
        /* user requested pause... */
        ckt->CKTcurrentAnalysis = DOING_TRAN;
        ckt->CKTstat->STATtranTime += (*(SPfrontEnd->IFseconds))()-startTime;
        ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters;
        ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime -
                startdTime;
        ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime -
                startsTime;
        ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime -
                startlTime;
        ckt->CKTstat->STATtranCombTime += ckt->CKTstat->STATcombineTime -
                startcTime;
        ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime -
                startkTime;
        return(E_PAUSE);
    }
resume:
#ifdef STEPDEBUG
    if( (ckt->CKTdelta <= ckt->CKTfinalTime/50) && 
            (ckt->CKTdelta <= ckt->CKTmaxStep)) {
            ;
    } else {
        if(ckt->CKTfinalTime/50<ckt->CKTmaxStep) {
            (void)printf("limited by Tstop/50\n");
        } else {
            (void)printf("limited by Tmax\n");
        }
    }
#endif
#ifdef HAS_WINDOWS
	 SetAnalyse( "tran", (int)((ckt->CKTtime * 100) / ckt->CKTfinalTime));
#endif
    ckt->CKTdelta = 
            MIN(ckt->CKTdelta,ckt->CKTmaxStep);
#ifdef XSPICE
/* gtri - begin - wbk - Cut integration order if first timepoint after breakpoint */
    if(ckt->CKTtime == g_mif_info.breakpoint.last)
        ckt->CKTorder = 1;
/* gtri - end   - wbk - Cut integration order if first timepoint after breakpoint */

#endif


	/* are we at a breakpoint, or indistinguishably close? */
    if ((ckt->CKTtime == *(ckt->CKTbreaks)) || (*(ckt->CKTbreaks) -
    				(ckt->CKTtime) <= ckt->CKTdelmin)) {
        /* first timepoint after a breakpoint - cut integration order */
        /* and limit timestep to .1 times minimum of time to next breakpoint,
         *  and previous timestep
         */
        ckt->CKTorder = 1;
#ifdef STEPDEBUG
        if( (ckt->CKTdelta >.1* ckt->CKTsaveDelta) ||
                (ckt->CKTdelta > .1*(*(ckt->CKTbreaks+1)-*(ckt->CKTbreaks))) ) {
            if(ckt->CKTsaveDelta < (*(ckt->CKTbreaks+1)-*(ckt->CKTbreaks)))  {
                (void)printf("limited by pre-breakpoint delta\n");
            } else {
                (void)printf("limited by next breakpoint\n");
            }
        }
#endif

            ckt->CKTdelta = MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta,
                    *(ckt->CKTbreaks+1)-*(ckt->CKTbreaks)));

        if(firsttime) {
            ckt->CKTdelta /= 10;
#ifdef STEPDEBUG
            (void)printf("delta cut for initial timepoint\n");
#endif
        } 

#ifdef XSPICE
   }

/* gtri - begin - wbk - Add Breakpoint stuff */

    if(ckt->CKTtime + ckt->CKTdelta >= g_mif_info.breakpoint.current) {
        /* If next time > temporary breakpoint, force it to the breakpoint */
        /* And mark that timestep was set by temporary breakpoint */
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
        g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
    }
    else {
        /* Else, mark that timestep was not set by temporary breakpoint */
        g_mif_info.breakpoint.last = 1.0e30;
    }

/* gtri - end - wbk - Add Breakpoint stuff */

/* gtri - begin - wbk - Modify Breakpoint stuff */
    /* Throw out any permanent breakpoint times <= current time */
    while(1) {
        if(*(ckt->CKTbreaks) <= (ckt->CKTtime + ckt->CKTminBreak))
            CKTclrBreak(ckt);
        else
            break;
    }
    /* Force the breakpoint if appropriate */
    if((ckt->CKTtime + ckt->CKTdelta) > *(ckt->CKTbreaks)) {
        ckt->CKTbreak = 1;
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = *(ckt->CKTbreaks) - ckt->CKTtime;
    }

/* gtri - end - wbk - Modify Breakpoint stuff */
#else /* !XSPICE */

	/* don't want to get below delmin for no reason */
	ckt->CKTdelta = MAX(ckt->CKTdelta, ckt->CKTdelmin*2.0);
    }
    else if(ckt->CKTtime + ckt->CKTdelta >= *(ckt->CKTbreaks)) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = *(ckt->CKTbreaks) - ckt->CKTtime;
#ifdef STEPDEBUG
        (void)printf("delta cut to %g to hit breakpoint\n",ckt->CKTdelta);
        fflush(stdout);
#endif
        ckt->CKTbreak = 1; /* why? the current pt. is not a bkpt. */
    }
#ifdef CLUSTER
    if(!CLUsync(ckt->CKTtime,&ckt->CKTdelta,0)){
      printf("Sync error!\n");
      exit(0);
    }
#endif
#ifdef PARALLEL_ARCH
    DGOP_( &type, &(ckt->CKTdelta), &length, "min" );
#endif /* PARALLEL_ARCH */

#endif /* XSPICE */

#ifdef XSPICE
/* gtri - begin - wbk - Do event solution */

    if(ckt->evt->counts.num_insts > 0) {

        /* if time = 0 and op_alternate was specified as false during */
        /* dcop analysis, call any changed instances to let them */
        /* post their outputs with their associated delays */
        if((ckt->CKTtime == 0.0) && (! ckt->evt->options.op_alternate))
            EVTiter(ckt);

        /* while there are events on the queue with event time <= next */
        /* projected analog time, process them */
        while((g_mif_info.circuit.evt_step = EVTnext_time(ckt))
               <= (ckt->CKTtime + ckt->CKTdelta)) {

            /* Initialize temp analog bkpt to infinity */
            g_mif_info.breakpoint.current = 1e30;

            /* Pull items off queue and process them */
            EVTdequeue(ckt, g_mif_info.circuit.evt_step);
            EVTiter(ckt);

            /* If any instances have forced an earlier */
            /* next analog time, cut the delta */
            if(*(ckt->CKTbreaks) < g_mif_info.breakpoint.current)
                if(*(ckt->CKTbreaks) > (ckt->CKTtime + ckt->CKTminBreak))
                    g_mif_info.breakpoint.current = *(ckt->CKTbreaks);
            if(g_mif_info.breakpoint.current < (ckt->CKTtime + ckt->CKTdelta)) {
                /* Breakpoint must be > last accepted timepoint */
                /* and >= current event time */
                if(g_mif_info.breakpoint.current > (ckt->CKTtime + ckt->CKTminBreak)
                   && (g_mif_info.breakpoint.current >= g_mif_info.circuit.evt_step)) {
                    ckt->CKTsaveDelta = ckt->CKTdelta;
                    ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
                    g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
                }
            }

        } /* end while next event time <= next analog time */
    } /* end if there are event instances */

/* gtri - end - wbk - Do event solution */
#endif
    for(i=5;i>=0;i--) {
        ckt->CKTdeltaOld[i+1]=ckt->CKTdeltaOld[i];
    }
    ckt->CKTdeltaOld[0]=ckt->CKTdelta;

    temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
    for(i=ckt->CKTmaxOrder;i>=0;i--) {
        ckt->CKTstates[i+1] = ckt->CKTstates[i];
    }
    ckt->CKTstates[0] = temp;

/* 600 */
    while (1) {
#ifdef CLUSTER
      redostep = 1;
#endif
#ifdef XSPICE
/* gtri - add - wbk - 4/17/91 - Fix Berkeley bug */
/* This is needed here to allow CAPask to output currents */
/* during Transient analysis.  A grep for CKTcurrentAnalysis */
/* indicates that it should not hurt anything else ... */

        ckt->CKTcurrentAnalysis = DOING_TRAN;

/* gtri - end - wbk - 4/17/91 - Fix Berkeley bug */
#endif
        olddelta=ckt->CKTdelta;
        /* time abort? */
        ckt->CKTtime += ckt->CKTdelta;
#ifdef CLUSTER
	CLUinput(ckt);
#endif	
        ckt->CKTdeltaOld[0]=ckt->CKTdelta;
        NIcomCof(ckt);
#ifdef PREDICTOR
        error = NIpred(ckt);
#endif /* PREDICTOR */
        save_mode = ckt->CKTmode;
        save_order = ckt->CKTorder;
#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Initialize temporary breakpoint to infinity */
        g_mif_info.breakpoint.current = 1.0e30;

/* gtri - end - wbk - Add Breakpoint stuff */


/* gtri - begin - wbk - add convergence problem reporting flags */
        /* delta is forced to equal delmin on last attempt near line 650 */
        if(ckt->CKTdelta <= ckt->CKTdelmin)
	  ckt->enh->conv_debug.last_NIiter_call = MIF_TRUE;
        else
	  ckt->enh->conv_debug.last_NIiter_call = MIF_FALSE;
/* gtri - begin - wbk - add convergence problem reporting flags */
	


/* gtri - begin - wbk - Call all hybrids */

/* gtri - begin - wbk - Set evt_step */

        if(ckt->evt->counts.num_insts > 0) {
	  g_mif_info.circuit.evt_step = ckt->CKTtime;
        }
/* gtri - end - wbk - Set evt_step */
#endif
	
        converged = NIiter(ckt,ckt->CKTtranMaxIter);
	
#ifdef XSPICE
        if(ckt->evt->counts.num_insts > 0) 
		{
            g_mif_info.circuit.evt_step = ckt->CKTtime;
            EVTcall_hybrids(ckt);
        }		
/* gtri - end - wbk - Call all hybrids */

#endif	
        ckt->CKTstat->STATtimePts ++;
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        if(firsttime) {
            for(i=0;i<ckt->CKTnumStates;i++) {
                *(ckt->CKTstate2+i) = *(ckt->CKTstate1+i);
                *(ckt->CKTstate3+i) = *(ckt->CKTstate1+i);
            }
        }
		/* txl, cpl addition */
		if (converged == 1111) {
            return(converged);
		}

        if(converged != 0) {
#ifndef CLUSTER	
            ckt->CKTtime = ckt->CKTtime -ckt->CKTdelta;
            ckt->CKTstat->STATrejected ++;
#endif	    
            ckt->CKTdelta = ckt->CKTdelta/8;
#ifdef STEPDEBUG
           (void)printf("delta cut to %g for non-convergance\n",ckt->CKTdelta);
            fflush(stdout);
#endif
            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Force backup if temporary breakpoint is < current time */
        } else if(g_mif_info.breakpoint.current < ckt->CKTtime) {
            ckt->CKTsaveDelta = ckt->CKTdelta;
            ckt->CKTtime -= ckt->CKTdelta;
            ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
            g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

/* gtri - end - wbk - Add Breakpoint stuff */
#endif

        } else {
            if (firsttime) {
#ifdef WANT_SENSE2
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
                    save1 = ckt->CKTmode;
                    save2 = ckt->CKTorder;
                    ckt->CKTmode = save_mode;
                    ckt->CKTorder = save_order;
                    if(error = CKTsenDCtran(ckt)) return(error);
                    ckt->CKTmode = save1;
                    ckt->CKTorder = save2;
                }
#endif
                firsttime =0;
#ifndef CLUSTER		
                goto nextTime;  /* no check on
                                 * first time point
                                 */
#else
		redostep = 0;
		goto chkStep;
#endif				 
            }
            new = ckt->CKTdelta;
            error = CKTtrunc(ckt,&new);
            if(error) {
                ckt->CKTcurrentAnalysis = DOING_TRAN;
                ckt->CKTstat->STATtranTime += 
                        (*(SPfrontEnd->IFseconds))()-startTime;
                ckt->CKTstat->STATtranIter += 
                        ckt->CKTstat->STATnumIter - startIters;
                ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime
                        - startdTime;
                ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime
                        - startsTime;
		ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime
			- startlTime;
		ckt->CKTstat->STATtranCombTime += ckt->CKTstat->STATcombineTime
			- startcTime;
		ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime
			- startkTime;
                return(error);
            }
            if(new>.9 * ckt->CKTdelta) {
                if(ckt->CKTorder == 1) {
                    new = ckt->CKTdelta;
                    ckt->CKTorder = 2;
                    error = CKTtrunc(ckt,&new);
                    if(error) {
                        ckt->CKTcurrentAnalysis = DOING_TRAN;
                        ckt->CKTstat->STATtranTime += 
                                (*(SPfrontEnd->IFseconds))()-startTime;
                        ckt->CKTstat->STATtranIter += 
                                ckt->CKTstat->STATnumIter - startIters;
                        ckt->CKTstat->STATtranDecompTime += 
                                ckt->CKTstat->STATdecompTime - startdTime;
                        ckt->CKTstat->STATtranSolveTime += 
                                ckt->CKTstat->STATsolveTime - startsTime;
			ckt->CKTstat->STATtranLoadTime +=
				ckt->CKTstat->STATloadTime - startlTime;
			ckt->CKTstat->STATtranCombTime +=
				ckt->CKTstat->STATcombineTime - startcTime;
			ckt->CKTstat->STATtranSyncTime +=
				ckt->CKTstat->STATsyncTime - startkTime;
                        return(error);
                    }
                    if(new <= 1.05 * ckt->CKTdelta) {
                        ckt->CKTorder = 1;
                    }
                }
                /* time point OK  - 630*/
                ckt->CKTdelta = new;
#ifdef STEPDEBUG
               (void)printf(
                  "delta set to truncation error result: %g. Point accepted\n",
                  ckt->CKTdelta);
                fflush(stdout);
#endif
#ifdef WANT_SENSE2
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)){
                    save1 = ckt->CKTmode;
                    save2 = ckt->CKTorder;
                    ckt->CKTmode = save_mode;
                    ckt->CKTorder = save_order;
                    if(error = CKTsenDCtran(ckt)) return(error);
                    ckt->CKTmode = save1;
                    ckt->CKTorder = save2;
                }
#endif
#ifndef CLUSTER
                /* go to 650 - trapezoidal */		
                goto nextTime;
#else
		redostep = 0;
		goto chkStep;
#endif		
            } else {
#ifndef CLUSTER	    
                ckt->CKTtime = ckt->CKTtime -ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
#endif		
                ckt->CKTdelta = new;
#ifdef STEPDEBUG
                (void)printf(
                    "delta set to truncation error result:point rejected\n");
#endif
            }
        }
#ifdef PARALLEL_ARCH
	DGOP_( &type, &(ckt->CKTdelta), &length, "min" );
#endif /* PARALLEL_ARCH */

        if (ckt->CKTdelta <= ckt->CKTdelmin) {
            if (olddelta > ckt->CKTdelmin) {
                ckt->CKTdelta = ckt->CKTdelmin;
#ifdef STEPDEBUG
                (void)printf("delta at delmin\n");
#endif
            } else {
                ckt->CKTcurrentAnalysis = DOING_TRAN;
                ckt->CKTstat->STATtranTime += 
                        (*(SPfrontEnd->IFseconds))()-startTime;
                ckt->CKTstat->STATtranIter += 
                        ckt->CKTstat->STATnumIter - startIters;
                ckt->CKTstat->STATtranDecompTime += 
                        ckt->CKTstat->STATdecompTime - startdTime;
                ckt->CKTstat->STATtranSolveTime += 
                        ckt->CKTstat->STATsolveTime - startsTime;
		ckt->CKTstat->STATtranLoadTime +=
			ckt->CKTstat->STATloadTime - startlTime;
		ckt->CKTstat->STATtranCombTime +=
			ckt->CKTstat->STATcombineTime - startcTime;
		ckt->CKTstat->STATtranSyncTime +=
			ckt->CKTstat->STATsyncTime - startkTime;
		errMsg = CKTtrouble((void *) ckt, "Timestep too small");
                return(E_TIMESTEP);
            }
        }
#ifdef XSPICE
/* gtri - begin - wbk - Do event backup */

        if(ckt->evt->counts.num_insts > 0)
            EVTbackup(ckt, ckt->CKTtime + ckt->CKTdelta);

/* gtri - end - wbk - Do event backup */
#endif	
#ifdef CLUSTER
    chkStep:
	if(CLUsync(ckt->CKTtime,&ckt->CKTdelta,redostep)){
	  goto nextTime;
	} else {
	  ckt->CKTtime -= olddelta;
	  ckt->CKTstat->STATrejected ++;
	}
#endif	
    }
    /* NOTREACHED */
}
