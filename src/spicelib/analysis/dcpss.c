// (diff buffer-file-name "dctran.c" "-b -w -U2")
/**********
 Author: 2010-05 Stefano Perticaroli ``spertica''
 **********/

/* subroutine to do DC PSS analysis */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/pssdefs.h"
#include "ngspice/sperror.h"

/* for FFT */
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"
#include "ngspice/fteparse.h"
#include "ngspice/const.h"
#include "../../frontend/fourier.h"
#include "../../frontend/variable.h"

#include "ngspice/config.h"
#include "cktaccept.h"
#include "ngspice/trandefs.h"
#include "ngspice/fteext.h"
#include "ngspice/missing_math.h"

#ifdef XSPICE
/* gtri - add - wbk - Add headers */
#include "ngspice/miftypes.h"

#include "ngspice/evt.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk - Add headers */
#endif

#ifdef CLUSTER
#include "ngspice/cluster.h"
#endif

#ifdef HAS_WINDOWS    /* hvogt 10.03.99, nach W. Mues */
void SetAnalyse( char * Analyse, int Percent);
#endif


#define INIT_STATS() \
do { \
    startTime = SPfrontEnd->IFseconds();        \
    startIters = ckt->CKTstat->STATnumIter;     \
    startdTime = ckt->CKTstat->STATdecompTime;  \
    startsTime = ckt->CKTstat->STATsolveTime;   \
    startlTime = ckt->CKTstat->STATloadTime;    \
    startcTime = ckt->CKTstat->STATcombineTime; \
    startkTime = ckt->CKTstat->STATsyncTime;    \
} while(0)

#define UPDATE_STATS(analysis) \
do { \
    ckt->CKTcurrentAnalysis = analysis; \
    ckt->CKTstat->STATtranTime += SPfrontEnd->IFseconds() - startTime; \
    ckt->CKTstat->STATtranIter += ckt->CKTstat->STATnumIter - startIters; \
    ckt->CKTstat->STATtranDecompTime += ckt->CKTstat->STATdecompTime - startdTime; \
    ckt->CKTstat->STATtranSolveTime += ckt->CKTstat->STATsolveTime - startsTime; \
    ckt->CKTstat->STATtranLoadTime += ckt->CKTstat->STATloadTime - startlTime; \
    ckt->CKTstat->STATtranCombTime += ckt->CKTstat->STATcombineTime - startcTime; \
    ckt->CKTstat->STATtranSyncTime += ckt->CKTstat->STATsyncTime - startkTime; \
} while(0)


int
CKTfour(int, int, double *, double *, double *, double, double *, double *, double *, double *,double *);

int
DCpss(CKTcircuit *ckt, int restart)
{
    PSSan *job = (PSSan *) ckt->CKTcurJob;

    int oscnNode;
    int i;
    double olddelta;
    double delta;
    double newdelta;
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
    IFuid	freqUid;
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
    /* new variables - to be reorganized */
    double  time_temp, gf_history[1024], rr_history[1024];
    int msize, in_stabilization = 1, in_pss = 0, shooting_cycle_counter = 0, k;
    long int nextTime_count=0, ntc_start_sh=0;
    double *RHS_copy_se, err_0 = 1e30, time_temp_0, delta_t;
    double time_err_min_1=0, time_err_min_0=0, err_min_0=1.0e30, err_min_1, delta_0, delta_1;
    double flag_tu_2, times_fft[8192], err_1=0, err_max, time_err_max;
    int flag_tu_1=0, pss_cycle_counter=1, pss_points_cycle=0, i4, i5, k1,rest;
    int count_1, count_2, count_3, count_4, count_5, count_6, count_7, dynamic_test=0;
    double ntc_mv, ntc_vec[4], ntc_old, gf_last_0=1e+30, gf_last_1=313;
    double err_last = 0, thd;
    double *psstimes, *pssvalues, *pssValues, tv_03, tv_04,
           *pssfreqs, *pssmags, *pssphases, *pssnmags, *pssnphases, *pssResults,
           *RHS_max, *RHS_min, err_conv_ref, *S_old, *S_diff;

    printf("Periodic Steady State analysis started.\n");

    oscnNode = job->PSSoscNode->number;
    printf("PSS guessed frequency %g.\n", ckt->CKTguessedFreq);
    printf("PSS points %ld.\n", ckt->CKTpsspoints);
    printf("PSS harmonics number %d.\n", ckt->CKTharms);
    printf("PSS steady coefficient %g.\n", ckt->CKTsteady_coeff);

    /* set first delta time step and circuit time */
    delta=ckt->CKTstep;
    ckt->CKTtime=ckt->CKTinitTime;
    ckt->CKTfinalTime=ckt->CKTstabTime;
    /* printf("initial delta: %g\n", ckt->CKTdelta); */

    /* 100906 - Paste from dctran.c */
    if(restart || ckt->CKTtime == 0) {
        delta=MIN(1/(ckt->CKTguessedFreq)/100,ckt->CKTstep);

#ifdef STEPDEBUG
        printf("delta = %g    finalTime/200: %g    CKTstep: %g\n",delta,ckt->CKTfinalTime/200,ckt->CKTstep);
#endif
        /* begin LTRA code addition */
        if (ckt->CKTtimePoints != NULL)
            FREE(ckt->CKTtimePoints);

        if (ckt->CKTstep >= ckt->CKTmaxStep)
            maxstepsize = ckt->CKTstep;
        else
            maxstepsize = ckt->CKTmaxStep;

        ckt->CKTsizeIncr = 10;
        ckt->CKTtimeIndex = -1; /* before the DC soln has been stored */
        ckt->CKTtimeListSize = (int)(1 / (ckt->CKTguessedFreq) / maxstepsize + 0.5);
        ltra_num = CKTtypelook("LTRA");
        if (ltra_num >= 0 && ckt->CKThead[ltra_num] != NULL)
            ckt->CKTtimePoints = NEWN(double, ckt->CKTtimeListSize);
        /* end LTRA code addition */

        if(ckt->CKTbreaks) FREE(ckt->CKTbreaks);
        ckt->CKTbreaks = TMALLOC(double, 2);
        if(ckt->CKTbreaks == NULL) return(E_NOMEM);
        ckt->CKTbreaks[0] = 0;
        ckt->CKTbreaks[1] = ckt->CKTfinalTime;
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
        SPfrontEnd->IFnewUid (ckt, &timeUid, NULL,
                              "time", UID_OTHER, NULL);
        error = SPfrontEnd->OUTpBeginPlot (
            ckt, ckt->CKTcurJob,
            "Time Domain Periodic Steady State",
            timeUid, IF_REAL,
            numNames, nameList, IF_REAL, &(job->PSSplot_td));
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
        if(ckt->evt->counts.num_insts != 0) {
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

#ifdef STEPDEBUG
        if(converged != 0) {
            fprintf(stdout,"\nTransient solution failed -\n");
            CKTncDump(ckt);
            fprintf(stdout,"\n");
            fflush(stdout);
        } else if (!ft_noacctprint && !ft_noinitprint) {
            fprintf(stdout,"\nInitial Transient Solution\n");
            fprintf(stdout,"--------------------------\n\n");
            fprintf(stdout,"%-30s %15s\n", "Node", "Voltage");
            fprintf(stdout,"%-30s %15s\n", "----", "-------");
            for(node=ckt->CKTnodes->next; node; node=node->next) {
                if (strstr(node->name, "#branch") || !strstr(node->name, "#"))
                    fprintf(stdout,"%-30s %15g\n", node->name,
                            ckt->CKTrhsOld[node->number] );
            }
            fprintf(stdout,"\n");
            fflush(stdout);
        }
#endif

        if(converged != 0) return(converged);
#ifdef XSPICE
/* gtri - add - wbk - 12/19/90 - Add IPC stuff */

        /* Send the operating point results for Mspice compatibility */
        if(g_ipc.enabled) {
            ipc_send_dcop_prefix();
            CKTdump(ckt, 0.0, job->PSSplot_td);
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
        for(i=0; i<7; i++) {
            ckt->CKTdeltaOld[i]=ckt->CKTmaxStep;
        }
        ckt->CKTdelta = delta;
#ifdef STEPDEBUG
        (void)printf("delta initialized to %g\n",ckt->CKTdelta);
#endif
        ckt->CKTsaveDelta = ckt->CKTfinalTime/50;

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)) {
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
        bcopy(ckt->CKTstate0, ckt->CKTstate1,
              (size_t) ckt->CKTnumStates * sizeof(double));

#ifdef WANT_SENSE2
        if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)) {
            size = SMPmatSize(ckt->CKTmatrix);
            for(i = 1; i<=size ; i++)
                ckt->CKTrhsOp[i] = ckt->CKTrhsOld[i];
        }
#endif

        INIT_STATS();
#ifdef CLUSTER
        CLUsetup(ckt);
#endif
    } else {
        /*saj As traninit resets CKTmode */
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        /* saj */
        INIT_STATS();
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
        firsttime=0;
        /* To get rawfile working saj*/
        error = SPfrontEnd->OUTpBeginPlot (
            NULL, NULL,
            NULL,
            NULL, 0,
            666, NULL, 666, &(job->PSSplot_td));
        if(error) {
            fprintf(stderr, "Couldn't relink rawfile\n");
            return error;
        }
        /*end saj*/
        goto resume;
    }

    /* 650 */
nextTime:

    nextTime_count=nextTime_count+1;
    /* Does not start from initial time avoiding IC issues */
    if ( nextTime_count>=10 && nextTime_count<8202 && in_stabilization) {
        times_fft[nextTime_count-10]=ckt->CKTtime;
    }

    /* begin LTRA code addition */
    if (ckt->CKTtimePoints) {
        ckt->CKTtimeIndex++;
        if (ckt->CKTtimeIndex >= ckt->CKTtimeListSize) {
            /* need more space */
            int need;
            if (in_stabilization) {
                need = (int)(0.5 + (ckt->CKTstabTime - ckt->CKTtime) / maxstepsize); /* FIXME, ceil ? */
            } else {
                need = (int)(0.5 + (time_temp + (1 / ckt->CKTguessedFreq) - ckt->CKTtime) / maxstepsize);
            }
            if (need < ckt->CKTsizeIncr)
                need = ckt->CKTsizeIncr;
            ckt->CKTtimeListSize += need;
            ckt->CKTtimePoints = TREALLOC(double, ckt->CKTtimePoints, ckt->CKTtimeListSize);
            ckt->CKTsizeIncr = (int) ceil(1.4 * ckt->CKTsizeIncr);
        }
        ckt->CKTtimePoints[ckt->CKTtimeIndex] = ckt->CKTtime;
    }
    /* end LTRA code addition */

    error = CKTaccept(ckt);
    /* check if current breakpoint is outdated; if so, clear */
    if (ckt->CKTtime > ckt->CKTbreaks[0]) CKTclrBreak(ckt);

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
    printf("Delta %g accepted at time %g (finaltime: %g)\n",ckt->CKTdelta,ckt->CKTtime,ckt->CKTfinalTime);
    fflush(stdout);
#endif /* STEPDEBUG */
    ckt->CKTstat->STATaccepted ++;
    ckt->CKTbreak=0;
    /* XXX Error will cause single process to bail. */
    if(error)  {
        UPDATE_STATS(DOING_TRAN);
        return(error);
    }
#ifdef XSPICE
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */

    if(g_ipc.enabled) {

        if ( in_pss && pss_cycle_counter==1 ) {
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
                CKTdump(ckt, ckt->CKTtime, job->PSSplot_td);
                ipc_send_data_suffix();

                if(ipc_firsttime) {
                    ipc_firsttime = IPC_FALSE;
                    ipc_secondtime = IPC_TRUE;
                } else if(ipc_secondtime) {
                    ipc_secondtime = IPC_FALSE;
                }

                g_ipc.last_time = ckt->CKTtime;
            }
        }
    } else
/* gtri - modify - wbk - 12/19/90 - Send IPC stuff */
#endif
#ifdef CLUSTER
        if ( in_pss && pss_cycle_counter==1 )
            CLUoutput(ckt);
#endif
    if ( in_pss && pss_cycle_counter==1 ) {
        if(ckt->CKTtime >= ckt->CKTinitTime)
            CKTdump(ckt, ckt->CKTtime, job->PSSplot_td);
        psstimes[pss_points_cycle] = ckt->CKTtime;
        for(count_1=1; count_1<msize+1; count_1++)
            pssvalues[count_1-1 + pss_points_cycle*msize] = ckt->CKTrhsOld[count_1];
        pss_points_cycle++;
    }
#ifdef XSPICE
/* gtri - begin - wbk - Update event queues/data for accepted timepoint */
    /* Note: this must be done AFTER sending results to SI so it can't */
    /* go next to CKTaccept() above */
    if(ckt->evt->counts.num_insts > 0)
        EVTaccept(ckt, ckt->CKTtime);
/* gtri - end - wbk - Update event queues/data for accepted timepoint */
#endif
    ckt->CKTstat->STAToldIter = ckt->CKTstat->STATnumIter;

    /* ***********************************/
    /* ******* SHOOTING CODE BLOCK *******/
    /* ***********************************/
    if (in_stabilization) {
        /* test if stabTime has been reached */
        if ( AlmostEqualUlps( ckt->CKTtime , ckt->CKTstabTime, 100 ) ) {
            /* use the 'dirty hack' to get near the fundamental frequency */
            if (nextTime_count<8192) {
                for(count_2=1; count_2<nextTime_count-2; count_2++) {
                    if (times_fft[count_2]>0) {
                        delta_t += (times_fft[count_2]-times_fft[count_2-1])/nextTime_count;
                    }
                }
            } else {
                for(count_2=1; count_2<8192; count_2++) {
                    if (times_fft[count_2]>0) {
                        delta_t += (times_fft[count_2]-times_fft[count_2-1])/8192;
                    }
                }
            }
            if (ckt->CKTguessedFreq>1.1/delta_t/10 || ckt->CKTguessedFreq<0.9/delta_t/10) {
                ckt->CKTguessedFreq=1.0/delta_t/10; /*** FREQUENCY INITIAL  GUESS ***/
                printf("Frequency initial guess changed to %g from stabilization transient analysis.\n",ckt->CKTguessedFreq);
            }
            time_temp=ckt->CKTtime;
            ckt->CKTfinalTime=time_temp+2/(ckt->CKTguessedFreq);
            /* set the first requested breakpoint */
            CKTsetBreak(ckt, time_temp+1/(ckt->CKTguessedFreq));
            printf("Exiting from stabilization.\n");
            printf("Time of first shooting evaluation will be %1.10g.\n",time_temp+1/(ckt->CKTguessedFreq));
            /* next time is no more in stab - unset the flag */
            in_stabilization=0;
            /* get matrix size and allocate memory */
            msize = SMPmatSize(ckt->CKTmatrix);
            RHS_copy_se = TMALLOC(double, msize);
            RHS_max     = TMALLOC(double, msize);
            RHS_min     = TMALLOC(double, msize);
            S_old       = TMALLOC(double, msize);
            S_diff      = TMALLOC(double, msize);
            /* print RHS on exiting from stab */
            printf("RHS on exiting from stabilization: ");
            for(count_3 = 1; count_3 <= msize; count_3++) {
                RHS_copy_se[count_3-1] = ckt->CKTrhsOld[count_3];
                printf("%-15g ", RHS_copy_se[count_3-1]);
            }
            printf("\n");
        }

        /* ELSE not in stabilization but in shooting */
    } else if ( !in_pss ) {

        double err, f_proj;

        /* error norms of RHS solution on every accepted nextTime if out of stabilization */
        err=0;
        for(count_4 = 1; count_4 <= msize; count_4++) {
            double diff = ckt->CKTrhsOld[count_4] - RHS_copy_se[count_4-1];
            err += diff * diff;
            /* save max and min per node or branch on every estimated period */
            if (RHS_max[count_4-1] < ckt->CKTrhsOld[count_4])
                RHS_max[count_4-1] = ckt->CKTrhsOld[count_4];
            if (RHS_min[count_4-1] > ckt->CKTrhsOld[count_4])
                RHS_min[count_4-1] = ckt->CKTrhsOld[count_4];
        }
        err=sqrt(err);
        /*** frequency projection ***/
        f_proj=(err-err_last)*(ckt->CKTguessedFreq);

        err_last=err;

        /* Start frequency estimation */
        if (err<err_0 && ckt->CKTtime>=time_temp+0.5/ckt->CKTguessedFreq) { /* far enough from time temp... */
            if (err<err_min_0) {
                err_min_1=err_min_0; 			/* store previous minimum of RHS vector error */
                err_min_0=err; 			/* store minimum of RHS vector error */
                time_err_min_1=time_err_min_0; 	/* store previous minimum of RHS vector error time */
                time_err_min_0=ckt->CKTtime; 		/* store minimum of RHS vector error time */
                delta_1=delta_0;
                delta_0=ckt->CKTdelta;
            }
        }
        err_0=err;

        if (err>err_1 && ckt->CKTtime>=time_temp+0.1/ckt->CKTguessedFreq) { /* far enough from time temp... */
            if (err>err_max) {
                err_max=err; 				/* store maximum of RHS vector error */
                time_err_max=ckt->CKTtime; 		/* store maximum of RHS vector error time */
            }
        }
        err_1=err;

        /* if evolution is near shooting */
        if ( AlmostEqualUlps( ckt->CKTtime, time_temp+1/ckt->CKTguessedFreq, 10 ) || (ckt->CKTtime > time_temp+1/ckt->CKTguessedFreq) ) {
            if (shooting_cycle_counter == 0) {
                /* If first time in shooting we warn about that ! */
                ntc_start_sh=nextTime_count;
                printf("In shooting...\n");
            }

            /* Take mean value of number of next time steps in one shooting evaluation - 4 frame window */
            ntc_vec[3]=ntc_vec[2];
            ntc_vec[2]=ntc_vec[1];
            ntc_vec[1]=ntc_vec[0];
            ntc_vec[0]=nextTime_count-ntc_old;
            ntc_mv=(ntc_vec[0]+ntc_vec[1]+ntc_vec[2]+ntc_vec[3])*0.25;
            ntc_old=nextTime_count;
            printf("\n----------------\n");
            printf("Shooting cycle iteration number: %3d ||", shooting_cycle_counter);
            printf("NTC_MV: %g || f_proj: %g\n", ntc_mv, f_proj); /* for debugging purpose */
            printf("Print of dynamically consistent nodes voltages or branches currents:\n");

            for(i=1, node=ckt->CKTnodes->next; node; i++, node=node->next) {

                if (!strstr(node->name, "#")) {


	            double tv_01= MAX(fabs(RHS_max[i-1]), fabs(RHS_min[i-1]));

                    err_conv_ref += ((RHS_max[i-1] - RHS_min[i-1]) * 1e-3 + 1e-6) * 7 * ckt->CKTsteady_coeff;
                    if ( fabs(RHS_max[i-1] - RHS_min[i-1]) > 10*1e-6) {
                        S_diff[i-1] = (RHS_max[i-1] - RHS_min[i-1]) / tv_01 - S_old[i-1];
                        S_old[i-1]  = (RHS_max[i-1] - RHS_min[i-1]) / tv_01;
                        if(fabs(S_old[i-1]) > 0.1) printf("Node voltage   %15s: RHS diff. %-15g || Conv. ref. %-15g || RHS max. %-15g || RHS min. %-15g || Snode %-15g\n", node->name,
                                                              ckt->CKTrhsOld[i] - RHS_copy_se[i-1],
                                                              ((RHS_max[i-1] - RHS_min[i-1]) * 1e-3 + 1e-6) * 7 * ckt->CKTsteady_coeff,
                                                              RHS_max[i-1],
                                                              RHS_min[i-1],
                                                              S_diff[i-1] // (RHS_max[i-1] - RHS_min[i-1]) / (RHS_max[i-1] + RHS_min[i-1]) / 0.5
                                                             );
                        dynamic_test++; /* test on voltage dynamic consistence */
                    } else {
                        S_old[i-1]  = 0;
                        S_diff[i-1] = 0;
                    }
                } else {


	            double tv_01= MAX(fabs(RHS_max[i-1]), fabs(RHS_min[i-1]));

                    err_conv_ref += ((RHS_max[i-1] - RHS_min[i-1]) * 1e-3 + 1e-9) * 7 * ckt->CKTsteady_coeff;
                    if ( fabs(RHS_max[i-1] - RHS_min[i-1]) > 10*1e-9) {
                        S_diff[i-1] = (RHS_max[i-1] - RHS_min[i-1]) / tv_01 - S_old[i-1];
                        S_old[i-1]  = (RHS_max[i-1] - RHS_min[i-1]) / tv_01;
                        if(fabs(S_old[i-1]) > 0.1) printf("Branch current %15s: RHS diff. %-15g || Conv. ref. %-15g || RHS max. %-15g || RHS min. %-15g || Sbranch %-15g \n", node->name,
                                                              ckt->CKTrhsOld[i] - RHS_copy_se[i-1],
                                                              ((RHS_max[i-1] - RHS_min[i-1]) * 1e-3 + 1e-9) * 7 * ckt->CKTsteady_coeff,
                                                              RHS_max[i-1],
                                                              RHS_min[i-1],
                                                              S_diff[i-1] // (RHS_max[i-1] - RHS_min[i-1]) / (RHS_max[i-1] + RHS_min[i-1]) / 0.5
                                                             );
                        dynamic_test++; /* test on current dynamic consistence */
                    } else {
                        S_old[i-1]  = 0;
                        S_diff[i-1] = 0;
                    }
                }
            }
            if (dynamic_test==0) {
                /* Test for dynamic existence */
                printf("No detectable dynamic on voltages nodes or currents branches. PSS analysis aborted.\n");
                FREE(RHS_copy_se);
                FREE(RHS_max);
                FREE(RHS_min);
                FREE(S_old);
                FREE(S_diff);
                return(OK);
            }
            if ((time_err_min_0-time_temp)<0) {
                /* Something has gone wrong... */
                printf("Cannot find a minimum for error vector in estimated period. Try to adjust tstab! PSS analysis aborted.\n");
                FREE(RHS_copy_se);
                FREE(RHS_max);
                FREE(RHS_min);
                FREE(S_old);
                FREE(S_diff);
                return(OK);
            }
            printf("Global Convergence Error reference: %g.\n", err_conv_ref/dynamic_test);
            /*** FREQUENCY ESTIMATION UPDATE ***/
            if ( err_min_0==err || err_min_0==1e+30 ) {
                ckt->CKTguessedFreq=(ckt->CKTguessedFreq)+f_proj;
#ifdef STEPDEBUG
                printf("Frequency DOWN:  est per %g, err min %g, err min 1 %g, err max %g, ntc %ld, err %g, err_last %g\n", time_err_min_0-time_temp,err_min_0,err_min_1,err_max,nextTime_count,err,err_last);
#endif
                gf_last_1=gf_last_0;
                gf_last_0=ckt->CKTguessedFreq;
            } else {
                ckt->CKTguessedFreq=1/(time_err_min_0-time_temp);
#ifdef STEPDEBUG
                printf("Frequency UP:  est per %g, err min %g, err min 1 %g, err max %g, ntc %ld, err %g, err_last %g\n", time_err_min_0-time_temp,err_min_0,err_min_1,err_max,nextTime_count,err,err_last);
#endif
                gf_last_1=gf_last_0;
                gf_last_0=ckt->CKTguessedFreq;
            }
            /* Store auxiliary variable of time_temp */
            time_temp_0=time_temp;
            /* Next evaluation of shooting will be updated time (time_temp) summed to updated guessed period */
            time_temp=ckt->CKTtime;
            /* IMPORTANT! Final time must be updated! Otherwise delta time can be wrongly calculated */
            ckt->CKTfinalTime=time_temp+2/ckt->CKTguessedFreq;
            /* Set next the breakpoint */
            CKTsetBreak(ckt, time_temp+1/(ckt->CKTguessedFreq));
            /* Store error history */
            rr_history[shooting_cycle_counter]=err;
            gf_history[shooting_cycle_counter]=ckt->CKTguessedFreq;
            shooting_cycle_counter++;
            printf("Updated guessed frequency: %1.10lg .\n",ckt->CKTguessedFreq);
            printf("Next shooting evaluation time is %1.10g and current time is %1.10g.\n",time_temp+1/(ckt->CKTguessedFreq),ckt->CKTtime);
            /* shooting exit condition */
            if ( shooting_cycle_counter>ckt->CKTsc_iter || (rr_history[shooting_cycle_counter-1]<err_conv_ref/dynamic_test)) {
#ifdef STEPDEBUG
                printf("\nFrequency estimation (FE) and RHS residual (RR) evolution.\n");
#endif
                for(count_5=0; count_5<shooting_cycle_counter-1; count_5++) {
#ifdef STEPDEBUG
                    if (count_5==0) {
                        printf("%-3d -> FE: %15.10g || RR: %15.10g\n",count_5+1,1.0/delta_t/10,rr_history[count_5+1]); /* the very lucky case */
                    } else {
                        printf("%-3d -> FE: %15.10g || RR: %15.10g\n",count_5+1,gf_history[count_5],rr_history[count_5+1]);
                    }
#endif
                    /* reuse variables */
                    if (rr_history[count_5]<err_0) {
                        err_0=rr_history[count_5];
                        k=count_5;
                    }
                }
                if (shooting_cycle_counter<=ckt->CKTsc_iter) {
                    ckt->CKTguessedFreq=gf_history[shooting_cycle_counter-1];
                    printf("\nConvergence reached. Final circuit time is %1.10g s and predicted fundamental frequency is %g Hz.\n",ckt->CKTtime,ckt->CKTguessedFreq);
                    in_pss=1; /* PERIODIC STEADY STATE NOT REACHED however set the flag */
                } else {
                    ckt->CKTguessedFreq=gf_history[k-1];
                    printf("\nConvergence not reached. However the most near convergence iteration has predicted (iteration %d) a fundamental frequency of %g Hz.\n",k,ckt->CKTguessedFreq);
                    in_pss=1; /* PERIODIC STEADY STATE REACHED set the flag */
                }
                /* Allocates memory for nodes data in PSS */
                psstimes   = TMALLOC(double, ckt->CKTpsspoints);
                pssvalues  = TMALLOC(double, msize*ckt->CKTpsspoints);
                pssValues  = TMALLOC(double, ckt->CKTpsspoints);
                pssfreqs   = TMALLOC(double, ckt->CKTharms);
                pssmags    = TMALLOC(double, ckt->CKTharms);
                pssphases  = TMALLOC(double, ckt->CKTharms);
                pssnmags   = TMALLOC(double, ckt->CKTharms);
                pssnphases = TMALLOC(double, ckt->CKTharms);
                pssResults = TMALLOC(double, msize*ckt->CKTharms);
            }
            /* restore maximum and minimum error for next search */
            err_min_0=1.0e+30;
            err_max=-1.0e+30;
            err_0=1.e30;
            err_1=-1.0e+30;
            tv_03=err_conv_ref;
            err_conv_ref=0;
            tv_04=dynamic_test;
            dynamic_test=0;
            rest=shooting_cycle_counter%2;
            /* Reset actual RHS reference for next shooting evaluation */
            for(count_6 = 1; count_6 <= msize; count_6++) {
                RHS_copy_se[count_6-1] = ckt->CKTrhsOld[count_6];
            }
#ifdef STEPDEBUG
            printf("RHS on new shooting cycle: ");
            for(count_3 = 1; count_3 <= msize; count_3++) {
                printf("%-15g ", RHS_copy_se[count_3-1]);
            }
            printf("\n");
#endif
            if (in_pss!=1) {
                for(count_7 = 1; count_7 <= msize; count_7++) {
                    /* reset max and min per node or branch on every shooting cycle */
                    RHS_max[count_7-1] = -1.0e+30;
                    RHS_min[count_7-1] =  1.0e+30;
                }
            }
            printf("----------------\n\n");
        }
    } else {
        /* return on the converged shooting condition */
        if ( AlmostEqualUlps( ckt->CKTtime, time_temp+1/ckt->CKTguessedFreq, 10 ) || (ckt->CKTtime > time_temp+1/ckt->CKTguessedFreq) ) {
            /* restore time */
            ckt->CKTtime=time_temp;
            /* Set the breakpoint */
            CKTsetBreak(ckt, time_temp+1/(ckt->CKTguessedFreq));
            pss_cycle_counter++;
            if (pss_cycle_counter>1) {
                /* End plot in time domain */
                SPfrontEnd->OUTendPlot (job->PSSplot_td);
                /* The following line must be placed just before a new OUTpBeginPlot is called */
                error = CKTnames(ckt,&numNames,&nameList);
                if (error) return (error);
                SPfrontEnd->IFnewUid (ckt, &freqUid, NULL,
                                      "frequency", UID_OTHER, NULL);
                error = SPfrontEnd->OUTpBeginPlot (
                    ckt, ckt->CKTcurJob,
                    "Frequency Domain Periodic Steady State",
                    freqUid, IF_REAL,
                    numNames, nameList, IF_REAL, &(job->PSSplot_fd));
                tfree(nameList);
                /* ************************* */
                /* Fourier transform on data */
                /* ************************* */
                for(i4 = 1; i4 <= msize; i4++) {
                    for(k1=0; k1<ckt->CKTpsspoints; k1++) {
                        pssValues[k1] = pssvalues[k1*msize + (i4-1)];
                    }
                    CKTfour(ckt->CKTpsspoints, ckt->CKTharms, &thd, psstimes, pssValues,
                            ckt->CKTguessedFreq, pssfreqs, pssmags, pssphases, pssnmags,
                            pssnphases);
                    for(k1 = 1; k1 <= ckt->CKTharms; k1++) {
                        pssResults[(k1-1) + (i4-1)*msize] = pssmags[k1-1];
                    }
                }
                for(k1 = 1; k1 <= ckt->CKTharms; k1++) {
                    for(i4 = 1; i4 <= msize; i4++) {
                        ckt->CKTrhsOld[i4] =pssResults[(k1-1)+(i4-1)*msize] ;
                    }
                    CKTdump(ckt, pssfreqs[k1-1], job->PSSplot_fd);
                }
                /* End plot in freq domain */
                SPfrontEnd->OUTendPlot (job->PSSplot_fd);
                FREE(RHS_copy_se);
                FREE(RHS_max);
                FREE(RHS_min);
                FREE(S_old);
                FREE(S_diff);
                FREE(pssfreqs);
                FREE(psstimes);
                FREE(pssValues);
                FREE(pssResults);
                FREE(pssmags);
                FREE(pssphases);
                FREE(pssnmags);
                FREE(pssnphases);
                return(OK);
            }
        }
    }
    /* ********************************** */
    /* **** END SHOOTING CODE BLOCK ***** */
    /* ********************************** */

    if( SPfrontEnd->IFpauseTest() ) {
        /* user requested pause... */
        UPDATE_STATS(DOING_TRAN);
        return(E_PAUSE);
    }

    /* RESUME */
resume:
#ifdef STEPDEBUG
    if( (ckt->CKTdelta <= ckt->CKTfinalTime/50) &&
            (ckt->CKTdelta <= ckt->CKTmaxStep)) {
        ;
    } else {
        if(ckt->CKTfinalTime/50<ckt->CKTmaxStep) {
            (void)printf("limited by Tstop/50\n");
        } else {
            (void)printf("limited by Tmax == %g\n",ckt->CKTmaxStep);
        }
    }
#endif
#ifdef HAS_WINDOWS
    if (ckt->CKTtime == 0.)
        SetAnalyse( "tran init", 0);
    else if (( !in_pss ) && (shooting_cycle_counter > 0))
        SetAnalyse( "shooting", shooting_cycle_counter);
    else
        SetAnalyse( "tran", (int)((ckt->CKTtime * 1000.) / ckt->CKTfinalTime));
#endif
    ckt->CKTdelta =
        MIN(ckt->CKTdelta,ckt->CKTmaxStep);
#ifdef XSPICE
/* gtri - begin - wbk - Cut integration order if first timepoint after breakpoint */
    /* if(ckt->CKTtime == g_mif_info.breakpoint.last) */
    if ( AlmostEqualUlps( ckt->CKTtime, g_mif_info.breakpoint.last, 100 ) )
        ckt->CKTorder = 1;
/* gtri - end   - wbk - Cut integration order if first timepoint after breakpoint */

#endif

    /* are we at a breakpoint, or indistinguishably close? */
    /* if ((ckt->CKTtime == ckt->CKTbreaks[0]) || (ckt->CKTbreaks[0] - */
    if ( ckt->CKTbreaks[0] - ckt->CKTtime <= ckt->CKTdelmin ) {
        /*if ( AlmostEqualUlps( ckt->CKTtime, ckt->CKTbreaks[0], 100 ) || (ckt->CKTbreaks[0] -
         *    (ckt->CKTtime) <= ckt->CKTdelmin)) {*/
        /* first timepoint after a breakpoint - cut integration order */
        /* and limit timestep to .1 times minimum of time to next breakpoint,
         *  and previous timestep
         */
        ckt->CKTorder = 1;
        if( (ckt->CKTdelta >.1* ckt->CKTsaveDelta) ||
                (ckt->CKTdelta > .1 * (ckt->CKTbreaks[1] - ckt->CKTbreaks[0])) ) {
            if(ckt->CKTsaveDelta < (ckt->CKTbreaks[1] - ckt->CKTbreaks[0]))  {
#ifdef STEPDEBUG
                (void)printf("limited by pre-breakpoint delta (saveDelta: %1.10g, nxt_breakpt: %1.10g, curr_breakpt: %1.10g and CKTtime: %1.10g\n",
                             ckt->CKTsaveDelta, ckt->CKTbreaks[1], ckt->CKTbreaks[0], ckt->CKTtime);
#endif
            } else {
#ifdef STEPDEBUG
                (void)printf("limited by next breakpoint\n");
                (void)printf("(saveDelta: %1.10g, Delta: %1.10g, CKTtime: %1.10g and delmin: %1.10g\n",ckt->CKTsaveDelta,ckt->CKTdelta,ckt->CKTtime,ckt->CKTdelmin);
#endif
            }
        }

        if ( ckt->CKTbreaks[1] - ckt->CKTbreaks[0] == 0 )
            ckt->CKTdelta = ckt->CKTdelmin;
        else
            ckt->CKTdelta = MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta,
                                ckt->CKTbreaks[1] - ckt->CKTbreaks[0]));

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
    } else {
        /* Else, mark that timestep was not set by temporary breakpoint */
        g_mif_info.breakpoint.last = 1.0e30;
    }

/* gtri - end - wbk - Add Breakpoint stuff */

/* gtri - begin - wbk - Modify Breakpoint stuff */
    /* Throw out any permanent breakpoint times <= current time */
    for(;;) {
#ifdef STEPDEBUG
        printf("    brk_pt: %g    ckt_time: %g    ckt_min_break: %g\n", ckt->CKTbreaks[0], ckt->CKTtime, ckt->CKTminBreak);
#endif
        if(AlmostEqualUlps(ckt->CKTbreaks[0],ckt->CKTtime, 100) ||
           ckt->CKTbreaks[0] <= ckt->CKTtime + ckt->CKTminBreak) {
            printf("throwing out permanent breakpoint times <= current time (brk pt: %g)\n", ckt->CKTbreaks[0]);
            printf("ckt_time: %g    ckt_min_break: %g\n", ckt->CKTtime, ckt->CKTminBreak);
            CKTclrBreak(ckt);
        } else {
            break;
        }
    }
    /* Force the breakpoint if appropriate */
    if(ckt->CKTtime + ckt->CKTdelta > ckt->CKTbreaks[0]) {
        ckt->CKTbreak = 1;
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
    }

/* gtri - end - wbk - Modify Breakpoint stuff */
#else /* !XSPICE */

        /* don't want to get below delmin for no reason */
        ckt->CKTdelta = MAX(ckt->CKTdelta, ckt->CKTdelmin*2.0);
    }
    else if(ckt->CKTtime + ckt->CKTdelta >= ckt->CKTbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = ckt->CKTbreaks[0] - ckt->CKTtime;
        /*(void)printf("delta cut to %g to hit breakpoint\n",ckt->CKTdelta);*/
        fflush(stdout);
        ckt->CKTbreak = 1; /* why? the current pt. is not a bkpt. */
    }
#ifdef CLUSTER
    if(!CLUsync(ckt->CKTtime,&ckt->CKTdelta,0)) {
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
            if(ckt->CKTbreaks[0] < g_mif_info.breakpoint.current)
                if(ckt->CKTbreaks[0] > ckt->CKTtime + ckt->CKTminBreak)
                    g_mif_info.breakpoint.current = ckt->CKTbreaks[0];
            if(g_mif_info.breakpoint.current < ckt->CKTtime + ckt->CKTdelta) {
                /* Breakpoint must be > last accepted timepoint */
                /* and >= current event time */
                if(g_mif_info.breakpoint.current > ckt->CKTtime + ckt->CKTminBreak  &&
                   g_mif_info.breakpoint.current >= g_mif_info.circuit.evt_step) {
                    ckt->CKTsaveDelta = ckt->CKTdelta;
                    ckt->CKTdelta = g_mif_info.breakpoint.current - ckt->CKTtime;
                    g_mif_info.breakpoint.last = ckt->CKTtime + ckt->CKTdelta;
                }
            }

        } /* end while next event time <= next analog time */
    } /* end if there are event instances */

/* gtri - end - wbk - Do event solution */
#endif
    for(i5=5; i5>=0; i5--)
        ckt->CKTdeltaOld[i5+1] = ckt->CKTdeltaOld[i5];
    ckt->CKTdeltaOld[0] = ckt->CKTdelta;

    {
        double *temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
        for(i5=ckt->CKTmaxOrder; i5>=0; i5--)
            ckt->CKTstates[i5+1] = ckt->CKTstates[i5];
        ckt->CKTstates[0] = temp;
    }

    /* 600 */
    for (;;) {
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

        /* ************************************ */
        /* ********** CKTtime update ********** */
        /* ************************************ */
        /* delta manipulation */
        if (!in_stabilization && !in_pss) {
            if ( (ckt->CKTtime-(time_temp+1/ckt->CKTguessedFreq)<1/ckt->CKTguessedFreq/10) && (ckt->CKTbreak==0) ) {
                if ( !(flag_tu_1) ) flag_tu_2=ckt->CKTdelta; /* store previous delta */
                if ( (ckt->CKTtime-(time_temp+1/ckt->CKTguessedFreq)<1/ckt->CKTguessedFreq/1.0e5) && (ckt->CKTbreak==0) ) {
                    if ( (ckt->CKTtime-(time_temp+1/ckt->CKTguessedFreq)<1/ckt->CKTguessedFreq/1.0e7) && (ckt->CKTbreak==0) ) {
                        if (rr_history[shooting_cycle_counter-1]<tv_03/tv_04*100) {
                            ckt->CKTdelta=1/ckt->CKTguessedFreq/1.0e5; /* get closer to accurate solution? */
                        } else {
                            ckt->CKTdelta=1/ckt->CKTguessedFreq/1.0e4;
                        }
                    } else {
                        ckt->CKTdelta=1/ckt->CKTguessedFreq/1.0e1;
                    }
                } else {
                    ckt->CKTdelta=1/ckt->CKTguessedFreq/0.25e1;
                }
                flag_tu_1=1;
            } else {
                if (flag_tu_1) {
                    ckt->CKTdelta=flag_tu_2; /* restore prevoius delta */
                    flag_tu_1=0;
                }
            }
        }
        if ( in_pss ) ckt->CKTdelta=1/ckt->CKTguessedFreq/((ckt->CKTpsspoints-1)); /* fixed delta in PSS */
        /* ************************************ */
        /* ******* END CKTtime update ********* */
        /* ************************************ */

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
        if(ckt->evt->counts.num_insts > 0) {
            g_mif_info.circuit.evt_step = ckt->CKTtime;
            EVTcall_hybrids(ckt);
        }
/* gtri - end - wbk - Call all hybrids */

#endif
        ckt->CKTstat->STATtimePts ++;
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        if(firsttime) {
            for(i=0; i<ckt->CKTnumStates; i++) {
                ckt->CKTstate2[i] = ckt->CKTstate1[i];
                ckt->CKTstate3[i] = ckt->CKTstate1[i];
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
            /*printf("delta cut to %g for non-convergance\n",ckt->CKTdelta);*/
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
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)) {
                    save1 = ckt->CKTmode;
                    save2 = ckt->CKTorder;
                    ckt->CKTmode = save_mode;
                    ckt->CKTorder = save_order;
                    if(error = CKTsenDCtran(ckt)) return(error);
                    ckt->CKTmode = save1;
                    ckt->CKTorder = save2;
                }
#endif
                firsttime=0;
#ifndef CLUSTER
                goto nextTime;  /* no check on
		* first time point
		*/
#else
                redostep = 0;
                goto chkStep;
#endif
            }
            newdelta = ckt->CKTdelta;
            error = CKTtrunc(ckt,&newdelta);
            if(error) {
                UPDATE_STATS(DOING_TRAN);
                return(error);
            }
            if(newdelta>.9 * ckt->CKTdelta) {
                if(ckt->CKTorder == 1) {
                    newdelta = ckt->CKTdelta;
                    ckt->CKTorder = 2;
                    error = CKTtrunc(ckt,&newdelta);
                    if(error) {
                        UPDATE_STATS(DOING_TRAN);
                        return(error);
                    }
                    if(newdelta <= 1.05 * ckt->CKTdelta) {
                        ckt->CKTorder = 1;
                    }
                }
                /* time point OK  - 630*/
                ckt->CKTdelta = newdelta;
#ifdef	NDEV
                /* show a time process indicator, by Gong Ding, gdiso@ustc.edu */
                if(ckt->CKTtime/ckt->CKTfinalTime*100<10.0)
                    printf("%%%3.2lf\b\b\b\b\b",ckt->CKTtime/ckt->CKTfinalTime*100);
                else   if(ckt->CKTtime/ckt->CKTfinalTime*100<100.0)
                    printf("%%%4.2lf\b\b\b\b\b\b",ckt->CKTtime/ckt->CKTfinalTime*100);
                else
                    printf("%%%5.2lf\b\b\b\b\b\b\b",ckt->CKTtime/ckt->CKTfinalTime*100);
                fflush(stdout);
#endif

#ifdef STEPDEBUG
                (void)printf(
                    "delta set to truncation error result: %g. Point accepted at CKTtime: %g\n",
                    ckt->CKTdelta,ckt->CKTtime);
                fflush(stdout);
#endif

#ifdef WANT_SENSE2
                if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN)) {
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
                ckt->CKTdelta = newdelta;
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
                /*#ifdef STEPDEBUG*/
                (void)printf("delta at delmin\n");
                /*#endif*/
            } else {
                UPDATE_STATS(DOING_TRAN);
                errMsg = CKTtrouble(ckt, "Timestep too small");
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
        if(CLUsync(ckt->CKTtime,&ckt->CKTdelta,redostep)) {
            goto nextTime;
        } else {
            ckt->CKTtime -= olddelta;
            ckt->CKTstat->STATrejected ++;
        }
#endif
    }
    /* NOTREACHED */
}

int
CKTfour(int ndata,		/* number of entries in the Time and
                                   Value arrays */
        int numFreq,		/* number of harmonics to calculate */
        double *thd,		/* total harmonic distortion (percent)
                                   to be returned */
        double *Time,		/* times at which the voltage/current
                                   values were measured*/
        double *Value,		/* voltage or current vector whose
                                   transform is desired */
        double FundFreq,	/* the fundamental frequency of the
                                   analysis */
        double *Freq,		/* the frequency value of the various
                                   harmonics */
        double *Mag,		/* the Magnitude of the fourier
                                   transform */
        double *Phase,		/* the Phase of the fourier transform */
        double *nMag,		/* the normalized magnitude of the
                                   transform: nMag(fund)=1*/
        double *nPhase)		/* the normalized phase of the
                                   transform: Nphase(fund)=0 */
{
    /* Note: we can consider these as a set of arrays.  The sizes are:
     * Time[ndata], Value[ndata], Freq[numFreq], Mag[numfreq],
     * Phase[numfreq], nMag[numfreq], nPhase[numfreq]
     *
     * The arrays must all be allocated by the caller.
     * The Time and Value array must be reasonably distributed over at
     * least one full period of the fundamental Frequency for the
     * fourier transform to be useful.  The function will take the
     * last period of the frequency as data for the transform.
     *
     * We are assuming that the caller has provided exactly one period
     * of the fundamental frequency.  */
    int i;
    int j;
    double tmp;

    NG_IGNORE(Time);

    /* clear output/computation arrays */

    for(i=0; i<numFreq; i++) {
        Mag[i]=0;
        Phase[i]=0;
    }
    for(i=0; i<ndata; i++) {
        for(j=0; j<numFreq; j++) {
            Mag[j]   += (Value[i]*sin(j*2.0*M_PI*i/((double) ndata)));
            Phase[j] += (Value[i]*cos(j*2.0*M_PI*i/((double) ndata)));
        }
    }

    Mag[0] = Phase[0]/ndata;
    Phase[0]=nMag[0]=nPhase[0]=Freq[0]=0;
    *thd = 0;
    for(i=1; i<numFreq; i++) {
        tmp = Mag[i]*2.0 /ndata;
        Phase[i] *= 2.0/ndata;
        Freq[i] = i * FundFreq;
        Mag[i] = sqrt(tmp*tmp+Phase[i]*Phase[i]);
        Phase[i] = atan2(Phase[i],tmp)*180.0/M_PI;
        nMag[i] = Mag[i]/Mag[1];
        nPhase[i] = Phase[i]-Phase[1];
        if(i>1) *thd += nMag[i]*nMag[i];
    }
    *thd = 100*sqrt(*thd);
    return(OK);
}
