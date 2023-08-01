/**********
Copyright 2019 Holger Vogt  All rights reserved.
Author: 2019 Holger Vogt
Modified BSD license
**********/

/* subroutine to generate OP by TRANSIENT analysis */

#include <errno.h>

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "cktaccept.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/missing_math.h"
#include "com_optran.h"


/* for setting breakpoints required by dbs data base */
extern struct dbcomm *dbs;
#include "ngspice/ftedebug.h"

#ifdef XSPICE
/* gtri - add - wbk - Add headers */
#include "ngspice/miftypes.h"

#include "ngspice/evt.h"
#include "ngspice/enh.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk - Add headers */
#endif

#ifdef SHARED_MODULE
extern int add_bkpt(void);
extern int sharedsync(double*, double*, double, double, double, int, int*, int);
extern int ng_ident;      /* for debugging */
#endif

int OPclrBreak(CKTcircuit* ckt);
int OPsetBreak(CKTcircuit* ckt, double time);

static double *opbreaks;
static int OPbreakSize;
static double opfinaltime = 1e-6;
static double opstepsize = 1e-8;
static double opramptime = 0.;
static bool nooptran = TRUE;

/* command to set the 6 optran flags
    CKTnoOpIter (default 0, set by 'option noopiter')
    CKTnumGminSteps
    CKTnumSrcSteps
    opstepsize
    opfinaltime
    opramptime

    A typical command may be
    optran 0 0 0 50u 10m 0
    (no initial iteration, no gmin stepping, no source stepping,
    stepsize for optran 50 us, optran run until 10 ms, no supply ramping.

    If com_optran is given in .spiceinit, the circuit is not yet loaded. So
    we firstly fill the static vars opstepsize, opfinaltime,
    and opramptime. Later from inp.c we call com_optran again and set the
    data in ft_curckt->ci_defTask.
    */
void com_optran(wordlist* wl) {
    wordlist* wltmp = wl;
    char* stpstr;
    int err;
    int optrancom;
    static bool dataset = FALSE;
    static bool getdata = FALSE;
    static bool opiter = TRUE;
    static int ngminsteps = 1;
    static int nsrcsteps = 1;

    /* current circuit */
    if (ft_curckt && dataset && wl == NULL) { /* called from inp.c */
        ft_curckt->ci_defTask->TSKnoOpIter = (opiter != 0);
        ft_curckt->ci_defTask->TSKnumGminSteps = ngminsteps;
        ft_curckt->ci_defTask->TSKnumSrcSteps = nsrcsteps;
        getdata = FALSE; /* allow more calls to 'optran' */
        return;
    }
    else if (!ft_curckt && !dataset && wl == NULL) {
        fprintf(stderr, "Error: syntax error with command 'optran'!\n");
        fprintf(stderr, "    Command ingnored\n");
        return;
    }
    else if (ft_curckt && !dataset && wl == NULL) {
        /* no optran wanted */
        return;
    }
    else if (!ft_curckt && !dataset) { /* called from .spiceinit */
        getdata = TRUE;
    }

    if (!getdata && !ft_curckt) {
        /* no circuit, but optran already set */
        return;
    }

    int saved = errno;
    errno = 0;
    nooptran = FALSE;
    /* wordlist with 6 parameters */
    optrancom = (int)strtol(wltmp->wl_word, &stpstr, 10);
    if ((errno == ERANGE) || (*stpstr != '\0'))
        goto bugquit;
    if (optrancom == 0) {
        if (getdata) {
            opiter = TRUE;
        }
        else {
            ft_curckt->ci_defTask->TSKnoOpIter = 1;
        }
    }
    else {
        if (getdata) {
            opiter = FALSE;
        }
        else {
            ft_curckt->ci_defTask->TSKnoOpIter = 0;
        }
    }
    wltmp = wltmp->wl_next;
    optrancom = (int)strtol(wltmp->wl_word, &stpstr, 10);
    if ((errno == ERANGE) || (*stpstr != '\0'))
        goto bugquit;
    if (getdata) {
        ngminsteps = optrancom;
    }
    else {
        ft_curckt->ci_defTask->TSKnumGminSteps = optrancom;
    }
    wltmp = wltmp->wl_next;
    optrancom = (int)strtol(wltmp->wl_word, &stpstr, 10);
    if ((errno == ERANGE) || (*stpstr != '\0'))
        goto bugquit;
    if (getdata) {
        nsrcsteps = optrancom;
    }
    else {
        ft_curckt->ci_defTask->TSKnumSrcSteps = optrancom;
    }
    wltmp = wltmp->wl_next;
    stpstr = wltmp->wl_word;
    opstepsize = INPevaluate(&stpstr, &err, 1);
    if (err || (*stpstr != '\0'))
        goto bugquit;
    wltmp = wltmp->wl_next;
    stpstr = wltmp->wl_word;
    opfinaltime = INPevaluate(&stpstr, &err, 1);
    if (err || (*stpstr != '\0'))
        goto bugquit;
    wltmp = wltmp->wl_next;
    stpstr = wltmp->wl_word;
    opramptime = INPevaluate(&stpstr, &err, 1);
    if (err || (*stpstr != '\0'))
        goto bugquit;
    if (opstepsize > opfinaltime) {
        fprintf(stderr, "Error: Optran step size larger than final time.\n");
        goto bugquit;
    }
    if (opstepsize > opfinaltime/50.) {
        fprintf(stderr, "Warning: Optran step size potentially too large.\n");
    }
    if (opramptime > opfinaltime) {
        fprintf(stderr, "Error: Optran ramp time larger than final time.\n");
        goto bugquit;
    }
    /* optran deselected by setting opstepsize to 0 */
    if (opstepsize == 0)
        nooptran = TRUE;

    dataset = TRUE;
    if (errno == 0)
        errno = saved;
    return;
bugquit:
    fprintf(stderr, "Error in command 'optran'\n");
}

int OPclrBreak(CKTcircuit *ckt)
{
    double *tmp;
    int j;

    NG_IGNORE(ckt);

    if (OPbreakSize > 2) {
        tmp = TMALLOC(double, OPbreakSize - 1);
        if (tmp == NULL)
            return (E_NOMEM);
        for (j = 1; j < OPbreakSize; j++) {
            tmp[j - 1] = opbreaks[j];
        }
        FREE(opbreaks);
        OPbreakSize--;
        opbreaks = tmp;
    }
    else {
        opbreaks[0] = opbreaks[1];
        opbreaks[1] = opfinaltime;
    }
    return (OK);
}


int OPsetBreak(CKTcircuit *ckt, double time)
{
    double *tmp;
    int i, j;


    for (i = 0; i < OPbreakSize; i++) {
        if (opbreaks[i] > time) { /* passed */
            if ((opbreaks[i] - time) <= ckt->CKTminBreak) {
                /* very close together - take earlier point */
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e replaces %e\n", ckt->CKTtime, time,
                        ckt->CKTbreaks[i]);
                CKTbreakDump(ckt);
#endif
                opbreaks[i] = time;
                return (OK);
            }
            if (i > 0 && time - opbreaks[i - 1] <= ckt->CKTminBreak) {
                /* very close together, but after, so skip */
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e skipped\n", ckt->CKTtime, time);
                CKTbreakDump(ckt);
#endif
                return (OK);
            }
            /* fits in middle - new array & insert */
            tmp = TMALLOC(double, OPbreakSize + 1);
            if (tmp == NULL)
                return (E_NOMEM);
            for (j = 0; j < i; j++) {
                tmp[j] = opbreaks[j];
            }
            tmp[i] = time;
#ifdef TRACE_BREAKPOINT
            printf("[t:%e] \t %e added\n", ckt->CKTtime, time);
            CKTbreakDump(ckt);
#endif
            for (j = i; j < OPbreakSize; j++) {
                tmp[j + 1] = opbreaks[j];
            }
            FREE(opbreaks);
            OPbreakSize++;
            opbreaks = tmp;

            return (OK);
        }
    }
    /* never found it - beyond end of time - extend out idea of time */
    if (time - opbreaks[OPbreakSize - 1] <= ckt->CKTminBreak) {
        /* very close tegether - keep earlier, throw out new point */
#ifdef TRACE_BREAKPOINT
        printf("[t:%e] \t %e skipped (at the end)\n", ckt->CKTtime, time);
        CKTbreakDump(ckt);
#endif
        return (OK);
    }
    /* fits at end - grow array & add on */
    opbreaks = TREALLOC(double, opbreaks, OPbreakSize + 1);
    OPbreakSize++;
    opbreaks[OPbreakSize - 1] = time;
#ifdef TRACE_BREAKPOINT
    printf("[t:%e] \t %e added at end\n", ckt->CKTtime, time);
    CKTbreakDump(ckt);
#endif
    return (OK);
}


/* Do a simple transient simulation, starting with time 0 until opfinaltime.
   No output vectors are generated, actual times and breakpoints are kept local.
   When returning, the matrix is left in its current state.
   Thus this algorithm creates an operating point to start other simulations.
   The code is derived from dctran.c by removing all un-needed parts.*/
int
OPtran(CKTcircuit *ckt, int oldconverged)
{
    int i;
    double olddelta;
    double delta;
    double newdelta;
    double *temp;

    double optime = 0;

    int converged;
    int firsttime;
    int error;

    double maxstepsize = 0.0, prevmaxstepsize = 0.0, prevstepsize=0.;

    int ltra_num;

#if defined SHARED_MODULE
    int redostep;
#endif

    /* if optran command has not been given (in .spiceinit or in .control section),
       we don' use optran */
    if (nooptran)
        return oldconverged;
/*
    ACAN *acjob = (ACAN *) ckt->CKTcurJob;
    TRANan *trjob = (TRANan *) ckt->CKTcurJob;
    NOISEAN *nojob = (NOISEAN *) ckt->CKTcurJob;
*/
    if(optime == 0) {
#ifdef HAS_PROGREP
        SetAnalyse("optran init", 0);
#endif
        SPfrontEnd->IFerrorf(ERR_INFO, "Transient op started");
        if (opramptime > 0) {
            CKTnode* n;
            ckt->CKTsrcFact = 0.;
            SPfrontEnd->IFerrorf(ERR_INFO, "Ramptime enabled");
            for (n = ckt->CKTnodes; n; n = n->next)
                ckt->CKTrhsOld[n->number] = 0;

            for (i = 0; i < ckt->CKTnumStates; i++)
                ckt->CKTstate0[i] = 0;

            /*  First, try a straight solution with all sources at zero */
            converged = NIiter(ckt, ckt->CKTdcTrcvMaxIter);
        }
#if 0
        /* Set the final time */
        /* If we are in transient simulation */
        if (type == 4 && ckt->CKTstep > 0.)
            opfinaltime = 100. * ckt->CKTstep;
        /* We are in AC mode */
        else if (type == 1 && acjob->ACstartFreq > 0) {
            opfinaltime = 0.1 / acjob->ACstartFreq;
            ckt->CKTstep = opfinaltime / 1000.;
        }
        /* Noise analysis */
        else if (type == 8 && nojob->NstartFreq > 0) {
            opfinaltime = 0.1 / nojob->NstartFreq;
            ckt->CKTstep = opfinaltime / 1000.;
        }
#else
        prevmaxstepsize = ckt->CKTmaxStep;
        prevstepsize = ckt->CKTstep;
        ckt->CKTmaxStep = ckt->CKTstep = opstepsize;
#endif
        delta=MIN(opfinaltime/100,ckt->CKTstep)/10;

        /* begin LTRA code addition */
        if (ckt->CKTtimePoints != NULL)
            FREE(ckt->CKTtimePoints);

        if (ckt->CKTstep >= ckt->CKTmaxStep)
            maxstepsize = ckt->CKTstep;
        else
            maxstepsize = ckt->CKTmaxStep;

        ckt->CKTsizeIncr = 100;
        ckt->CKTtimeIndex = -1; /* before the DC soln has been stored */
        ckt->CKTtimeListSize = (int) ceil( opfinaltime / maxstepsize );
        ltra_num = CKTtypelook("LTRA");
        if (ltra_num >= 0 && ckt->CKThead[ltra_num] != NULL)
            ckt->CKTtimePoints = TMALLOC(double, ckt->CKTtimeListSize);
        /* end LTRA code addition */

        opbreaks = TMALLOC(double, 2);
        if(opbreaks == NULL) return(E_NOMEM);
        opbreaks[0] = 0;
        opbreaks[1] = opfinaltime;
        OPbreakSize = 2;

#ifdef SHARED_MODULE
        add_bkpt();
#endif

        firsttime = 1;

#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Initialize the temporary breakpoint variables to infinity */
        g_mif_info.breakpoint.current = 1.0e30;
        g_mif_info.breakpoint.last    = 1.0e30;

/* gtri - end - wbk - Add Breakpoint stuff */
#endif
        ckt->CKTorder = 1;
        for(i=0;i<7;i++) {
            ckt->CKTdeltaOld[i]=ckt->CKTmaxStep;
        }
        ckt->CKTdelta = delta;
#ifdef STEPDEBUG
        (void)printf("delta initialized to %g\n",ckt->CKTdelta);
#endif
        ckt->CKTsaveDelta = opfinaltime/50;

        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
        /* modeinittran set here */
        ckt->CKTag[0]=ckt->CKTag[1]=0;
        memcpy(ckt->CKTstate1, ckt->CKTstate0,
              (size_t) ckt->CKTnumStates * sizeof(double));

    } else {
        /* saj As traninit resets CKTmode */
        ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITPRED;
        if(ckt->CKTminBreak==0) ckt->CKTminBreak=ckt->CKTmaxStep*5e-5;
        firsttime=0;
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
            need = (int) ceil( (opfinaltime - optime) / maxstepsize );
            if (need < ckt->CKTsizeIncr)
                need = ckt->CKTsizeIncr;
            ckt->CKTtimeListSize += need;
            ckt->CKTtimePoints = TREALLOC(double, ckt->CKTtimePoints, ckt->CKTtimeListSize);
            ckt->CKTsizeIncr = (int) ceil(1.4 * ckt->CKTsizeIncr);
        }
        ckt->CKTtimePoints[ckt->CKTtimeIndex] = optime;
    }
    /* end LTRA code addition */

    error = CKTaccept(ckt);
    /* check if current breakpoint is outdated; if so, clear */
    if (optime > opbreaks[0])
        OPclrBreak(ckt);

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

    ckt->CKTbreak = 0;
    /* XXX Error will cause single process to bail. */
    if(error)  {
        tfree(opbreaks);
        return(error);
    }
#ifdef XSPICE
/* gtri - begin - wbk - Update event queues/data for accepted timepoint */
    /* Note: this must be done AFTER sending results to SI so it can't */
    /* go next to CKTaccept() above */
    if(ckt->evt->counts.num_insts > 0)
        EVTaccept(ckt, optime);
/* gtri - end - wbk - Update event queues/data for accepted timepoint */
#endif

    /* We are finished */
    if(AlmostEqualUlps( optime, opfinaltime, 100 ) ) {
        tfree(opbreaks);
        SPfrontEnd->IFerrorf(ERR_INFO, "Transient op finished successfully");
        ckt->CKTmaxStep = prevmaxstepsize;
        ckt->CKTstep = prevstepsize;
        return(OK);
    }

resume:
#ifdef HAS_PROGREP
    if (optime > 0)
        SetAnalyse( "optran", (int)((optime * 1000.) / opfinaltime + 0.5));
#endif

    ckt->CKTdelta =
            MIN(ckt->CKTdelta,ckt->CKTmaxStep);
#ifdef XSPICE
/* gtri - begin - wbk - Cut integration order if first timepoint after breakpoint */
    /* if(optime == g_mif_info.breakpoint.last) */
    if ( AlmostEqualUlps( optime, g_mif_info.breakpoint.last, 100 ) )
        ckt->CKTorder = 1;
/* gtri - end   - wbk - Cut integration order if first timepoint after breakpoint */

#endif

  /* are we at a breakpoint, or indistinguishably close? */
    /* if ((optime == opbreaks[0]) || (opbreaks[0] - */
    if ( AlmostEqualUlps( optime, opbreaks[0], 100 ) ||
         opbreaks[0] - optime <= ckt->CKTdelmin) {
        /* first timepoint after a breakpoint - cut integration order */
        /* and limit timestep to .1 times minimum of time to next breakpoint,
         * and previous timestep
         */
        ckt->CKTorder = 1;

        ckt->CKTdelta = MIN(ckt->CKTdelta, .1 * MIN(ckt->CKTsaveDelta,
            opbreaks[1] - opbreaks[0]));

        if(firsttime) {
            /* set a breakpoint to reduce ringing of current in devices */
            if (ckt->CKTmode & MODEUIC)
                OPsetBreak(ckt, ckt->CKTstep);

            ckt->CKTdelta /= 10;
#ifdef STEPDEBUG
            (void)printf("delta cut for initial timepoint\n");
#endif
        }

#ifndef XSPICE
        /* don't want to get below delmin for no reason */
        ckt->CKTdelta = MAX(ckt->CKTdelta, ckt->CKTdelmin*2.0);
#endif

    }

#ifndef XSPICE
    else if(optime + ckt->CKTdelta >= opbreaks[0]) {
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = opbreaks[0] - optime;
        ckt->CKTbreak = 1; /* why? the current pt. is not a bkpt. */
    }
#endif /* !XSPICE */


#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

    if(optime + ckt->CKTdelta >= g_mif_info.breakpoint.current) {
        /* If next time > temporary breakpoint, force it to the breakpoint */
        /* And mark that timestep was set by temporary breakpoint */
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = g_mif_info.breakpoint.current - optime;
        g_mif_info.breakpoint.last = optime + ckt->CKTdelta;
    } else {
        /* Else, mark that timestep was not set by temporary breakpoint */
        g_mif_info.breakpoint.last = 1.0e30;
    }

/* gtri - end - wbk - Add Breakpoint stuff */

/* gtri - begin - wbk - Modify Breakpoint stuff */
    /* Throw out any permanent breakpoint times <= current time */
    for (;;) {
#ifdef STEPDEBUG
        printf("    brk_pt: %g    ckt_time: %g    ckt_min_break: %g\n",opbreaks[0], optime, ckt->CKTminBreak);
#endif
        if(AlmostEqualUlps(opbreaks[0], optime, 100) ||
           opbreaks[0] <= optime + ckt->CKTminBreak) {
#ifdef STEPDEBUG
            printf("throwing out permanent breakpoint times <= current time (brk pt: %g)\n",opbreaks[0]);
            printf("    ckt_time: %g    ckt_min_break: %g\n",optime, ckt->CKTminBreak);
#endif
            OPclrBreak(ckt);
        } else {
            break;
        }
    }
    /* Force the breakpoint if appropriate */
    if(optime + ckt->CKTdelta > opbreaks[0]) {
        ckt->CKTbreak = 1;
        ckt->CKTsaveDelta = ckt->CKTdelta;
        ckt->CKTdelta = opbreaks[0] - optime;
    }

/* gtri - end - wbk - Modify Breakpoint stuff */

#ifdef SHARED_MODULE
        /* Either directly go to next time step, or modify ckt->CKTdelta depending on
           synchronization requirements. sharedsync() returns 0. */
    sharedsync(&optime, &ckt->CKTdelta, 0, opfinaltime,
        ckt->CKTdelmin, 0, &ckt->CKTstat->STATrejected, 0);
#endif

/* gtri - begin - wbk - Do event solution */

    if(ckt->evt->counts.num_insts > 0) {

        /* if time = 0 and op_alternate was specified as false during */
        /* dcop analysis, call any changed instances to let them */
        /* post their outputs with their associated delays */
        if((optime == 0.0) && (! ckt->evt->options.op_alternate))
            EVTiter(ckt);

        /* while there are events on the queue with event time <= next */
        /* projected analog time, process them */
        while((g_mif_info.circuit.evt_step = EVTnext_time(ckt))
               <= (optime + ckt->CKTdelta)) {

            /* Initialize temp analog bkpt to infinity */
            g_mif_info.breakpoint.current = 1e30;

            /* Pull items off queue and process them */
            EVTdequeue(ckt, g_mif_info.circuit.evt_step);
            EVTiter(ckt);

            /* If any instances have forced an earlier */
            /* next analog time, cut the delta */
            if(opbreaks[0] < g_mif_info.breakpoint.current)
                if(opbreaks[0] > optime + ckt->CKTminBreak)
                    g_mif_info.breakpoint.current = opbreaks[0];
            if(g_mif_info.breakpoint.current < optime + ckt->CKTdelta) {
                /* Breakpoint must be > last accepted timepoint */
                /* and >= current event time */
                if(g_mif_info.breakpoint.current >  optime + ckt->CKTminBreak  &&
                   g_mif_info.breakpoint.current >= g_mif_info.circuit.evt_step) {
                    ckt->CKTsaveDelta = ckt->CKTdelta;
                    ckt->CKTdelta = g_mif_info.breakpoint.current - optime;
                    g_mif_info.breakpoint.last = optime + ckt->CKTdelta;
                }
            }

        } /* end while next event time <= next analog time */
    } /* end if there are event instances */

/* gtri - end - wbk - Do event solution */
#else

#ifdef SHARED_MODULE
    /* Either directly go to next time step, or modify ckt->CKTdelta depending on
       synchronization requirements. sharedsync() returns 0.
    */
    sharedsync(&optime, &ckt->CKTdelta, 0, opfinaltime,
        ckt->CKTdelmin, 0, &ckt->CKTstat->STATrejected, 0);
#endif

#endif
    for(i=5; i>=0; i--)
        ckt->CKTdeltaOld[i+1] = ckt->CKTdeltaOld[i];
    ckt->CKTdeltaOld[0] = ckt->CKTdelta;

    temp = ckt->CKTstates[ckt->CKTmaxOrder+1];
    for(i=ckt->CKTmaxOrder;i>=0;i--) {
        ckt->CKTstates[i+1] = ckt->CKTstates[i];
    }
    ckt->CKTstates[0] = temp;

    for (;;) {
#if defined SHARED_MODULE
        redostep = 1;
#endif

        olddelta=ckt->CKTdelta;
        /* time abort? */
        optime += ckt->CKTdelta;

        /* supply ramping, when opramptime > 0 */
        if (opramptime > 0)
            ckt->CKTsrcFact = 0.5 * (1 - cos(M_PI * optime / opramptime));

        ckt->CKTdeltaOld[0]=ckt->CKTdelta;
        NIcomCof(ckt);

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
            g_mif_info.circuit.evt_step = optime;
        }
/* gtri - end - wbk - Set evt_step */
#endif

        converged = NIiter(ckt,ckt->CKTtranMaxIter);

#ifdef XSPICE
        if(ckt->evt->counts.num_insts > 0) {
            g_mif_info.circuit.evt_step = optime;
            EVTcall_hybrids(ckt);
        }
/* gtri - end - wbk - Call all hybrids */

#endif
        ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITPRED;
        if(firsttime) {
            memcpy(ckt->CKTstate2, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
            memcpy(ckt->CKTstate3, ckt->CKTstate1,
                   (size_t) ckt->CKTnumStates * sizeof(double));
        }
        /* txl, cpl addition */
        if (converged == 1111) {
            tfree(opbreaks);
            return(converged);
        }

        if(converged != 0) {

#ifndef SHARED_MODULE
            optime = optime -ckt->CKTdelta;
#else
            redostep = 1;
#endif

            ckt->CKTdelta = ckt->CKTdelta/8;

            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC) | MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

#ifdef XSPICE
/* gtri - begin - wbk - Add Breakpoint stuff */

        /* Force backup if temporary breakpoint is < current time */
        } else if(g_mif_info.breakpoint.current < optime) {
            ckt->CKTsaveDelta = ckt->CKTdelta;
            optime -= ckt->CKTdelta;
            ckt->CKTdelta = g_mif_info.breakpoint.current - optime;
            g_mif_info.breakpoint.last = optime + ckt->CKTdelta;

            if(firsttime) {
                ckt->CKTmode = (ckt->CKTmode&MODEUIC)|MODETRAN | MODEINITTRAN;
            }
            ckt->CKTorder = 1;

/* gtri - end - wbk - Add Breakpoint stuff */
#endif

        } else {
            if (firsttime) {
                firsttime = 0;
#if !defined SHARED_MODULE
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
                tfree(opbreaks);
                return(error);
            }
            if (newdelta > .9 * ckt->CKTdelta) {
                if ((ckt->CKTorder == 1) && (ckt->CKTmaxOrder > 1)) { /* don't rise the order for backward Euler */
                    newdelta = ckt->CKTdelta;
                    ckt->CKTorder = 2;
                    error = CKTtrunc(ckt, &newdelta);
                    if (error) {
                        tfree(opbreaks);
                        return(error);
                    }
                    if (newdelta <= 1.05 * ckt->CKTdelta) {
                        ckt->CKTorder = 1;
                    }
                }
                /* time point OK  - 630 */
                ckt->CKTdelta = newdelta;

#if !defined SHARED_MODULE
                /* go to 650 - trapezoidal */
                goto nextTime;
#else
                redostep = 0;
                goto chkStep;
#endif
            } else {

#ifndef SHARED_MODULE
                optime = optime -ckt->CKTdelta;
                ckt->CKTstat->STATrejected ++;
#else
                redostep = 1;
#endif

                ckt->CKTdelta = newdelta;
            }
        }

        if (ckt->CKTdelta <= ckt->CKTdelmin) {
            if (olddelta > ckt->CKTdelmin) {
                ckt->CKTdelta = ckt->CKTdelmin;
            } else {
                errMsg = CKTtrouble(ckt, "Timestep too small");
                tfree(opbreaks);
                return(E_TIMESTEP);
            }
        }
#ifdef XSPICE
/* gtri - begin - wbk - Do event backup */

        if(ckt->evt->counts.num_insts > 0)
            EVTbackup(ckt, optime + ckt->CKTdelta);

/* gtri - end - wbk - Do event backup */
#endif

#ifdef SHARED_MODULE
        /* redostep == 0:
           Either directly go to next time step, or modify ckt->CKTdelta depending on
           synchronization requirements. sharedsync() returns 0.
           redostep == 1:
           No convergence, or too large truncation error.
           Redo the last time step by subtracting olddelta, and modify ckt->CKTdelta
           depending on synchronization requirements. sharedsync() returns 1.
           User-supplied redo request:
           sharedsync() may return 1 if the user has decided to do so in the callback
           function.
        */
chkStep:
        if(sharedsync(&optime, &ckt->CKTdelta, olddelta, opfinaltime,
                 ckt->CKTdelmin, redostep, &ckt->CKTstat->STATrejected, 1) == 0)
            goto nextTime;
#endif

    }
    /* NOTREACHED */
}
