/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

/* Do a run of the circuit, of the given type. Type "resume" is special --
 * it means to resume whatever simulation that was in progress. The
 * return value of this routine is 0 if the exit was ok, and 1 if there was
 * a reason to interrupt the circuit (interrupt typed at the keyboard,
 * error in the simulation, etc). args should be the entire command line,
 * e.g. "tran 1 10 20 uic"
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteinp.h"
#include "ngspice/sim.h"
#include "ngspice/devdefs.h"
#include "ngspice/inpdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ifsim.h"

#include "circuits.h"
#include "shyu.h"


int
if_sens_run(CKTcircuit *ckt, wordlist *args, INPtables *tab)
{
    JOB *senseJob;
    JOB *opJob;
    struct card *current;
    IFvalue ptemp;
    IFvalue *parm;
    char buf[BSIZE_SP];
    int err;
    char *token;
    char *steptype;
    char *name;
    char *line;
    struct card deck;
    int error;
    int save;
    int flag = 0;
    int which = -1;

    (void) sprintf(buf, ".%s", wl_flatten(args));

    deck.nextcard     = NULL;
    deck.actualLine   = NULL;
    deck.error        = NULL;
    deck.linenum      = 0;
    deck.linenum_orig = 0;
    deck.line         = buf;

    current = &deck;
    line = current->line;
    INPgetTok(&line, &token, 1);

    if (ft_curckt->ci_specTask) {
        err = ft_sim->deleteTask (ft_curckt->ci_ckt, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "deleteTask");
            return (0); /* temporary */
        }
    }
    err = ft_sim->newTask (ft_curckt->ci_ckt, &(ft_curckt->ci_specTask),
                           "special", & (ft_curckt->ci_defTask));
    if (err) {
        ft_sperror(err, "newTask");
        return (0); /* temporary */
    }

    which = ft_find_analysis("options");

    if (which == -1) {
        /* in DEEP trouble */
        ft_sperror(err, "in DEEP trouble");
        return (0); /* temporary */
    }

    err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "options",
                               & (ft_curckt->ci_specOpt), ft_curckt->ci_specTask);
    if (err) {
        ft_sperror(err, "createOptions");
        return (0); /* temporary */
    }

    ft_curckt->ci_curOpt  = ft_curckt->ci_specOpt;
    ft_curckt->ci_curTask = ft_curckt->ci_specTask;

    which = ft_find_analysis("SEN");

    if (which == -1) {
        current->error = INPerrCat(
            current->error,
            INPmkTemp("sensetivity analysis unsupported\n"));
        return (0);
    }

    err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "sense",
                               & senseJob, ft_curckt->ci_specTask);
    if (err) {
        ft_sperror(err, "createSense");
        return (0); /* temporary */
    }

    save = which;

    INPgetTok(&line, &token, 1);

    if (strcmp(token, "ac") == 0) {
        JOB *acJob;
        which = ft_find_analysis("AC");
        if (which == -1) {
            current->error = INPerrCat
                (current->error,
                 INPmkTemp("ac analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "acan",
                                   & acJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createAC"); /* or similar error message */
            return (0); /* temporary */
        }

        INPgetTok(&line, &steptype, 1); /* get DEC, OCT, or LIN */
        ptemp.iValue = 1;
        error = INPapName(ckt, which, acJob, steptype, &ptemp);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab);/* number of points*/
        error = INPapName(ckt, which, acJob, "numsteps", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstart */
        error = INPapName(ckt, which, acJob, "start", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstop */
        error = INPapName(ckt, which, acJob, "stop", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));

    }

    if (strcmp(token, "op") == 0) {
        which = ft_find_analysis("DCOP");
        if (which == -1) {
            current->error = INPerrCat
                (current->error,
                 INPmkTemp("DC operating point analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "dcop",
                                   & opJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createOP"); /* or similar error message */
            return (0);
        }
    }

    if (strcmp(token, "dc") == 0) {
        JOB *dcJob;
        /* .dc SRC1NAME Vstart1 Vstop1 Vinc1 [SRC2NAME Vstart2 */
        /*        Vstop2 Vinc2 */
        which = ft_find_analysis("DCTransfer");
        if (which == -1) {
            current->error = INPerrCat
                (current->error,
                 INPmkTemp("DC transfer curve analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "DCtransfer",
                                   & dcJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createOP"); /* or similar error message */
            return (0);
        }
        INPgetTok(&line, &name, 1);
        INPinsert(&name, tab);
        ptemp.uValue = name;
        error = INPapName(ckt, which, dcJob, "name1", &ptemp);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstart1 */
        error = INPapName(ckt, which, dcJob, "start1", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstop1 */
        error = INPapName(ckt, which, dcJob, "stop1", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vinc1 */
        error = INPapName(ckt, which, dcJob, "step1", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        if (*line) {
            if (*line == 'd')
                goto next;
            INPgetTok(&line, &name, 1);
            INPinsert(&name, tab);
            ptemp.uValue = name;
            error = INPapName(ckt, which, dcJob, "name2", &ptemp);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstart1 */
            error = INPapName(ckt, which, dcJob, "start2", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vstop1 */
            error = INPapName(ckt, which, dcJob, "stop2", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            parm = INPgetValue(ckt, &line, IF_REAL, tab); /* vinc1 */
            error = INPapName(ckt, which, dcJob, "step2", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
        }
    }

    if (strcmp(token, "tran") == 0) {
        JOB *tranJob;
        which = ft_find_analysis("TRAN");
        if (which == -1) {
            current->error = INPerrCat
                (current->error,
                 INPmkTemp("transient analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "tranan",
                                   & tranJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createTRAN");
            return (0);
        }

        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Tstep */
        error = INPapName(ckt, which, tranJob, "tstep", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Tstop*/
        error = INPapName(ckt, which, tranJob, "tstop", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        if (*line) {
            if (*line == 'd')
                goto next;
            if (*line == 'u')
                goto uic;
            parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Tstart */
            error = INPapName(ckt, which, tranJob, "tstart", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
            if (*line == 'u')
                goto uic;
            parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Tmax */
            error = INPapName(ckt, which, tranJob, "tmax", parm);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
        uic:
            if (*line == 'u') {
                INPgetTok(&line, &name, 1);
                if (strcmp(name, "uic") == 0) {
                    ptemp.iValue = 1;
                    error = INPapName(ckt, which, tranJob, "tstart", &ptemp);
                    if (error)
                        current->error = INPerrCat(current->error, INPerror(error));
                }
            }
        }
    }

#ifdef WITH_PSS
    /* *********************** */
    /* PSS - Spertica - 100910 */
    /* *********************** */
    if (strcmp(token, "pss") == 0) {
        JOB *pssJob;
        which = ft_find_analysis("PSS");
        if (which == -1) {
            current->error = INPerrCat
                (current->error,
                 INPmkTemp("periodic steady state analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis (ft_curckt->ci_ckt, which, "pssan",
                                   & pssJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createPSS");
            return (0);
        }

        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Guessed Frequency */
        error = INPapName(ckt, which, pssJob, "fguess", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));

        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* Stabilization time */
        error = INPapName(ckt, which, pssJob, "stabtime", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));

        parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* PSS points */
        error = INPapName(ckt, which, pssJob, "points", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));

        parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* PSS points */
        error = INPapName(ckt, which, pssJob, "harmonics", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
    }
#endif

#ifdef RFSPICE
    if (strcmp(token, "sp") == 0) {
        JOB* spJob;
        which = ft_find_analysis("SP");
        if (which == -1) {
            current->error = INPerrCat
            (current->error,
                INPmkTemp("S-Param analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis(ft_curckt->ci_ckt, which, "span",
            &spJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createSP"); /* or similar error message */
            return (0); /* temporary */
        }

        INPgetTok(&line, &steptype, 1); /* get DEC, OCT, or LIN */
        ptemp.iValue = 1;
        error = INPapName(ckt, which, spJob, steptype, &ptemp);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab);/* number of points*/
        error = INPapName(ckt, which, spJob, "numsteps", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstart */
        error = INPapName(ckt, which, spJob, "start", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstop */
        error = INPapName(ckt, which, spJob, "stop", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* fstop */
        error = INPapName(ckt, which, spJob, "donoise", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
    }
#ifdef WITH_HB
    if (strcmp(token, "hb") == 0) {
        JOB* spJob;
        which = ft_find_analysis("HB");
        if (which == -1) {
            current->error = INPerrCat
            (current->error,
                INPmkTemp("S-Param analysis unsupported\n"));
            return (0); /* temporary */
        }
        err = ft_sim->newAnalysis(ft_curckt->ci_ckt, which, "hban",
            &spJob, ft_curckt->ci_specTask);
        if (err) {
            ft_sperror(err, "createHB"); /* or similar error message */
            return (0); /* temporary */
        }

        INPgetTok(&line, &steptype, 1); /* get DEC, OCT, or LIN */
        ptemp.iValue = 1;
        error = INPapName(ckt, which, spJob, steptype, &ptemp);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab);/* number of points*/
        error = INPapName(ckt, which, spJob, "numsteps", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstart */
        error = INPapName(ckt, which, spJob, "start", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_REAL, tab); /* fstop */
        error = INPapName(ckt, which, spJob, "stop", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
        parm = INPgetValue(ckt, &line, IF_INTEGER, tab); /* fstop */
        error = INPapName(ckt, which, spJob, "donoise", parm);
        if (error)
            current->error = INPerrCat(current->error, INPerror(error));
    }
#endif
#endif

next:
    while (*line) { /* read the entire line */

        IFparm *if_parm;

        if (flag)
            INPgetTok(&line, &token, 1);
        else
            flag = 1;

        if_parm = ft_find_analysis_parm(save, token);

        if (!if_parm) {
            /* didn't find it! */
            current->error = INPerrCat
                (current->error,
                 INPmkTemp(" Error: unknown parameter on .sens - ignored \n"));
            continue;
        }

        /* found it, analysis which, parameter i */
        if (if_parm->dataType & IF_FLAG) {
            /* one of the keywords! */
            ptemp.iValue = 1;
            error = ft_sim->setAnalysisParm (ckt, senseJob,
                                             if_parm->id, &ptemp, NULL);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
        } else {
            parm = INPgetValue (ckt, &line, if_parm->dataType, tab);
            error = ft_sim->setAnalysisParm (ckt, senseJob,
                                             if_parm->id, parm, NULL);
            if (error)
                current->error = INPerrCat(current->error, INPerror(error));
        }
    }

    if ((err = ft_sim->doAnalyses (ckt, 1, ft_curckt->ci_curTask)) != OK) {
        ft_sperror(err, "doAnalyses");
        return (0); /* temporary */
    }

    return (0);
}
