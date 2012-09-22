/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * The signal routines for spice 3 and nutmeg.
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ftedev.h"
#include <setjmp.h>
#include <signal.h>
#include "signal_handler.h"
#include "plotting/graf.h"

#ifdef HAS_WINDOWS
void winmessage(char* new_msg);
#endif

#ifdef HAVE_GNUREADLINE
/* Added GNU Readline Support 11/3/97 -- Andrew Veliath <veliaa@rpi.edu> */
/* from spice3f4 patch to ng-spice. jmr */
#include <readline/readline.h>
#include <readline/history.h>
#endif

#ifdef HAVE_BSDEDITLINE
/* SJB added edit line support 2005-05-05 */
#include <editline/readline.h>
#endif /* HAVE_BSDEDITLINE */

JMP_BUF jbuf;

/* The (void) signal handlers... SIGINT is the only one that gets reset (by
 * cshpar) so it is global. They are ifdef BSD because of the sigmask
 * stuff in sigstop. We set the interrupt flag and return if ft_setflag
 * is TRUE.
 */

/*  invoke this function upon keyboard interrupt  */
RETSIGTYPE
ft_sigintr(void)
{
    /* fprintf(cp_err, "Received interrupt.  Handling it  . . . . .\n"); */

    /* Reinstall ft_signintr as the signal handler. */
    (void) signal(SIGINT, (SIGNAL_FUNCTION) ft_sigintr);

    gr_clean();  /* Clean up plot window */

    if (ft_intrpt) {    /* check to see if we're being interrupted repeatedly */
        fprintf(cp_err, "\nInterrupted again (ouch)\n");
    } else {
        fprintf(cp_err, "\nInterrupted once . . .\n");
        ft_intrpt = TRUE;
    }

    if (ft_setflag) {
        return;     /* just return without aborting simulation if ft_setflag = TRUE */
    }

    /* sjb - what to do for editline???
       The following are not supported in editline */
#if defined(HAVE_GNUREADLINE)
    /*  Clean up readline after catching signals  */
    /*  One or all of these might be superfluous  */
    (void) rl_free_line_state();
    (void) rl_cleanup_after_signal();
    (void) rl_reset_after_signal();
#endif /* defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE) */

    /* To restore screen after an interrupt to a plot for instance */
    cp_interactive = TRUE;
    cp_resetcontrol();

    /* here we jump to the start of command processing in main() after resetting everything.  */
    LONGJMP(jbuf, 1);
}


RETSIGTYPE
sigfloat(int sig, int code)
{
    NG_IGNORE(sig);

    gr_clean();
    fperror("Error", code);
    rewind(cp_out);
    (void) signal(SIGFPE, (SIGNAL_FUNCTION) sigfloat);
    LONGJMP(jbuf, 1);
}


/* This should give a new prompt if cshpar is waiting for input.  */

#ifdef SIGTSTP

RETSIGTYPE
sigstop(void)
{
    gr_clean();
    cp_ccon(FALSE);
    (void) signal(SIGTSTP, SIG_DFL);
    (void) kill(getpid(), SIGTSTP); /* This should stop us */
}


RETSIGTYPE
sigcont(void)
{
    (void) signal(SIGTSTP, (SIGNAL_FUNCTION) sigstop);
    if (cp_cwait)
        LONGJMP(jbuf, 1);
}


#endif


/* Special (void) signal handlers. */

RETSIGTYPE
sigill(void)
{
    fprintf(cp_err, "\ninternal error -- illegal instruction\n");
    fatal();
}


RETSIGTYPE
sigbus(void)
{
    fprintf(cp_err, "\ninternal error -- bus error\n");
    fatal();
}


RETSIGTYPE
sigsegv(void)
{
    fprintf(cp_err, "\ninternal error -- segmentation violation\n");
#ifdef HAS_WINDOWS
    winmessage("Fatal error in NGSPICE");
#endif
    fatal();
}


RETSIGTYPE
sig_sys(void)
{
    fprintf(cp_err, "\ninternal error -- bad argument to system call\n");
    fatal();
}
