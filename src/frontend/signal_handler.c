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

#ifdef HAS_WINGUI
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


/* The purpose of ft_sigintr_cleanup() is to handle all processing of asynchronous
 * signals which require user process context. Some kernel services are not
 * allowed to be called from asynchronous signal handlers. (e.g. mutexes)
 */

void
ft_sigintr_cleanup(void)
{
    gr_clean();  /* Clean up plot window */

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
    cp_resetcontrol(TRUE);
}


/*  invoke this function upon keyboard interrupt  */
void
ft_sigintr(void)
{
    static int interrupt_counter = 0;

    /* fprintf(cp_err, "Received interrupt.  Handling it  . . . . .\n"); */

    /* Reinstall ft_signintr as the signal handler. */
    (void) signal(SIGINT, (SIGNAL_FUNCTION) ft_sigintr);

    if (ft_intrpt) {    /* check to see if we're being interrupted repeatedly */
        fprintf(cp_err, "\nInterrupted again (ouch)\n");
        interrupt_counter++;
    } else {
        fprintf(cp_err, "\nInterrupted once . . .\n");
        ft_intrpt = TRUE;
        interrupt_counter = 1;
    }

    if (interrupt_counter >= 3) {
        fprintf(cp_err, "\nKilling, since %d interrupts have been requested\n\n", interrupt_counter);
        controlled_exit(1);
    }

    if (ft_setflag) {
        return;     /* just return without aborting simulation if ft_setflag = TRUE */
    }

    /* here we jump to the start of command processing in main() after resetting everything.  */
    LONGJMP(jbuf, 1);
}


void
sigfloat(int code)
{
    fperror("Error", code);
    rewind(cp_out);
    (void) signal(SIGFPE, (SIGNAL_FUNCTION) sigfloat);
    LONGJMP(jbuf, 1);
}


/* This should give a new prompt if cshpar is waiting for input.  */

#ifdef SIGTSTP

void
sigstop(void)
{
    gr_clean();
    cp_ccon(FALSE);
    (void) signal(SIGTSTP, SIG_DFL);
    (void) kill(getpid(), SIGTSTP); /* This should stop us */
}


void
sigcont(void)
{
    (void) signal(SIGTSTP, (SIGNAL_FUNCTION) sigstop);
    if (cp_cwait)
        LONGJMP(jbuf, 1);
}


#endif


/* Special (void) signal handlers. */

void
sigill(void)
{
    fprintf(cp_err, "\ninternal error -- illegal instruction\n");
    fatal();
}


void
sigbus(void)
{
    fprintf(cp_err, "\ninternal error -- bus error\n");
    fatal();
}


void
sigsegv(void)
{
    fprintf(cp_err, "\ninternal error -- segmentation violation\n");
#ifdef HAS_WINGUI
    winmessage("Fatal error in NGSPICE");
#endif
    fatal();
}

void
sigsegvsh(void)
{
    fprintf(cp_err, "\ninternal error -- segmentation violation\n");
    controlled_exit(EXIT_SEGV);
}


void
sig_sys(void)
{
    fprintf(cp_err, "\ninternal error -- bad argument to system call\n");
    fatal();
}
