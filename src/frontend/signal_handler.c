/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * The signal routines for spice 3 and nutmeg.
 */


#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include "ftedev.h"
#include <setjmp.h>
#include <signal.h>
#include "signal_handler.h"




extern jmp_buf jbuf;

/* The (void) signal handlers... SIGINT is the only one that gets reset (by
 * cshpar) so it is global. They are ifdef BSD because of the sigmask
 * stuff in sigstop. We set the interrupt flag and return if ft_setflag
 * is TRUE.
 */


extern pid_t getpid (void);

/* not using SIGINT with GNU Readline - AV */
#ifndef HAVE_GNUREADLINE
RETSIGTYPE
ft_sigintr(void)
{

    gr_clean();

    (void) signal( SIGINT, (SIGNAL_FUNCTION) ft_sigintr );

    if (ft_intrpt)
        fprintf(cp_err, "Interrupt (ouch)\n");
    else {
        fprintf(cp_err, "Interrupt\n");
        ft_intrpt = TRUE;
    }
    if (ft_setflag)
        return;
/* To restore screen after an interrupt to a plot for instance
 */

    cp_interactive = TRUE;
    cp_resetcontrol();
    longjmp(jbuf, 1);
}
#endif /* !HAVE_GNUREADLINE */

RETSIGTYPE
sigfloat(int sig, int code)
{
    gr_clean();
    fperror("Error", code);
    rewind(cp_out);
    (void) signal( SIGFPE, (SIGNAL_FUNCTION) sigfloat );
    longjmp(jbuf, 1);
}

/* This should give a new prompt if cshpar is waiting for input.  */

#    ifdef SIGTSTP

RETSIGTYPE
sigstop(void)
{
    gr_clean();
    cp_ccon(FALSE);
    (void) signal(SIGTSTP, SIG_DFL);
    (void) kill(getpid(), SIGTSTP); /* This should stop us */
    return;
}

RETSIGTYPE
sigcont(void)
{
    (void) signal(SIGTSTP, (SIGNAL_FUNCTION) sigstop);
    if (cp_cwait)
        longjmp(jbuf, 1);
}

#    endif

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
    fatal();
}

RETSIGTYPE
sig_sys(void)
{
    fprintf(cp_err, 
        "\ninternal error -- bad argument to system call\n");
    fatal();
}


