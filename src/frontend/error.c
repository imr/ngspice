/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Print out in more detail what a floating point error was.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include <signal.h>


#ifdef HAS_WINGUI
void winmessage(char *new_msg);
extern void  UpdateMainText(void);
#elif defined SHARED_MODULE
extern ATTRIBUTE_NORETURN void shared_exit(int status);
#endif

/* global error message buffer */
char ErrorMessage[1024];


ATTRIBUTE_NORETURN void
controlled_exit(int status)
{
#ifdef HAS_WINGUI
    if (status) {
        UpdateMainText(); /* get any remaining error messages into main text window */
        winmessage("Fatal error in NGSPICE");
    }
    exit(status);
#elif defined SHARED_MODULE
    /* do not exit, if shared ngspice, but call back */
    shared_exit(status);
#else
    if (status)
        fprintf(stderr, "\nERROR: fatal error in ngspice, exit(%d)\n", status);
    exit(status);
#endif
}


void
fperror(char *mess, int code)
{
    NG_IGNORE(code);
    fprintf(cp_err, "%s: floating point exception.\n", mess);
}


/* Print a spice error message. */
void
ft_sperror(int code, char *mess)
{
    char *errstring = if_errstring(code);
    fprintf(cp_err, "%s: %s\n", mess, errstring);
    tfree(errstring);
}


void
fatal(void)
{
    cp_ccon(FALSE);

#if defined(FTEDEBUG) && defined(SIGQUIT)
    (void) signal(SIGQUIT, SIG_DFL);
    (void) kill(getpid(), SIGQUIT);
#endif

#if defined SHARED_MODULE
    /* do not exit, if shared ngspice, but call back */
    shared_exit(EXIT_BAD);
#else
    exit(EXIT_BAD);
#endif
}


/* These error messages are from internal consistency checks. */
void
internalerror(char *message)
{
    fprintf(stderr, "ERROR: (internal)  %s\n", message);
}


/* These errors are from external routines like fopen. */
void
externalerror(char *message)
{
    fprintf(stderr, "ERROR: (external)  %s\n", message);
}
