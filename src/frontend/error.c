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


/* global error message buffer */
char ErrorMessage[1024];


#ifdef HAS_WINDOWS
void winmessage(char* new_msg);
#endif


void
controlled_exit(int status)
{
#ifdef HAS_WINDOWS
    if (status)
        winmessage("Fatal error in NGSPICE");
#else
    if (status)
        fprintf(stderr, "\nERROR: fatal error in ngspice, exit(%d)\n", status);
#endif
    exit(status);
}


void
fperror(char *mess, int code)
{
    NG_IGNORE(code);
    fprintf(cp_err, "%s: floating point exception.\n", mess);
    return;
}


/* Print a spice error message. */
void
ft_sperror(int code, char *mess)
{
    fprintf(cp_err, "%s: %s\n", mess, if_errstring(code));
    return;
}


void
fatal(void)
{
    cp_ccon(FALSE);

#if defined(FTEDEBUG) && defined(SIGQUIT)
    (void) signal(SIGQUIT, SIG_DFL);
    (void) kill(getpid(), SIGQUIT);
#endif

    exit(EXIT_BAD);
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
