/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 *
 * Print out in more detail what a floating point error was.
 */

#include "ngspice.h"
#include "cpdefs.h"
#include "ftedefs.h"
#include <signal.h>
#include "error.h"


/* global error message buffer */
char ErrorMessage[1024];



void
fperror(char *mess, int code)
{
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
#ifdef FTEDEBUG
#ifdef SIGQUIT
    (void) signal(SIGQUIT, SIG_DFL);
    (void) kill(getpid(), SIGQUIT);
#endif
#endif
    exit(EXIT_BAD);
}

/* These error messages are from internal consistency checks. */
void
internalerror(char *message)
{

    fprintf(stderr, "internal error:  %s\n", message);

}

/* These errors are from external routines like fopen. */
void
externalerror(char *message)
{

    fprintf(stderr, "external error:  %s\n", message);

}
