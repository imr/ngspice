/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Jeffrey M. Hsu
**********/

/*
  This files contains the routines to evalute arguments to a command
  and prompt the user if necessary.
*/

#include "ngspice/ngspice.h"
#include "ngspice/fteinput.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"

#include "arg.h"
#include "variable.h"


static void common(char *string, struct wordlist *wl, struct comm *command);


/* returns a private copy of the string */

char *
prompt(FILE *fp)
{
    char    buf[100];
    char    *p;
    size_t  n;

    if (!fgets(buf, sizeof(buf), fp))
        return NULL;
    n = strlen(buf) - 1;
    buf[n] = '\0';      /* fgets leaves the \n */
    p = TMALLOC(char, n + 1);
    strcpy(p, buf);
    return p;
}


int
countargs(wordlist *wl)
{
    int number = 0;
    wordlist *w;

    for (w = wl; w; w = w->wl_next)
        number++ ;

    return (number);
}


wordlist *
process(wordlist *wlist)
{
    wlist = cp_variablesubst(wlist);
    wlist = cp_bquote(wlist);
    wlist = cp_doglob(wlist);
    return (wlist);
}


void
arg_print(wordlist *wl, struct comm *command)
{
    common("which variable", wl, command);
}


void
arg_plot(wordlist *wl, struct comm *command)
{
    common("which variable", wl, command);
}


void
arg_load(wordlist *wl, struct comm *command)
{
    /* just call com_load */
    command->co_func(wl);
}


void arg_let(wordlist *wl, struct comm *command)
{
    common("which vector", wl, command);
}


void
arg_set(wordlist *wl, struct comm *command)
{
    common("which variable", wl, command);
}


void
arg_display(wordlist *wl, struct comm *command)
{
    NG_IGNORE(wl);
    NG_IGNORE(command);

    /* just return; display does the right thing */
}


/* a common prompt routine */
static void
common(char *string, struct wordlist *wl, struct comm *command)
{
    struct wordlist *w;
    char *buf;

    if (!countargs(wl)) {
        outmenuprompt(string);
        if ((buf = prompt(cp_in)) == NULL) /* prompt aborted */
            return;               /* don't execute command */
        /* do something with the wordlist */
        w = wl_cons(buf, NULL);

        w = process(w);
        /* O.K. now call fn */
        command->co_func(w);
    }
}


void
outmenuprompt(char *string)
{
    fprintf(cp_out, "%s: ", string);
    fflush(cp_out);
    return;
}
